'''
This module contains the main RAG QA logic, as well as the auxiliary 
functions to support it.
'''

import os
import re
import asyncio
from functools import lru_cache
import logging
from typing import List, Set, Union, Dict, Optional, Any, Callable

# Data processing
import numpy as np
import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Models and scoring
from bert_score import score as bert_score
from rank_bm25 import BM25Okapi
from transformers import pipeline, MarianMTModel, MarianTokenizer

# LangChain components
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

import config as cfg

# ------------------------------------------------------------------ #
#                           Global Vars                              #
# ------------------------------------------------------------------ #
# MARK: Global Vars

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Low confidence keyword embeddings, used to penalize certain answers 
_LOW_CONF_EMB = None

LOW_CONFIDENCE_KEYWORDS = [
    "I don't know.", "I'm not sure about that.", "Sorry, I can't help with that.",
    "No information available.", "insufficient evidence", "lacking data", "unclear"
]

# ------------------------------------------------------------------ #
#                         Evaluation Metrics                         #
# ------------------------------------------------------------------ #
# MARK: Evaluation Metrics

def rank_and_filter_sources(embeddings, query_embedding, retrieved_docs, 
                            threshold=0.3, top_k=5):
    """
    Ranks and filters retrieved documents, then thresholding is applied 
    to retrieve the top_k documents.
    """
    scored = []
    seen_sources = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "")
        if source in seen_sources:
            continue
        emb = embeddings.embed_query(doc.page_content)
        score = float(cosine_similarity([query_embedding], [emb])[0][0])
        if score > threshold:
            scored.append((doc, score, emb))
            seen_sources.add(source)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def compute_confidence(answer_emb, doc_embs, embeddings=None, bert_f1=0.0, top_k=5):
    """
    Computes confidence score for the answer by combining multiple evaluation
    metrics.
    """
    global _LOW_CONF_EMB
    if _LOW_CONF_EMB is None and embeddings is not None:
        _LOW_CONF_EMB = embeddings.embed_documents(LOW_CONFIDENCE_KEYWORDS)

    # Low confidence keyword penalty
    kw_sims = [
        np.dot(answer_emb, kw) / (np.linalg.norm(answer_emb) * np.linalg.norm(kw))
        for kw in _LOW_CONF_EMB
    ]
    exp_kw = np.exp(kw_sims)
    penalty = 1.0 - (np.sum(kw_sims * exp_kw) / np.sum(exp_kw))
    keyword_conf = float(np.clip(penalty, 0.0, 1.0))

    # Document alignment
    if doc_embs:
        sims = [
            np.dot(answer_emb, demb) / (np.linalg.norm(answer_emb) * np.linalg.norm(demb))
            for demb in doc_embs]
        top_sims = sorted(sims, reverse=True)[:top_k]
        doc_conf = float(np.max(top_sims)) if top_sims else 0.0 
    else:
        doc_conf = 0.0

    confidence = 0.3 * keyword_conf + 0.2 * doc_conf + 0.5 * bert_f1 

    logging.debug(f"Confidence: kw={keyword_conf:.2f}, doc={doc_conf:.2f}, bert_f1={bert_f1:.2f}, combined={confidence:.2f}")
    return confidence

def compute_bert_score(candidates, references, lang="en"):
    """
    Computes BERTScore between answer (candidates) and the previously retrieved
    top_k documents (references).
    """
    try:
        P, R, F1 = bert_score(
        cands=candidates,
        refs=references,
        lang=lang,
        device="cuda" if torch.cuda.is_available() else "cpu",
        rescale_with_baseline=False)
    except Exception  as e:
        logging.warning(f"BERTScore failed: {e}")
        return {"precision": [0.0], "recall": [0.0], "f1": [0.0], "mean_f1": 0.0}
    
    f1_list = F1.tolist()
    mean_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    logging.debug(f"BERTScore mean_f1={mean_f1:.2f}")
    return {"precision": P.tolist(), "recall": R.tolist(), "f1": f1_list, "mean_f1": mean_f1}

def compute_recall_at_k(retrieved_ids, relevant_ids, top_k):
    """
    Computes recall@k for the retrieved documents of the answer.
    """
    if top_k <= 0 or not relevant_ids or not retrieved_ids:
        logging.debug(f"Recall@{top_k}: invalid inputs (retrieved={len(retrieved_ids)}, relevant={len(relevant_ids)})")
        return 0.0
    relevant_set = set(relevant_ids)
    top_k_ids = retrieved_ids[:min(top_k, len(retrieved_ids))]
    hits = sum(1 for rid in top_k_ids if rid in relevant_set)
    recall = hits / len(relevant_set)
    logging.debug(f"Recall@{top_k}: hits={hits}, recall={recall:.2f}")
    return min(recall,1)

def is_informative(answer, confidence, threshold=0.3, min_words=10, vague_keywords=None):
    """
    Checks if the answer is informative based on the confidence and length of the answer.
    This is used to filter out low quality answers and inform the user during the conversation.
    """
    if not answer or confidence < threshold:
        return False
    if len(answer.split()) < min_words:
        return False
    # vague = vague_keywords or ["some", "various", "many"]
    # if any(v in answer.lower() for v in vague):
    #     return False
    return True

# ------------------------------------------------------------------ #
#                              QA Chain                              #
# ------------------------------------------------------------------ #
# MARK: QA Chain

def clean_output(text, question):
    """
    Filters the language model output to remove excessively 
    verbose answers. 
    """

    # NOTE: Depending on the model used and the library api, the models
    # sometime returns unnecessary information, such as repeated questions
    # or non-useful context. To avoid using this information we regex 
    # remove it.
    patterns = [
        r"Context:.*?(?=\nAnswer:|\Z)",
        r"Question:.*?(?=\nAnswer:|\Z)",
        r"Answer:\s*",
        r"\n\s*\n+",
        re.escape(question),
        r"^\s+|\s+$",
    ]
    cleaned = text.strip()
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = cleaned.split(". ")
    unique_sentences = []
    for s in sentences:
        if s.strip() and s.strip() not in unique_sentences:
            unique_sentences.append(s.strip())
    result = ". ".join(unique_sentences)
    if result and not result.endswith("."):
        result += "."
    return result

def deduplicate_documents(retrieved_documents):
    """
    Sometimes the mode returns duplicated documents, to avoid this
    simply filter them out.
    """
    logger.debug("Deduplicating documents")
    texts = [doc.page_content for doc in retrieved_documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)
    unique_indices = []
    for i in range(len(texts)):
        if all(sim_matrix[i, j] < 0.95 for j in unique_indices):
            unique_indices.append(i)
    return [retrieved_documents[i] for i in unique_indices]

async def load_documents_async():
    """
    Loads the documents from the corpus file and returns them as a list of Document objects.
    """

    logger.debug(f"Checking for corpus at {cfg.CORPUS_PATH}")
    if not os.path.exists(cfg.CORPUS_PATH):
        logger.error(f"Corpus file not found: {cfg.CORPUS_PATH}")
        raise FileNotFoundError(f"Corpus file not found: {cfg.CORPUS_PATH}")
    try:
        with open(cfg.CORPUS_PATH, "r", encoding="utf-8", errors="replace") as f:
            text = await asyncio.get_event_loop().run_in_executor(None, f.read)
        docs = []
        # Split by every occurrence of "Link:" using a positive lookahead so all entries are preserved.
        entries = re.split(r"(?=Link:)", text.strip())
        logger.debug(f"Found {len(entries)} entries in corpus")
        for i, entry in enumerate(entries):
            link_match = re.search(r"Link:\s*(\S+)", entry)
            abstract_match = re.search(r"Abstract:\s*(.+)", entry, re.DOTALL)
            if link_match and abstract_match:
                link = link_match.group(1).strip()
                abstract = abstract_match.group(1).strip()
                docs.append(Document(page_content=abstract, metadata={"source": link}))
            else:
                logger.warning(f"Invalid entry {i}: missing link or abstract")
            if i % 50 == 0:
                logger.debug(f"Processed {i} entries")
        if not docs:
            logger.error("No valid Link/Abstract pairs found in the corpus")
            raise ValueError("No valid Link/Abstract pairs found in the corpus")
        logger.debug(f"Loaded {len(docs)} documents")
        # Instead of deduplicating, we simply return the documents up to MAX_DOCS.
        return docs[:cfg.MAX_DOCS]
    except Exception as e:
        logger.error(f"Error loading documents: {e}", exc_info=True)
        raise

def load_faiss_index(docs, embeddings, index_path):
    """
    Loads or creates a FAISS index for the given documents and embeddings.
    """

    if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
        logger.debug("Loading existing FAISS index")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    logger.debug("Creating new FAISS index")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = splitter.split_documents(docs)
    logger.debug(f"Split into {len(split_docs)} chunks")
    
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    vectorstore.save_local(index_path)
    return vectorstore

def load_bm25_index(docs, index_path):
    """
    Loads or creates a BM25 index for the given documents.
    """
    if os.path.exists(index_path):
        logger.debug("Loading existing BM25 index")
        with open(index_path, "rb") as f:
            return pickle.load(f)

    logger.debug("Creating new BM25 index")
    documents = [doc.page_content for doc in docs]
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(bm25_index, f)
    
    return bm25_index

def get_qa_chain(model="qwen2:0.5b", temperature=0.5):
    """
    Initializes and returns the QA chain.
    """

    logger.debug("Starting QA chain initialization")

    # NOTE: If this is problematic, set the device to "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        docs = asyncio.run(load_documents_async())

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 32}
        )

        vectorstore = load_faiss_index(docs, embeddings, cfg.FAISS_INDEX_PATH)
        _ = load_bm25_index(docs, cfg.BM25_INDEX_PATH)

        bm25_retriever = BM25Retriever.from_documents(docs, search_kwargs={"k": 10})
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

        try:
            llm = Ollama(
                model=model,
                base_url="http://localhost:11434",
                temperature=temperature,
                num_predict=200
            )
        except Exception as e:
            logger.error(f"Failed to load Ollama model: {e}")
            raise

        prompt = PromptTemplate.from_template("""
            You are a nutrition expert. Answer the question using only the provided context, focusing on specific details (e.g., nutrients, health effects). 
            Do not add external knowledge, assumptions, or unrelated information. Avoid vague phrases like "generally healthy" or "can be healthy." 
            If the context is irrelevant, insufficient, or does not directly address the question's subject (e.g., specific food or preparation method), state exactly: "Insufficient evidence to determine the health impact." 
            For complex dishes, focus only on components explicitly mentioned in the context. 
            Do not contradict established nutritional knowledge (e.g., fruits like pears and vegetables are typically healthy unless specific evidence suggests otherwise).
            You are not allowed to repeat the same answer over and over (e.g., fruits are healthy, fruits are healthy).
            Return only the answer.

            Context:
            {context}

            Question:
            {input}

            Answer:
        """)
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(hybrid_retriever, combine_docs_chain)

        logger.debug("QA chain initialized successfully")
        return qa_chain, embeddings

    except Exception as e:
        logger.error("Error initializing QA chain", exc_info=True)
        raise

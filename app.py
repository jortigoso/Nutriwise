import os
import re
import time
import random
import logging
import traceback
from datetime import datetime

import numpy as np
import fasttext
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import argparse

import config as cfg
import rag_qa
import translation

# ------------------------------------------------------------------ #
#                          Utility functions                         #
# ------------------------------------------------------------------ #
# MARK: Utility functions

def robust_invoke(qa_chain, query, retries=3, base_delay=2.0):
    """
    Wrapper for qa_chain.invoke(), with retries and exponential backoff
    to handle temporary unavailability of the model.
    """
    for i in range(retries):
        try:
            return qa_chain.invoke({"input": query})
        except Exception as e:
            if "503" in str(e) and i < retries - 1:
                wait = base_delay * (2 ** i) + random.random()
                logging.warning(f"503 on attempt {i+1}, retrying in {wait:.2f}s")
                time.sleep(wait)
            else:
                logging.error(f"Invocation error: {e}")
                raise
    raise RuntimeError("Model unavailable after retries")

def translate(text, source_lang, target_lang):
    """
    Translation wrapper function for use in the app with some more
    error handling.
    """
    if not text or not text.strip():
        return ""
    
    translated_text = translation.translate_text(
        text, source_lang=source_lang, target_lang=target_lang
    )
    return translated_text

def generate_summary(qa_chain, text, min_words=15, max_words=35):
    """
    Generates a summary using the LLM model directly.
    """
    try:
        summary_prompt = f"""
        Create a very concise summary of the following text.
        Requirements:
        - Use between {min_words} and {max_words} words.
        - Synthesize the main points into new sentences.
        - DO NOT copy any full sentences from the original text.
        - Focus on the key findings or conclusions.
        - Be specific and direct.
        - Avoid phrases like "generally" or "studies show".
        - If health effects are mentioned, state them clearly.

        Text to summarize:
        {text}

        Write a brief, synthesized summary:
        """
        logging.debug('GENERANDO RESUMEN')
        result = robust_invoke(qa_chain, summary_prompt)
        summary = result.get("answer", "").strip()
        
        # Clean up formatting
        summary = re.sub(r'\s+', ' ', summary)
        summary = re.sub(r'[,\s]+\.', '.', summary)
        summary = re.sub(r'\.+', '.', summary)
        summary = re.sub(r'^(In summary|Overall|To summarize|Research shows|Studies indicate)[,:\s]+', '', summary, flags=re.IGNORECASE)
        
        # Count words and trim if necessary
        words = summary.split()
        if len(words) > max_words:
            sentences = summary.split('.')
            trimmed = []
            word_count = 0
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_words = len(sent.split())
                if word_count + sent_words <= max_words:
                    trimmed.append(sent)
                    word_count += sent_words
                else:
                    break
            summary = '. '.join(trimmed) + '.'
        
        # Final cleanup
        if summary:
            summary = summary[0].upper() + summary[1:]
            if not summary.endswith('.'):
                summary += '.'
            summary = re.sub(r'\.+', '.', summary)
        return summary
    except Exception as e:
        logging.error(f"Summary generation failed: {e}")
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        fallback = '. '.join(sentences[:2])
        fallback_words = fallback.split()[:max_words]
        fallback_summary = ' '.join(fallback_words).strip()
        if fallback_summary and not fallback_summary.endswith('.'):
            fallback_summary += '.'
        return fallback_summary
    
def generate_fact_checked_answer(qa_chain, query, response_text, min_words=15, max_words=200):
    """
    Generate an original answer that is directly related to the input query.
    For queries about a given food, fact-check the claim and present a longer,
    detailed answer strictly addressing that given food.
    The answer should:
      - Directly reference the query topic (the given food) without introducing artifacts.
      - Clearly state a practical conclusion about the health impact of the given food based on the evidence.
      - Provide an extended explanation with fact-checking details.
    The final answer is between min_words and max_words.
    """
    try:
        answer_prompt = f"""
        You are a skilled nutrition and health expert. The input query is:
        "{query}"
        Based on the following evidence:
        {response_text}
        Provide an extended and original answer that fact-checks the claim about that given food health effects.
        Ensure that your answer directly focuses on that given food. Do not include any unrelated topics.
        Elaborate on whether that given food is healthy or not, citing its potential benefits and risks.
        Write a detailed answer between {min_words} and {max_words} words.
        """
        result = robust_invoke(qa_chain, answer_prompt)
        answer = result.get("answer", "").strip()
        words = answer.split()
        if len(words) < min_words:
            extended_prompt = answer_prompt + "\nPlease add more details and ensure the answer focuses on that given food."
            result_ext = robust_invoke(qa_chain, extended_prompt)
            answer = result_ext.get("answer", "").strip()
            words = answer.split()
        if len(words) > max_words:
            answer = ' '.join(words[:max_words])
            if not answer.endswith('.'):
                answer += '.'
        answer = answer[0].upper() + answer[1:]
        return answer
    except Exception as e:
        logging.error(f"Fact-checked answer generation failed: {e}")
        fallback = generate_summary(response_text, min_words, max_words)
        return fallback
    
# ------------------------------------------------------------------ #
#                             Application                            #
# ------------------------------------------------------------------ #

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open(os.path.join("frontend", "index.html"), encoding="utf-8", errors="replace") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        logging.error(f"Serve error: {e}")
        return HTMLResponse(content="Error loading page", status_code=500)


def create_query_handler(qa_chain, embeddings, fasttext_model):
    @app.post("/query")
    async def handle_query(request: Request):
        try:
            response = {
                "answer": None,
                "confidence": 0.0,
                "sources": [],
                "summary": "",
                "metrics": {},
                "lang": "en"
            }
            data = await request.json()
            query = data.get("question", "").strip()
            if not query:
                return JSONResponse(response, status_code=400)

            src_lang = translation.detect_language(fasttext_model, query)
            if src_lang not in translation.SUPPORTED_LANGUAGES:
                msg = translate("Unsupported language. Use English or a supported language.", "en", src_lang)
                response["answer"] = msg
                response["lang"] = src_lang
                return JSONResponse(response, status_code=400)

            query_en = query if src_lang == "en" else translate(query, src_lang, "en")
            q_emb = embeddings.embed_query(query_en)
            result = robust_invoke(qa_chain, query_en)
            answer_text = result.get("answer", "")
            retrieved_docs = result.get("context", [])

            TOP_K = 5
            relevant_sources = rag_qa.rank_and_filter_sources(
                embeddings, q_emb, retrieved_docs, threshold=0.3, top_k=TOP_K
            )

            top_src_links = []
            seen_links = set()
            top_src_txts = []
            top_src_embs = []
            top_src_ids = []
            for idx, (doc, score, emb) in enumerate(relevant_sources):
                link = doc.metadata.get("source", "")
                if link not in seen_links:
                    top_src_links.append(link)
                    top_src_txts.append(doc.page_content)
                    top_src_embs.append(emb)
                    top_src_ids.append(doc.metadata.get("source", f"doc_{idx}"))
                    seen_links.add(link)
                    if len(top_src_links) >= TOP_K:
                        break

            final_answer = generate_fact_checked_answer(qa_chain, query_en, answer_text)
            summary_en = generate_summary(qa_chain, answer_text) if answer_text else ""

            bert_score = rag_qa.compute_bert_score([answer_text] * len(top_src_txts), top_src_txts, lang="en")
            bert_f1 = bert_score["mean_f1"] or 0.0
            recall_at_k = rag_qa.compute_recall_at_k(top_src_links, top_src_ids, top_k=TOP_K)
            answer_emb = embeddings.embed_query(answer_text)
            confidence = rag_qa.compute_confidence(answer_emb, top_src_embs, embeddings, bert_f1, TOP_K)

            if bert_f1 < 0.3 and confidence < 0.5:
                answer_text = "Sorry, there is insufficient evidence to respond to this."
                summary_en = ""
                top_src_links = []
                confidence = 0.0
                bert_f1 = 0.0

            answer_final = final_answer if src_lang == "en" else translate(final_answer, "en", src_lang)
            summary_final = summary_en if src_lang == "en" else translate(summary_en, "en", src_lang)

            if not rag_qa.is_informative(answer_final, confidence=confidence):
                apology = "Sorry, I cannot provide an informative answer."
                try:
                    final = apology if src_lang == "en" else translate(apology, "en", src_lang)
                except Exception as e:
                    logging.error(f"Translation of apology failed: {e}")
                    final = apology
                response["answer"] = final
                response["confidence"] = 0.0
                response["metrics"] = {"recall_at_k": 0.0, "bert_score_f1": 0.0}
                response["lang"] = src_lang
            else:
                response["answer"] = answer_final
                response["confidence"] = confidence
                response["sources"] = top_src_links
                response["summary"] = summary_final
                response["metrics"] = {"recall_at_k": recall_at_k, "bert_score_f1": bert_f1}
                response["lang"] = src_lang

            return JSONResponse(response, status_code=200)
        except Exception as e:
            logging.error(f"Query error: {e}\n{traceback.format_exc()}")
            return JSONResponse(
                {"error": str(e), "answer": None, "confidence": 0.0, "sources": [], "summary": "", "metrics": {}, "lang": "en"},
                status_code=500
            )
    return handle_query
    
# ------------------------------------------------------------------ #
#                                Main                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", "-m", type=str, default="qwen2:0.5b")
    argparser.add_argument("--temperature", "-t", type=float, default=0.5)
    # argparser.add_argument("--debug", action="store_true")
    # logging.basicConfig(level=logging.DEBUG if cfg.debug else logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load fast text model for language detection
    try:
        translation.download_fasttext_model(cfg.FT_MODEL_PATH)
        fasttext_model = fasttext.load_model(cfg.FT_MODEL_PATH)
        logging.info("Loaded fasttext language identification model.")
    except Exception as e:
        logging.error(f"Failed to load fasttext model: {e}")
        ft_model = None
        raise
    
    # Load the translation models and the QA chain
    try:
        qa_chain, embeddings = rag_qa.get_qa_chain()
        translation.preload_models([
            ("es", "en"), ("en", "es"),
            ("fr", "en"), ("en", "fr"),
            ("de", "en"), ("en", "de"),
            ("it", "en"), ("en", "it"),
            ("ru", "en"), ("en", "ru"),
            ("ko", "en"), ("en", "ko")
        ])
    except Exception as init_err:
        logging.error(f"Initialization failed: {init_err}")
        raise

    # Register the handler
    create_query_handler(qa_chain, embeddings, fasttext_model)

    uvicorn.run(app, host="127.0.0.1", port=5000)
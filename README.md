# Welcome to Nutriwise! 🌿

Nutriwise is a fact-checking system that verifies dietary and nutritional claims using a curated database of scientific articles and educational books. It provides accurate, multilingual responses grounded in evidence from PubMed abstracts and nutrition literature.

## Authors 👥

- **Lidia Jiménez Algarra**  
- **Irene Mancebo Laguna**  
- **Jorge Ortigoso Narro**  
- **Guillermo Rey Paniagua**  
Master’s in Machine Learning for Health  Universidad Carlos III de Madrid, 2025

## 1. Features

## ✅ System Capabilities

This application implements a **Retrieval-Augmented Generation (RAG) fact-checking system** designed to automatically assess the truthfulness of user-submitted statements using a curated document corpus (e.g., PubMed, Wikipedia, academic articles). The system fulfills the following functional requirements:

- **Verifies factual statements** using evidence retrieved from the knowledge base and assessed by a large language model (LLM).
- **Provides justifications** for its answers by citing relevant source documents and excerpts from the corpus.
- **Does not fabricate answers**: when the system lacks sufficient evidence to verify a claim, it clearly responds with:  
  **“I'm sorry, I do not have enough information in the corpus to verify this claim.”**
- **Supports multilingual interaction**: responses are generated in the same language as the user’s question (supports English, Spanish, French, German, Italian).

## ✨ Additional Features

To enhance usability, performance, and transparency, the system also includes:

- **Local LLM integration** via Ollama (e.g., `qwen2:0.5b`) to generate reliable, context-aware answers.
- **Hybrid document retrieval** combining FAISS (dense vector search) and BM25 (sparse keyword-based retrieval) for improved relevance.
- **Automatic translation** of queries and responses using pre-trained Helsinki-NLP models for seamless multilingual support.
- **Answer summarization**: generates concise, informative summaries using prompt-engineered LLM instructions.
- **Confidence scoring** based on a combination of BERTScore, cosine similarity, and detection of vague expressions.
- **Evaluation module**: includes benchmarking tools to compare models and settings using metrics like ROUGE-L, Recall@k, BERT F1, and latency.
- **Bias reduction heuristics** in translation, e.g., neutralizing gendered phrasing when translating back from non-English queries.
- **Includes a web-based user interface** that allows users to input questions and receive real-time answers with sources and confidence metrics.

## 2. Implementation

### Dataset 🧾

The dataset consists of:

- Over **20,000 scientific abstracts** from PubMed filtered for human nutrition relevance.
- **15 educational nutrition books**, segmented by pages and converted to `.txt` format.

All texts were preprocessed for uniform structure and clean integration.

### Vector Database 🛢️

- **FAISS** is used for semantic search with 384-dimensional dense vectors.
- **BM25Okapi** is integrated for sparse keyword matching.
- Document segmentation is done by page; top 20 chunks are retrieved and top 5 are re-ranked.
- Cosine similarity threshold of 0.3 ensures relevant retrievals.

### Models 🧠

Nutriwise integrates several models:

- `paraphrase-MiniLM-L6-v2` for embeddings.
- `qwen2:0.5b` (Ollama) for local answer generation.
- `Helsinki-NLP` for multilingual translation.
- `fastText lid.176.bin` (Facebook) for language detection.
- `BERTScore` for evaluating answer quality.


## 3. File Overview
| File/Folder            | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `app.py`               | Starts the FastAPI backend and defines the main API endpoints.              |
| `config.py`            | Configuration file with paths to corpus, indices, and model files.          |
| `evaluate.py`          | Runs benchmarks and computes metrics like BERTScore, ROUGE, and latency.    |
| `rag_qa.py`            | Core logic for retrieval, QA chain setup, scoring, and RAG mechanisms.      |
| `translation.py`       | Handles language detection and translation between supported languages.     |
| `frontend/`            | Contains static HTML files for the web-based user interface.                |


## 4. Setup Guide ⚙️

To install and run Nutriwise locally, follow these steps:

1. Clone the repostory
```
git clone https://github.com/jortigoso/Nutriwise.git
```
2. Move to the project directory
```
cd Nutriwise
```
3. Add execution permissions
```
chmod +x setup.sh
```
4. Run the setup script, which will download the requierements and run the application, accessed by the *app.py* file.
```
./setup.sh 
```
5. The graphical interface can be accessed in a web-browse. By default the application is hosted at
```
127.0.0.1:5000
```

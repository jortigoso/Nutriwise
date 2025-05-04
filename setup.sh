#!/bin/bash

# Exit on any error
set -e

# ───────────────────────────────────────────────
# Nutriwise Setup Script
# Retrieval-Augmented Generation Fact-Checker
# ───────────────────────────────────────────────

# ───── System Requirements ─────
VENV_DIR="venv"
PYTHON_VERSION="python3"
OLLAMA_VERSION="0.4.8"
FASTTEXT_MODEL_URL="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_PATH="data/models/lid.176.bin"
CORPUS_PATH="data/sources/books_pubmed_nutrition_corpus_MAX.txt"
FAISS_INDEX_PATH="data/indices/faiss"
BM25_INDEX_PATH="data/indices/bm25"

# ───── Check Python ─────
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo "❌ Python 3.8+ is required. Please install it."
    exit 1
fi

# ───── Virtual Environment ─────
if [ ! -d "$VENV_DIR" ]; then
    echo "✅ Creating virtual environment in $VENV_DIR..."
    $PYTHON_VERSION -m venv $VENV_DIR
else
    echo "ℹ️ Virtual environment already exists."
fi

source $VENV_DIR/bin/activate

# ───── Upgrade Pip ─────
echo "📦 Upgrading pip..."
pip install --upgrade pip

# ───── Install Dependencies ─────
echo "📥 Installing Python dependencies..."
cat > requirements.txt << EOL
numpy==1.26.4
fasttext-wheel==0.9.2
requests==2.32.3
uvicorn==0.34.0
fastapi==0.115.11
torch==2.5.1
scikit-learn==1.6.0
bert-score==0.3.13
rank-bm25==0.2.2
transformers==4.50.0
langchain==0.3.23
langchain-community==0.3.21
langchain-core==0.3.54
langchain-huggingface==0.1.2
faiss-cpu==1.10.0
ollama==0.4.8
rouge-score==0.1.2
sentence-transformers==2.2.2
EOL
pip install -r requirements.txt

# ───── Install Ollama ─────
if ! command -v ollama &> /dev/null; then
    echo "🔧 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama is already installed."
fi

# ───── Start Ollama ─────
echo "🚀 Starting Ollama server..."
ollama serve &

sleep 5  # allow server to initialize

# ───── Pull Ollama Model ─────
echo "📦 Pulling LLM model qwen2:0.5b..."
ollama pull qwen2:0.5b

# ───── Prepare Directories ─────
echo "📁 Creating required directories..."
mkdir -p data/models data/indices/faiss data/indices/bm25 data/sources

# ───── Download Language Detection Model ─────
if [ ! -f "$FASTTEXT_MODEL_PATH" ]; then
    echo "📥 Downloading FastText language ID model..."
    curl -o "$FASTTEXT_MODEL_PATH" "$FASTTEXT_MODEL_URL"
else
    echo "✅ FastText model already exists."
fi

# ───── Verify Corpus ─────
if [ ! -f "$CORPUS_PATH" ]; then
    echo "⚠️  Corpus file missing: $CORPUS_PATH. Please check config.py or add the file."
else
    echo "✅ Corpus file found."
fi

# ───── Start API ─────
echo "🚀 Launching FastAPI application..."
python app.py &

sleep 5  # wait for app to start

# ───── Complete ─────
echo ""
echo "🎉 Setup completed successfully!"
echo "Nutriwise is running at http://127.0.0.1:5000"
echo ""
echo "💡 Features:"
echo "- Factual verification via RAG over nutrition corpus (PubMed, books)"
echo "- Cited sources & multilingual answers (EN, ES, FR, DE, IT)"
echo "- Local LLM (qwen2:0.5b), FAISS + BM25 hybrid retrieval"
echo "- Translation via Helsinki-NLP, FastText language detection"
echo "- Confidence scoring, bias mitigation, summarization"
echo "- Web interface with real-time Q&A + source display"
echo ""
echo "🧠 To activate the virtual environment later:"
echo "source $VENV_DIR/bin/activate"

'''
This is the configuration file for the application.
'''

import os
# import logging

CORPUS_PATH = "data/sources/books_pubmed_nutrition_corpus_MAX.txt"
FAISS_INDEX_PATH = "data/indices/faiss/faiss_index"
BM25_INDEX_PATH = "data/indices/bm25/bm25_index.pkl"
MAX_DOCS = 40000

FT_MODEL_PATH = 'data/models/lid.176.bin'

# Disable Tensorflow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_LOGGING"] = "0"
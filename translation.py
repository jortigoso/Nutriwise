"""
This module contains the translation logic, as well as the auxiliary 
functions to support it.
"""

from transformers import MarianMTModel, MarianTokenizer
from functools import lru_cache
import logging
import os
import requests
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    ('es', 'en'): "Helsinki-NLP/opus-mt-es-en",
    ('en', 'es'): "Helsinki-NLP/opus-mt-en-es",
    ('it', 'en'): "Helsinki-NLP/opus-mt-it-en",
    ('en', 'it'): "Helsinki-NLP/opus-mt-en-it",
    ('ru', 'en'): "Helsinki-NLP/opus-mt-ru-en",
    ('en', 'ru'): "Helsinki-NLP/opus-mt-en-ru",
    ('fr', 'en'): "Helsinki-NLP/opus-mt-fr-en",
    ('en', 'fr'): "Helsinki-NLP/opus-mt-en-fr",
    ('de', 'en'): "Helsinki-NLP/opus-mt-de-en",
    ('en', 'de'): "Helsinki-NLP/opus-mt-en-de",
    ('pt', 'en'): "Helsinki-NLP/opus-mt-tc-big-es_en",
    ('en', 'pt'): "Helsinki-NLP/opus-mt-tc-big-en_es",
    ('nl', 'en'): "Helsinki-NLP/opus-mt-nl-en",
    ('en', 'nl'): "Helsinki-NLP/opus-mt-en-nl",
    ('pl', 'en'): "Helsinki-NLP/opus-mt-pl-en",
    ('en', 'pl'): "Helsinki-NLP/opus-mt-en-pl",
    ('sv', 'en'): "Helsinki-NLP/opus-mt-sv-en",
    ('en', 'sv'): "Helsinki-NLP/opus-mt-en-sv",
    ('fi', 'en'): "Helsinki-NLP/opus-mt-fi-en",
    ('en', 'fi'): "Helsinki-NLP/opus-mt-en-fi",
    ('cs', 'en'): "Helsinki-NLP/opus-mt-cs-en",
    ('en', 'cs'): "Helsinki-NLP/opus-mt-en-cs",
    ('hu', 'en'): "Helsinki-NLP/opus-mt-hu-en",
    ('en', 'hu'): "Helsinki-NLP/opus-mt-en-hu",
    ('tr', 'en'): "Helsinki-NLP/opus-mt-tr-en",
    ('en', 'tr'): "Helsinki-NLP/opus-mt-en-tr",
    ('ar', 'en'): "Helsinki-NLP/opus-mt-ar-en",
    ('en', 'ar'): "Helsinki-NLP/opus-mt-en-ar",
    ('zh', 'en'): "Helsinki-NLP/opus-mt-zh-en",
    ('en', 'zh'): "Helsinki-NLP/opus-mt-en-zh",
    ('hi', 'en'): "Helsinki-NLP/opus-mt-hi-en",
    ('en', 'hi'): "Helsinki-NLP/opus-mt-en-hi",
    ('ko', 'en'): "Helsinki-NLP/opus-mt-ko-en",
    ('en', 'ko'): "Helsinki-NLP/opus-mt-en-ko",
}

SUPPORTED_LANGUAGES = set(lang for pair in SUPPORTED_MODELS.keys() for lang in pair)

_TRANSLATION_MODELS = {}

def load_translation_model(source_lang, target_lang):
    """
    Preloads the translation model for the given source and target languages.
    """
    logger.debug(f"Loading translation model for {source_lang} -> {target_lang}")

    key = (source_lang, target_lang)
    if key not in SUPPORTED_MODELS:
        raise ValueError(f"Translation from {source_lang} to {target_lang} is not supported.")
    
    model_id = SUPPORTED_MODELS[key]
    if key not in _TRANSLATION_MODELS:
        tokenizer = MarianTokenizer.from_pretrained(model_id)
        model = MarianMTModel.from_pretrained(model_id)
        _TRANSLATION_MODELS[key] = (model, tokenizer)
        logger.info(f"Loaded translation model {model_id} successfully")

    return _TRANSLATION_MODELS[key]

# NOTE: Wrapper of the avobe function, used to preload all supported models
def preload_models(lang_pairs):
    for source_lang, target_lang in lang_pairs:
        try:
            load_translation_model(source_lang, target_lang)
        except Exception as e:
            logger.error(f"Failed to preload model for {source_lang}->{target_lang}: {e}")

@lru_cache(maxsize=1000)
def translate_text(text, source_lang, target_lang):
    """
    Translates the given text from the source language to the target language.
    NOTE: The function uses LRU caching to prevent unnecessary model loading.
    """
    logger.debug(f"Translating: '{text}' from {source_lang} to {target_lang}")

    if not text.strip():
        logger.debug("Empty text provided for translation")  # Changed from logger.text
        return text
    
    model, tokenizer = load_translation_model(source_lang, target_lang)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_tokens = model.generate(**inputs)
    translated = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    logger.debug(f"Translation result: '{translated}'")
    return translated

def download_fasttext_model(model_path, url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"):
    """
    Download the FastText language identification model if it doesn't exist.
    """
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if not os.path.exists(model_path):
            logging.info(f"Downloading FastText model from {url} to {model_path}")
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise Exception(f"Failed to download model: HTTP {response.status_code}")
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logging.info(f"Successfully downloaded FastText model to {model_path}")
        else:
            logging.info(f"FastText model already exists at {model_path}")
    except Exception as e:
        logging.error(f"Failed to download FastText model: {e}")
        raise

def detect_language(fasttext_model, text):
    """
    Detects the language of the given text. Used to determine the
    source and target languages for the translation model.
    """

    if not text.strip():
        logging.warning("Empty text provided for language detection.")
        return "en"
    # Quick heuristic for Spanish punctuation/characters
    if any(ch in text for ch in "¿¡ñáéíóú"):
        logging.debug("Spanish characters detected, returning 'es'.")
        return "es"
    try:
        if fasttext_model is None:
            raise ValueError("fastText model not loaded.")
        predictions = fasttext_model.predict(text, k=1)
        print(predictions)
        lang = predictions[0][0].replace("__label__", "")
        confidence = predictions[1][0]
        if confidence >= 0.5:
            logging.debug(f"Detected language: {lang} (confidence: {confidence:.2f})")
            return lang
        else:
            logging.warning(f"Low confidence ({confidence:.2f}) for language {lang}, defaulting to 'en'.")
            return "en"
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "en"
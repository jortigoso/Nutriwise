import os
import time
import json
import csv
from pathlib import Path
import numpy as np

import app
import rag_qa
import translation
import logging
import fasttext

from rouge_score import rouge_scorer

BENCHMARK_DIR = "data/becnhmark/test.jsonl"

# Carga benchmark JSONL
def load_benchmark(path: Path):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Evaluación
def evaluate(model, temp, top_k=5):
    try:
        translation.download_fasttext_model(cfg.FT_MODEL_PATH)
        fasttext_model = fasttext.load_model(cfg.FT_MODEL_PATH)
        logging.info("Loaded fasttext language identification model.")
    except Exception as e:
        logging.error(f"Failed to load fasttext model: {e}")
        ft_model = None
        raise
    
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

    bench = load_benchmark(BENCHMARK_DIR)

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    fieldnames = [
        'question', 'answer_pred', 'confidence',
        'recall_at_k', 'bert_f1',
        'rougeL_precision', 'rougeL_recall', 'rougeL_fmeasure', 
        'latency'
    ]

    # Sanitize model name for directory creation
    model_name = model.replace('/', '_').replace(':', '_')
    OUTPUT_DIR = f"eval_output/{model_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = f"{OUTPUT_DIR}/{model_name}_temp{temp}.csv"

    with open(out_csv, 'w', newline='', encoding='utf8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for entry in bench:
            q = entry['question'].strip()
            gold_answer = entry.get('answer', '').strip()

            # Detectar idioma y traducir a inglés
            src_lang = translation.detect_language(ft_model, q)
            q_en = q if src_lang == 'en' else app.translate(q, src_lang, 'en')
            gold_en = gold_answer if src_lang == 'en' else app.translate(gold_answer, src_lang, 'en')

            # Invocación al chain
            t0 = time.time()
            result = app.robust_invoke(qa_chain, q_en)  # Asumiendo que la temperatura se configura externamente
            answer_text = result.get('answer', '').strip()
            latency = time.time() - t0

            # Ranking de fuentes
            docs = result.get('context', [])
            q_emb = embeddings.embed_query(q_en)
            top = rag_qa.rank_and_filter_sources(embeddings, q_emb, docs, threshold=0.3, top_k=top_k)
            top_ids = [d.metadata.get('source', f'doc_{i}') for i, (d, _, _) in enumerate(top)]
            top_embs = [emb for _, _, emb in top]

            # Generación fact-checked
            final_ans = app.generate_fact_checked_answer(qa_chain, q_en, answer_text)

            # Cálculo de métricas
            bert = rag_qa.compute_bert_score(
                [answer_text] * len(top),
                [d.page_content for d, _, _ in top],
                lang='en'
            )
            bert_f1 = bert.get('mean_f1', 0.0)
            recall = rag_qa.compute_recall_at_k(top_ids, top_ids, top_k)
            ans_emb = embeddings.embed_query(answer_text)
            conf = rag_qa.compute_confidence(ans_emb, top_embs, embeddings, bert_f1, top_k)

            # Cálculo de Rouge-L
            rouge_scores = rouge.score(gold_en, answer_text)
            r_prec = rouge_scores['rougeL'].precision
            r_rec  = rouge_scores['rougeL'].recall
            r_f    = rouge_scores['rougeL'].fmeasure

            # Traduce respuesta final de vuelta si necesario
            ans_final = final_ans if src_lang == 'en' else app.translate(final_ans, 'en', src_lang)

            # Escribir fila
            writer.writerow({
                'question': q,
                'answer_pred': ans_final,
                'confidence': conf,
                'recall_at_k': recall,
                'bert_f1': bert_f1,
                'rougeL_precision': r_prec,
                'rougeL_recall': r_rec,
                'rougeL_fmeasure': r_f,
                'latency': latency,
            })

    print(f"Results saved to {out_csv}")

if __name__ == "__main__":
    models = ["qwen2:0.5b", "gemma3:1b", "granite3-dense:8b", "llama3-chatqa"]
    temperatures = [0.1, 0.3, 0.5, 0.7]

    # Evaluación de todas las combinaciones
    for model in models:
        for temp in temperatures:
            evaluate(model, temp)

    results = []
    for model in models:
        model_name = model.replace('/', '_').replace(':', '_')  # Sanitize model name for directory
        for temp in temperatures:
            csv_path = f"eval_output/{model_name}/{model_name}_temp{temp}.csv"
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf8') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                    metrics = ['confidence', 'recall_at_k', 'bert_f1', 'rougeL_precision', 
                               'rougeL_recall', 'rougeL_fmeasure', 'latency']
                    averages = {metric: np.mean([float(row[metric]) for row in data if row[metric]]) 
                                for metric in metrics}
                    results.append({
                        'model': model,  # Use original model name for results
                        'temp': temp,
                        **averages
                    })
    
    # Rank based on combined confidence
    sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)

    # Imprimir ranking
    print("\nRanking based in combined confidence metric:")
    for i, res in enumerate(sorted_results, 1):
        print(f"{i}. Model: {res['model']}, Temp: {res['temp']}, "
              f"RougeL F-measure: {res['rougeL_fmeasure']:.4f}, "
              f"BERT F1: {res['bert_f1']:.4f}, Latency: {res['latency']:.4f}")

    # Guardar resumen en CSV
    summary_csv = "eval_output/summary.csv"
    with open(summary_csv, 'w', newline='', encoding='utf8') as f:
        fieldnames = ['rank', 'model', 'temp', 'confidence', 'recall_at_k', 'bert_f1',
                      'rougeL_precision', 'rougeL_recall', 'rougeL_fmeasure', 'latency']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, res in enumerate(sorted_results, 1):
            writer.writerow({
                'rank': i,
                'model': res['model'],
                'temp': res['temp'],
                'confidence': res['confidence'],
                'recall_at_k': res['recall_at_k'],
                'bert_f1': res['bert_f1'],
                'rougeL_precision': res['rougeL_precision'],
                'rougeL_recall': res['rougeL_recall'],
                'rougeL_fmeasure': res['rougeL_fmeasure'],
                'latency': res['latency'],
            })
    print(f"\nSummary of averages saved to {summary_csv}")
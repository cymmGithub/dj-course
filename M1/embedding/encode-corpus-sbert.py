#!/usr/bin/env python3
"""
KODOWANIE KORPUSU - SENTENCE-BERT
===================================
Generuje embeddingi dla całego korpusu używając modeli Sentence-BERT.

Obsługuje różne modele, w tym:
- intfloat/multilingual-e5-small (uniwersalny multilingual)
- sdadas/stella-pl (najlepszy dla polskiego - NDCG@10: 60.52)
- radlab/polish-sts-v2 (polski model podobieństwa zdań)

Zapisuje:
- Macierz embeddingów: sbert_<model_name>_embeddings.npy
- Mapę zdań: sbert_sentence_map.json
"""

import numpy as np
import json
import logging
import os
import time
from sentence_transformers import SentenceTransformer
from corpora import CORPORA_FILES

# Ustawienie logowania
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- KONFIGURACJA ---
# Wybierz model (odkomentuj jeden):
# MODEL_NAME = 'intfloat/multilingual-e5-small'  # Uniwersalny multilingual
MODEL_NAME = 'sdadas/stella-pl'  # ⭐ Najlepszy dla polskiego
# MODEL_NAME = 'radlab/polish-sts-v2'  # Polski model STS

files = CORPORA_FILES["ALL"]

# Automatyczne generowanie nazwy pliku wyjściowego na podstawie modelu
model_slug = MODEL_NAME.replace('/', '_').replace('-', '_')
OUTPUT_EMBEDDINGS_FILE = f"sbert_{model_slug}_embeddings.npy"
OUTPUT_SENTENCE_MAP = "sbert_sentence_map.json"

print("\n" + "="*80)
print("  KODOWANIE KORPUSU - SENTENCE-BERT")
print("="*80)
print(f"Model: {MODEL_NAME}")
print("="*80)

# --- ETAP 1: Wczytanie Korpusu ---
def load_raw_sentences(file_list):
    """Wczytuje surowe zdania z listy plików."""
    raw_sentences = []
    print(f"\n[1/3] Wczytywanie tekstu z {len(file_list)} plików...")
    for file in file_list:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                raw_sentences.extend(lines)
        except FileNotFoundError:
            print(f"OSTRZEŻENIE: Nie znaleziono pliku '{file}'. Pomijam.")
        except Exception as e:
            print(f"BŁĄD podczas przetwarzania pliku '{file}': {e}")

    if not raw_sentences:
        raise ValueError("Korpus danych jest pusty lub nie został wczytany.")

    return raw_sentences

try:
    raw_sentences = load_raw_sentences(files)
    print(f"✓ Wczytano {len(raw_sentences):,} zdań do przetworzenia")
except ValueError as e:
    print(f"❌ BŁĄD: {e}")
    exit(1)

# --- ETAP 2: Ładowanie Modelu ---
print(f"\n[2/3] Ładowanie modelu Sentence-BERT: {MODEL_NAME}...")
try:
    model_sbert = SentenceTransformer(MODEL_NAME)
    print("✓ Model załadowany pomyślnie")
    print(f"  └─ Wymiar embeddingu: {model_sbert.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"❌ FATALNY BŁĄD podczas ładowania modelu {MODEL_NAME}: {e}")
    exit(1)

# --- ETAP 3: Generowanie Embeddingów ---
print(f"\n[3/3] Generowanie embeddingów dla {len(raw_sentences):,} zdań...")
print("(To może potrwać kilka minut...)")

start_time = time.time()

# Metoda .encode() automatycznie tokenizuje i generuje wektory
sentence_embeddings = model_sbert.encode(
    raw_sentences,
    show_progress_bar=True,
    convert_to_numpy=True,
    batch_size=32  # Możesz zwiększyć jeśli masz więcej RAM/GPU
)

end_time = time.time()
encoding_time = end_time - start_time

print(f"\n✓ Generowanie zakończone w {encoding_time:.2f}s")
print(f"  └─ Kształt macierzy embeddingów: {sentence_embeddings.shape}")

# --- ETAP 4: Zapisywanie ---
print(f"\n💾 Zapisywanie embeddingów...")

# Zapisz embeddingi
np.save(OUTPUT_EMBEDDINGS_FILE, sentence_embeddings)
emb_size_mb = os.path.getsize(OUTPUT_EMBEDDINGS_FILE) / (1024 * 1024)
print(f"✓ Embeddingi zapisane: {OUTPUT_EMBEDDINGS_FILE} ({emb_size_mb:.1f} MB)")

# Zapisz mapę zdań (jeśli nie istnieje)
if not os.path.exists(OUTPUT_SENTENCE_MAP):
    with open(OUTPUT_SENTENCE_MAP, "w", encoding="utf-8") as f:
        json.dump(raw_sentences, f, ensure_ascii=False, indent=2)
    print(f"✓ Mapa zdań zapisana: {OUTPUT_SENTENCE_MAP}")
else:
    print(f"ℹ️  Mapa zdań już istnieje: {OUTPUT_SENTENCE_MAP}")

print("\n" + "="*80)
print("  KODOWANIE KORPUSU ZAKOŃCZONE")
print("="*80)
print(f"\n📊 Podsumowanie:")
print(f"  ├─ Model: {MODEL_NAME}")
print(f"  ├─ Liczba zdań: {len(raw_sentences):,}")
print(f"  ├─ Wymiar wektora: {sentence_embeddings.shape[1]}")
print(f"  ├─ Czas kodowania: {encoding_time:.2f}s")
print(f"  └─ Plik wyjściowy: {OUTPUT_EMBEDDINGS_FILE}")
print(f"\n💡 Aby odpytać korpus, uruchom:")
print(f"  python query-sbert.py")
print(f"\n💡 WAŻNE: W query-sbert.py użyj tego samego modelu: {MODEL_NAME}")

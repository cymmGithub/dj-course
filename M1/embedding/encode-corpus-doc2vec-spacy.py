#!/usr/bin/env python3
"""
KODOWANIE KORPUSU - DOC2VEC (SPACY)
====================================
Wczytuje wytrenowany model Doc2Vec (doc2vec_model_spacy.model) i generuje
embeddingi dla wszystkich zdań z korpusu.

Zapisuje:
- Macierz embeddingów: doc2vec_spacy_corpus_embeddings.npy
- Mapę zdań: doc2vec_model_sentence_map_spacy.json (już istnieje z treningu)
"""

import numpy as np
import json
import logging
from gensim.models.doc2vec import Doc2Vec
import os
import time
import spacy

# Ustawienie logowania
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- KONFIGURACJA ---
MODEL_FILE = "doc2vec_model_spacy.model"
SENTENCE_MAP_FILE = "doc2vec_model_sentence_map_spacy.json"
OUTPUT_EMBEDDINGS_FILE = "doc2vec_spacy_corpus_embeddings.npy"

print("\n" + "="*80)
print("  KODOWANIE KORPUSU - DOC2VEC (SPACY)")
print("="*80)

# --- ETAP 1: Wczytanie Modelu Doc2Vec ---
print(f"\n[1/4] Wczytywanie modelu Doc2Vec: {MODEL_FILE}...")
try:
    model_d2v = Doc2Vec.load(MODEL_FILE)
    print(f"✓ Model załadowany pomyślnie")
    print(f"  ├─ Rozmiar słownika: {len(model_d2v.wv):,} tokenów")
    print(f"  ├─ Wymiar wektora: {model_d2v.vector_size}")
    print(f"  └─ Liczba dokumentów: {len(model_d2v.dv):,}")
except FileNotFoundError:
    print(f"❌ BŁĄD: Nie znaleziono pliku '{MODEL_FILE}'")
    print("\nAby wytrenować model, uruchom:")
    print("  python train-doc2vec-spacy.py")
    exit(1)

# --- ETAP 2: Wczytanie Mapy Zdań ---
print(f"\n[2/4] Wczytywanie mapy zdań: {SENTENCE_MAP_FILE}...")
try:
    with open(SENTENCE_MAP_FILE, 'r', encoding='utf-8') as f:
        raw_sentences = json.load(f)
    print(f"✓ Załadowano {len(raw_sentences):,} zdań z korpusu")
except FileNotFoundError:
    print(f"❌ BŁĄD: Nie znaleziono pliku '{SENTENCE_MAP_FILE}'")
    exit(1)

# --- ETAP 3: Wczytanie spaCy ---
print(f"\n[3/4] Wczytywanie modelu spaCy...")
try:
    nlp = spacy.load("pl_core_news_sm")
    print("✓ Model spaCy załadowany: pl_core_news_sm")
except OSError:
    print("❌ BŁĄD: Nie znaleziono modelu spaCy")
    print("Zainstaluj: python -m spacy download pl_core_news_sm")
    exit(1)

# --- ETAP 4: Generowanie Embeddingów dla Korpusu ---
print(f"\n[4/4] Generowanie embeddingów dla {len(raw_sentences):,} zdań...")
print("(To może potrwać kilka minut...)")

start_time = time.time()

# Zbieramy embeddingi bezpośrednio z wytrenowanego modelu
# Model Doc2Vec już ma embeddingi dla wszystkich dokumentów treningowych
corpus_embeddings = []

for i in range(len(raw_sentences)):
    # Pobierz embedding dla dokumentu o ID=str(i)
    # (w train-doc2vec-spacy.py używamy tags=[str(i)])
    doc_vector = model_d2v.dv[str(i)]
    corpus_embeddings.append(doc_vector)

    # Postęp
    if (i + 1) % 10000 == 0:
        print(f"  Przetworzone: {i + 1:,} / {len(raw_sentences):,} zdań")

corpus_embeddings = np.array(corpus_embeddings)
end_time = time.time()

print(f"\n✓ Generowanie zakończone w {end_time - start_time:.2f}s")
print(f"  Kształt macierzy embeddingów: {corpus_embeddings.shape}")

# --- ETAP 5: Zapisywanie Embeddingów ---
print(f"\n💾 Zapisywanie embeddingów do: {OUTPUT_EMBEDDINGS_FILE}...")
np.save(OUTPUT_EMBEDDINGS_FILE, corpus_embeddings)

file_size_mb = os.path.getsize(OUTPUT_EMBEDDINGS_FILE) / (1024 * 1024)
print(f"✓ Embeddingi zapisane ({file_size_mb:.1f} MB)")

print("\n" + "="*80)
print("  KODOWANIE KORPUSU ZAKOŃCZONE")
print("="*80)
print(f"\n📊 Podsumowanie:")
print(f"  ├─ Liczba zdań: {len(raw_sentences):,}")
print(f"  ├─ Wymiar wektora: {corpus_embeddings.shape[1]}")
print(f"  └─ Plik wyjściowy: {OUTPUT_EMBEDDINGS_FILE}")
print(f"\n💡 Aby odpytać korpus, uruchom:")
print(f"  python query-doc2vec-spacy.py")

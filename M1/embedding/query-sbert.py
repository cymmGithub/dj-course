#!/usr/bin/env python3
"""
ODPYTYWANIE KORPUSU - SENTENCE-BERT
====================================
Wczytuje zapisane embeddingi korpusu i model SBERT,
a następnie pozwala na wyszukiwanie podobnych zdań.

Testuje zarówno:
- Zdania wymyślone (spoza korpusu)
- Zdania z korpusu treningowego

WAŻNE: Użyj tego samego modelu, którym kodowałeś korpus w encode-corpus-sbert.py!
"""

import numpy as np
import json
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Ustawienie logowania
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- KONFIGURACJA ---
# WAŻNE: Użyj tego samego modelu co w encode-corpus-sbert.py!
# MODEL_NAME = 'intfloat/multilingual-e5-small'  # Uniwersalny multilingual
MODEL_NAME = 'sdadas/stella-pl'  # ⭐ Najlepszy dla polskiego
# MODEL_NAME = 'radlab/polish-sts-v2'  # Polski model STS

# Automatyczne określenie nazwy pliku embeddingów
model_slug = MODEL_NAME.replace('/', '_').replace('-', '_')
EMBEDDINGS_FILE = f"sbert_{model_slug}_embeddings.npy"
SENTENCE_MAP_FILE = "sbert_sentence_map.json"

print("\n" + "="*80)
print("  ODPYTYWANIE KORPUSU - SENTENCE-BERT")
print("="*80)
print(f"Model: {MODEL_NAME}")
print("="*80)

# --- ETAP 1: Wczytanie Mapy Zdań ---
print(f"\n[1/3] Wczytywanie mapy zdań: {SENTENCE_MAP_FILE}...")
try:
    with open(SENTENCE_MAP_FILE, 'r', encoding='utf-8') as f:
        raw_sentences = json.load(f)
    print(f"✓ Załadowano {len(raw_sentences):,} zdań z korpusu")
except FileNotFoundError:
    print(f"❌ BŁĄD: Nie znaleziono pliku '{SENTENCE_MAP_FILE}'")
    print("\nUruchom najpierw:")
    print("  python encode-corpus-sbert.py")
    exit(1)

# --- ETAP 2: Wczytanie Embeddingów Korpusu ---
print(f"\n[2/3] Wczytywanie embeddingów korpusu: {EMBEDDINGS_FILE}...")
try:
    corpus_embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✓ Embeddingi załadowane")
    print(f"  └─ Kształt macierzy: {corpus_embeddings.shape}")
except FileNotFoundError:
    print(f"❌ BŁĄD: Nie znaleziono pliku '{EMBEDDINGS_FILE}'")
    print("\nUruchom najpierw:")
    print("  python encode-corpus-sbert.py")
    print(f"\nUbedź się, że użyłeś modelu: {MODEL_NAME}")
    exit(1)

# --- ETAP 3: Wczytanie Modelu SBERT ---
print(f"\n[3/3] Ładowanie modelu Sentence-BERT: {MODEL_NAME}...")
try:
    model_sbert = SentenceTransformer(MODEL_NAME)
    print("✓ Model SBERT załadowany pomyślnie")
    print(f"  └─ Wymiar embeddingu: {model_sbert.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"❌ BŁĄD podczas ładowania modelu: {e}")
    exit(1)

# --- FUNKCJA POMOCNICZA: Wyszukiwanie Podobnych Zdań ---
def find_similar_sentences(query_sentence, top_n=5):
    """
    Wyszukuje zdania najbardziej podobne do zapytania.

    Args:
        query_sentence (str): Zdanie zapytania
        top_n (int): Liczba wyników do zwrócenia

    Returns:
        list: Lista tupli (similarity_score, sentence_text, sentence_index)
    """
    # 1. Generowanie embeddingu dla zapytania
    query_embedding = model_sbert.encode(
        [query_sentence],
        convert_to_numpy=True
    )

    # 2. Obliczenie podobieństwa kosinusowego
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    # 3. Znajdowanie top N najbardziej podobnych
    top_indices = np.argsort(similarities)[::-1][:top_n]

    # 4. Budowanie wyniku
    results = []
    for idx in top_indices:
        similarity = similarities[idx]
        sentence = raw_sentences[idx]
        results.append((similarity, sentence, idx))

    return results

# =========================================================================
# === TEST 1 - ZDANIA WYMYŚLONE (spoza korpusu) ===
# =========================================================================

print("\n" + "="*80)
print("  TEST 1: ZDANIA WYMYŚLONE (spoza korpusu)")
print("="*80)

test_queries_custom = [
    "Jestem głodny i bardzo chętnie zjadłbym coś.",
    "Wojsko wejdzie do miast i skończą się bunty",
    "Leczenie tego schorzenia jest bardzo ważne i wymaga interwencji lekarza.",
    "Piękny zachód słońca nad morzem.",
    "Matematyka jest fascynującą dziedziną nauki."
]

for query in test_queries_custom:
    print(f"\n🔍 Zapytanie: \"{query}\"")
    print("-" * 80)

    results = find_similar_sentences(query, top_n=5)

    if results:
        print(f"Top 5 najbardziej podobnych zdań:")
        for sim, sentence, idx in results:
            # Skróć zdanie jeśli jest bardzo długie
            display_sentence = sentence if len(sentence) <= 100 else sentence[:97] + "..."
            print(f"  [{idx:6d}] Sim: {sim:.4f} | {display_sentence}")
    else:
        print("  (Brak wyników)")

# =========================================================================
# === TEST 2 - ZDANIA Z KORPUSU TRENINGOWEGO ===
# =========================================================================

print("\n" + "="*80)
print("  TEST 2: ZDANIA Z KORPUSU")
print("="*80)
print("(Sprawdzamy, czy model dobrze znajduje podobne zdania dla zdań z korpusu)")

# Wybieramy losowe zdania z korpusu
np.random.seed(42)
test_indices = np.random.choice(len(raw_sentences), size=3, replace=False)

for idx in test_indices:
    query = raw_sentences[idx]

    print(f"\n🔍 Zapytanie [z korpusu, ID={idx}]: \"{query[:80]}...\"")
    print("-" * 80)

    results = find_similar_sentences(query, top_n=6)

    if results:
        print(f"Top 6 najbardziej podobnych zdań:")
        for sim, sentence, result_idx in results:
            # Zaznacz zdanie jeśli to to samo co zapytanie
            marker = "⭐ [TO SAMO]" if result_idx == idx else ""
            display_sentence = sentence if len(sentence) <= 80 else sentence[:77] + "..."
            print(f"  [{result_idx:6d}] Sim: {sim:.4f} | {display_sentence} {marker}")

# =========================================================================
# === INTERAKTYWNY TRYB (opcjonalny) ===
# =========================================================================

print("\n" + "="*80)
print("  TRYB INTERAKTYWNY")
print("="*80)
print("Możesz teraz wpisać własne zdania do wyszukiwania.")
print("Wpisz 'quit' lub 'exit' aby zakończyć.\n")

while True:
    try:
        user_query = input("Wpisz zdanie: ").strip()

        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\nDo widzenia! 👋")
            break

        if not user_query:
            continue

        print(f"\n🔍 Szukam podobnych zdań dla: \"{user_query}\"")
        print("-" * 80)

        results = find_similar_sentences(user_query, top_n=5)

        if results:
            print(f"Top 5 wyników:")
            for sim, sentence, idx in results:
                display_sentence = sentence if len(sentence) <= 100 else sentence[:97] + "..."
                print(f"  [{idx:6d}] Sim: {sim:.4f} | {display_sentence}")
        else:
            print("  (Brak wyników)")

        print()

    except (KeyboardInterrupt, EOFError):
        print("\n\nDo widzenia! 👋")
        break

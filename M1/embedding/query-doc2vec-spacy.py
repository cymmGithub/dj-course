#!/usr/bin/env python3
"""
ODPYTYWANIE KORPUSU - DOC2VEC (SPACY)
======================================
Wczytuje zapisane embeddingi korpusu i wytrenowany model Doc2Vec,
a następnie pozwala na wyszukiwanie podobnych zdań.

Testuje zarówno:
- Zdania wymyślone (spoza korpusu)
- Zdania z korpusu treningowego
"""

import numpy as np
import json
import logging
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import os
import spacy

# Ustawienie logowania
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- KONFIGURACJA ---
MODEL_FILE = "doc2vec_model_spacy.model"
SENTENCE_MAP_FILE = "doc2vec_model_sentence_map_spacy.json"
EMBEDDINGS_FILE = "doc2vec_spacy_corpus_embeddings.npy"

print("\n" + "="*80)
print("  ODPYTYWANIE KORPUSU - DOC2VEC (SPACY)")
print("="*80)

# --- ETAP 1: Wczytanie Modelu Doc2Vec ---
print(f"\n[1/4] Wczytywanie modelu Doc2Vec: {MODEL_FILE}...")
try:
    model_d2v = Doc2Vec.load(MODEL_FILE)
    print(f"✓ Model załadowany pomyślnie")
    print(f"  ├─ Wymiar wektora: {model_d2v.vector_size}")
    print(f"  └─ Liczba epok (dla inference): {model_d2v.epochs}")
except FileNotFoundError:
    print(f"❌ BŁĄD: Nie znaleziono pliku '{MODEL_FILE}'")
    print("\nUruchom najpierw:")
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

# --- ETAP 3: Wczytanie Embeddingów Korpusu ---
print(f"\n[3/4] Wczytywanie embeddingów korpusu: {EMBEDDINGS_FILE}...")
try:
    corpus_embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✓ Embeddingi załadowane")
    print(f"  └─ Kształt macierzy: {corpus_embeddings.shape}")
except FileNotFoundError:
    print(f"❌ BŁĄD: Nie znaleziono pliku '{EMBEDDINGS_FILE}'")
    print("\nUruchom najpierw:")
    print("  python encode-corpus-doc2vec-spacy.py")
    exit(1)

# --- ETAP 4: Wczytanie spaCy ---
print(f"\n[4/4] Wczytywanie modelu spaCy...")
try:
    nlp = spacy.load("pl_core_news_sm")
    print("✓ Model spaCy załadowany: pl_core_news_sm")
except OSError:
    print("❌ BŁĄD: Nie znaleziono modelu spaCy")
    print("Zainstaluj: python -m spacy download pl_core_news_sm")
    exit(1)

# --- FUNKCJA POMOCNICZA: Tokenizacja spaCy ---
def tokenize_with_spacy(text):
    """Tokenizacja używając spaCy z lemmatyzacją (jak w treningu)."""
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_space and token.text.strip()
    ]
    return tokens

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
    # 1. Tokenizacja zapytania
    query_tokens = tokenize_with_spacy(query_sentence)

    if not query_tokens:
        print("⚠️  OSTRZEŻENIE: Zapytanie nie zawiera żadnych tokenów po przetworzeniu")
        return []

    # 2. Generowanie embeddingu dla zapytania
    # Używamy infer_vector() z modelu Doc2Vec
    query_embedding = model_d2v.infer_vector(query_tokens, epochs=model_d2v.epochs)

    # 3. Obliczenie podobieństwa kosinusowego
    # Musimy zmienić kształt na (1, wymiar) dla cosine_similarity
    query_embedding_2d = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding_2d, corpus_embeddings)[0]

    # 4. Znajdowanie top N najbardziej podobnych
    top_indices = np.argsort(similarities)[::-1][:top_n]

    # 5. Budowanie wyniku
    results = []
    for idx in top_indices:
        similarity = similarities[idx]
        sentence = raw_sentences[idx]
        results.append((similarity, sentence, idx))

    return results

# =========================================================================
# === TESTY - ZDANIA WYMYŚLONE (spoza korpusu) ===
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
print("  TEST 2: ZDANIA Z KORPUSU TRENINGOWEGO")
print("="*80)
print("(Sprawdzamy, czy model dobrze odtwarza podobieństwa dla zdań, które widział)")

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

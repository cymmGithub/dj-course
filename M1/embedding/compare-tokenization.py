import numpy as np
import json
import logging
from gensim.models.doc2vec import Doc2Vec
from tokenizers import Tokenizer

# Ustawienie logowania dla gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Pliki dla modelu BPE
TOKENIZER_FILE_BPE = "../tokenizer/tokenizers/nkjp-tokenizer.json"
MODEL_FILE_BPE = "doc2vec_model_bpe.model"
SENTENCE_MAP_FILE_BPE = "doc2vec_model_sentence_map_bpe.json"

# Pliki dla modelu SIMPLE
MODEL_FILE_SIMPLE = "doc2vec_model_simple.model"
SENTENCE_MAP_FILE_SIMPLE = "doc2vec_model_sentence_map_simple.json"

print("\n" + "="*80)
print("  PORÓWNANIE TOKENIZACJI: BPE vs SIMPLE SPLIT")
print("="*80)

# --- ETAP 1: Wczytanie Modeli i Danych ---
print("\n--- Wczytywanie modeli ---")

# Wczytanie tokenizera BPE
try:
    tokenizer_bpe = Tokenizer.from_file(TOKENIZER_FILE_BPE)
    print(f"✓ Tokenizer BPE wczytany z: {TOKENIZER_FILE_BPE}")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku tokenizera BPE '{TOKENIZER_FILE_BPE}'.")
    raise

# Wczytanie modelu BPE
try:
    model_bpe = Doc2Vec.load(MODEL_FILE_BPE)
    print(f"✓ Model BPE wczytany z: {MODEL_FILE_BPE}")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku modelu BPE '{MODEL_FILE_BPE}'.")
    print("Uruchom najpierw: python train-doc2vec-bpe.py")
    raise

# Wczytanie mapy zdań BPE
try:
    with open(SENTENCE_MAP_FILE_BPE, "r", encoding="utf-8") as f:
        sentence_lookup_bpe = json.load(f)
    print(f"✓ Mapa zdań BPE wczytana z: {SENTENCE_MAP_FILE_BPE}")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku mapy zdań BPE '{SENTENCE_MAP_FILE_BPE}'.")
    raise

# Wczytanie modelu SIMPLE
try:
    model_simple = Doc2Vec.load(MODEL_FILE_SIMPLE)
    print(f"✓ Model SIMPLE wczytany z: {MODEL_FILE_SIMPLE}")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku modelu SIMPLE '{MODEL_FILE_SIMPLE}'.")
    print("Uruchom najpierw: python train-doc2vec.py")
    raise

# Wczytanie mapy zdań SIMPLE
try:
    with open(SENTENCE_MAP_FILE_SIMPLE, "r", encoding="utf-8") as f:
        sentence_lookup_simple = json.load(f)
    print(f"✓ Mapa zdań SIMPLE wczytana z: {SENTENCE_MAP_FILE_SIMPLE}")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku mapy zdań SIMPLE '{SENTENCE_MAP_FILE_SIMPLE}'.")
    raise

# --- ETAP 2: Porównanie Tokenizacji na Przykładach ---
print("\n" + "="*80)
print("  PORÓWNANIE TOKENIZACJI NA PRZYKŁADACH")
print("="*80)

test_sentences = [
    "Jestem głodny.",
    "Kot siedzi na macie.",
    "Piękna pogoda dzisiaj.",
    "Król Polski przyjechał do Warszawy."
]

for i, sentence in enumerate(test_sentences, 1):
    print(f"\n--- Przykład {i} ---")
    print(f"Zdanie: \"{sentence}\"")
    print()

    # Tokenizacja BPE
    tokens_bpe = tokenizer_bpe.encode(sentence).tokens
    print(f"BPE tokenization ({len(tokens_bpe)} tokenów):")
    print(f"  {tokens_bpe}")

    # Tokenizacja SIMPLE
    tokens_simple = sentence.split()
    print(f"\nSIMPLE tokenization ({len(tokens_simple)} tokenów):")
    print(f"  {tokens_simple}")

    # Porównanie długości
    diff = len(tokens_bpe) - len(tokens_simple)
    print(f"\nRóżnica: BPE ma {abs(diff)} {'więcej' if diff > 0 else 'mniej'} tokenów niż SIMPLE")

# --- ETAP 3: Porównanie Wnioskowania (Inference) ---
print("\n" + "="*80)
print("  PORÓWNANIE WNIOSKOWANIA (INFERENCE)")
print("="*80)

test_sentence = "Jestem głodny."
topn = 5

print(f"\nZdanie testowe: \"{test_sentence}\"")
print("\n" + "-"*80)

# === BPE Model ===
print("\n🔷 MODEL BPE:")
print("-"*40)
tokens_bpe = tokenizer_bpe.encode(test_sentence).tokens
print(f"Tokeny: {tokens_bpe}")

inferred_vector_bpe = model_bpe.infer_vector(tokens_bpe, epochs=model_bpe.epochs)
similar_docs_bpe = model_bpe.dv.most_similar([inferred_vector_bpe], topn=topn)

print(f"\n{topn} najbardziej podobnych zdań (BPE):")
for rank, (doc_id_str, similarity) in enumerate(similar_docs_bpe, 1):
    doc_index = int(doc_id_str)
    try:
        original_sentence = sentence_lookup_bpe[doc_index]
        print(f"  {rank}. Sim: {similarity:.4f} | {original_sentence[:80]}")
    except IndexError:
        print(f"  {rank}. Sim: {similarity:.4f} | BŁĄD: Nie znaleziono zdania")

# === SIMPLE Model ===
print("\n\n🔶 MODEL SIMPLE:")
print("-"*40)
tokens_simple = test_sentence.split()
print(f"Tokeny: {tokens_simple}")

inferred_vector_simple = model_simple.infer_vector(tokens_simple, epochs=model_simple.epochs)
similar_docs_simple = model_simple.dv.most_similar([inferred_vector_simple], topn=topn)

print(f"\n{topn} najbardziej podobnych zdań (SIMPLE):")
for rank, (doc_id_str, similarity) in enumerate(similar_docs_simple, 1):
    doc_index = int(doc_id_str)
    try:
        original_sentence = sentence_lookup_simple[doc_index]
        print(f"  {rank}. Sim: {similarity:.4f} | {original_sentence[:80]}")
    except IndexError:
        print(f"  {rank}. Sim: {similarity:.4f} | BŁĄD: Nie znaleziono zdania")

# --- ETAP 4: Statystyki Porównawcze ---
print("\n" + "="*80)
print("  STATYSTYKI PORÓWNAWCZE")
print("="*80)

print(f"\n📊 BPE Model:")
print(f"  ├─ Liczba wektorów zdań: {len(model_bpe.dv)}")
print(f"  ├─ Wymiar wektora: {model_bpe.vector_size}")
print(f"  └─ Średnia podobieństwa (top 5): {np.mean([s for _, s in similar_docs_bpe]):.4f}")

print(f"\n📊 SIMPLE Model:")
print(f"  ├─ Liczba wektorów zdań: {len(model_simple.dv)}")
print(f"  ├─ Wymiar wektora: {model_simple.vector_size}")
print(f"  └─ Średnia podobieństwa (top 5): {np.mean([s for _, s in similar_docs_simple]):.4f}")

# Porównanie wektorów
print(f"\n📏 Porównanie wektorów dla zdania testowego:")
print(f"  ├─ Norma wektora BPE: {np.linalg.norm(inferred_vector_bpe):.4f}")
print(f"  ├─ Norma wektora SIMPLE: {np.linalg.norm(inferred_vector_simple):.4f}")
print(f"  └─ Podobieństwo cosinusowe między wektorami: {np.dot(inferred_vector_bpe, inferred_vector_simple) / (np.linalg.norm(inferred_vector_bpe) * np.linalg.norm(inferred_vector_simple)):.4f}")

print("\n" + "="*80)
print("  PORÓWNANIE ZAKOŃCZONE")
print("="*80)

# --- ETAP 5: Interaktywny tryb porównawczy ---
print("\n\nCzy chcesz przetestować własne zdania? (t/n): ", end="")
try:
    choice = input().strip().lower()

    if choice == 't' or choice == 'y':
        while True:
            print("\n" + "-"*80)
            user_sentence = input("Wprowadź zdanie (lub 'q' aby zakończyć): ").strip()

            if user_sentence.lower() == 'q':
                print("Zakończono tryb interaktywny.")
                break

            if not user_sentence:
                print("Zdanie nie może być puste.")
                continue

            print(f"\n🔍 Testowanie: \"{user_sentence}\"")

            # BPE
            print("\n🔷 BPE:")
            tokens_bpe = tokenizer_bpe.encode(user_sentence).tokens
            print(f"  Tokeny ({len(tokens_bpe)}): {tokens_bpe}")
            vector_bpe = model_bpe.infer_vector(tokens_bpe, epochs=model_bpe.epochs)
            similar_bpe = model_bpe.dv.most_similar([vector_bpe], topn=3)
            print(f"  Top 3 podobne:")
            for rank, (doc_id, sim) in enumerate(similar_bpe, 1):
                sent = sentence_lookup_bpe[int(doc_id)]
                print(f"    {rank}. [{sim:.4f}] {sent[:60]}")

            # SIMPLE
            print("\n🔶 SIMPLE:")
            tokens_simple = user_sentence.split()
            print(f"  Tokeny ({len(tokens_simple)}): {tokens_simple}")
            vector_simple = model_simple.infer_vector(tokens_simple, epochs=model_simple.epochs)
            similar_simple = model_simple.dv.most_similar([vector_simple], topn=3)
            print(f"  Top 3 podobne:")
            for rank, (doc_id, sim) in enumerate(similar_simple, 1):
                sent = sentence_lookup_simple[int(doc_id)]
                print(f"    {rank}. [{sim:.4f}] {sent[:60]}")

except EOFError:
    print("\n\nZakończono.")
except KeyboardInterrupt:
    print("\n\nPrzerwano przez użytkownika.")

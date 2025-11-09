import numpy as np
import json
import logging
from gensim.models.doc2vec import Doc2Vec
from tokenizers import Tokenizer

# Ustawienie logowania dla gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Pliki wejściowe
TOKENIZER_FILE = "../tokenizer/tokenizers/all-tokenizer.json"
MODEL_FILE = "doc2vec_model_combined.model"
SENTENCE_MAP_FILE = "doc2vec_model_sentence_map_combined.json"

print("=" * 50)
print("=== ROZPOCZYNAM ETAP WNIOSKOWANIA (INFERENCE) ===")
print("=" * 50)

# --- ETAP 1: Wczytanie Modelu i Danych ---
print("\nWczytywanie wytrenowanego modelu...")
try:
    loaded_model = Doc2Vec.load(MODEL_FILE)
    print(f"Model wczytany pomyślnie z: '{MODEL_FILE}'")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku modelu '{MODEL_FILE}'.")
    print("Uruchom najpierw skrypt 'train-doc2vec.py' aby wytrenować model.")
    raise

print("Wczytywanie mapy zdań...")
try:
    with open(SENTENCE_MAP_FILE, "r", encoding="utf-8") as f:
        sentence_lookup = json.load(f)
    print(f"Mapa zdań wczytana pomyślnie z: '{SENTENCE_MAP_FILE}'")
    print(f"Liczba zdań w korpusie: {len(sentence_lookup)}")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku mapy zdań '{SENTENCE_MAP_FILE}'.")
    raise

print("Wczytywanie tokenizera...")
try:
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    print(f"Tokenizer wczytany pomyślnie z: '{TOKENIZER_FILE}'")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku tokenizera '{TOKENIZER_FILE}'.")
    raise

# --- ETAP 2: Testowanie Modelu ---
# 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥 TESTOWANIE 🔥🔥🔥🔥🔥🔥🔥🔥
new_sentence = "Jestem głodny."
print(f"\n\nZdanie do wnioskowania: \"{new_sentence}\"")

# Tokenizacja nowego zdania
new_tokens = tokenizer.encode(new_sentence).tokens
print(f"Tokeny: {new_tokens}")

# Generowanie wektora dla nowego zdania
inferred_vector = loaded_model.infer_vector(new_tokens, epochs=loaded_model.epochs)
print(f"\nWygenerowany wektor (embedding) dla zdania. Kształt: {inferred_vector.shape}")

# Znajdowanie najbardziej podobnych wektorów z przestrzeni dokumentów/zdań
# topn - liczba najbardziej podobnych zdań do znalezienia
topn = 5
most_similar_docs = loaded_model.dv.most_similar([inferred_vector], topn=topn)

print(f"\n{topn} najbardziej podobnych zdań z korpusu (Doc2Vec Inference):")
for doc_id_str, similarity in most_similar_docs:
    # Konwertujemy ID (string) z powrotem na indeks (int)
    doc_index = int(doc_id_str)

    # Używamy indeksu do odnalezienia oryginalnego tekstu
    try:
        original_sentence = sentence_lookup[doc_index]
        print(f"  - Sim: {similarity:.4f} | Zdanie (ID: {doc_id_str}): {original_sentence}")
    except IndexError:
        print(f"  - Sim: {similarity:.4f} | BŁĄD: Nie znaleziono zdania dla ID: {doc_id_str}")

print("\n" + "=" * 50)
print("=== ETAP WNIOSKOWANIA ZAKOŃCZONY ===")
print("=" * 50)

# --- ETAP 3: Interaktywny tryb (opcjonalnie) ---
print("\n\nCzy chcesz przetestować własne zdania? (t/n): ", end="")
try:
    choice = input().strip().lower()

    if choice == 't' or choice == 'y':
        while True:
            print("\n" + "-" * 50)
            user_sentence = input("Wprowadź zdanie (lub 'q' aby zakończyć): ").strip()

            if user_sentence.lower() == 'q':
                print("Zakończono tryb interaktywny.")
                break

            if not user_sentence:
                print("Zdanie nie może być puste.")
                continue

            # Tokenizacja
            user_tokens = tokenizer.encode(user_sentence).tokens

            # Generowanie wektora
            user_vector = loaded_model.infer_vector(user_tokens, epochs=loaded_model.epochs)

            # Znajdowanie podobnych zdań
            similar_docs = loaded_model.dv.most_similar([user_vector], topn=5)

            print(f"\n5 najbardziej podobnych zdań do: \"{user_sentence}\"")
            for doc_id_str, similarity in similar_docs:
                doc_index = int(doc_id_str)
                try:
                    original_sentence = sentence_lookup[doc_index]
                    print(f"  - Sim: {similarity:.4f} | {original_sentence}")
                except IndexError:
                    print(f"  - Sim: {similarity:.4f} | BŁĄD: Nie znaleziono zdania dla ID: {doc_id_str}")
except EOFError:
    print("\n\nZakończono.")
except KeyboardInterrupt:
    print("\n\nPrzerwano przez użytkownika.")

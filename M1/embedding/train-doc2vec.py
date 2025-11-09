import numpy as np
import json
import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import glob
import time
from corpora import CORPORA_FILES

# Ustawienie logowania dla gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# files = CORPORA_FILES["NKJP"]
files = CORPORA_FILES["ALL"]
# files = CORPORA_FILES["PAN_TADEUSZ"]

OUTPUT_MODEL_FILE = "doc2vec_model_simple.model"
OUTPUT_SENTENCE_MAP = "doc2vec_model_sentence_map_simple.json"

# Parametry treningu Doc2Vec
VECTOR_LENGTH = 300
WINDOW_SIZE = 5
MIN_COUNT = 4
WORKERS = 10
EPOCHS = 100
SG_MODE = 0

print("\n" + "="*80)
print("  TRENING DOC2VEC Z PROSTĄ TOKENIZACJĄ (SPLIT)")
print("="*80)

# --- ETAP 1: Wczytanie, Tokenizacja i Przygotowanie Danych ---

# Wczytywanie i agregacja tekstu
raw_sentences = []
print("Wczytywanie tekstu z plików...")
print(f"Liczba plików do wczytania: {len(files)}")

for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            raw_sentences.extend(lines)
    except FileNotFoundError:
        print(f"OSTRZEŻENIE: Nie znaleziono pliku '{file}'. Pomijam.")
        continue
    except Exception as e:
        print(f"BŁĄD podczas przetwarzania pliku '{file}': {e}")
        continue

if not raw_sentences:
    print("BŁĄD: Korpus danych jest pusty.")
    raise ValueError("Korpus danych jest pusty.")
print(f"✓ Wczytano {len(raw_sentences)} zdań")

# Konwersja na listę tokenów UŻYWAJĄC PROSTEGO SPLIT()
print(f"\nTokenizacja {len(raw_sentences)} zdań używając prostego split()...")
tokenized_sentences = [sentence.split() for sentence in raw_sentences]

# Statystyki tokenizacji prostej
total_tokens = sum(len(tokens) for tokens in tokenized_sentences)
avg_tokens = total_tokens / len(tokenized_sentences) if tokenized_sentences else 0
print(f"✓ Tokenizacja split() zakończona:")
print(f"  ├─ Łączna liczba tokenów: {total_tokens:,}")
print(f"  └─ Średnia długość sekwencji: {avg_tokens:.1f} tokenów")

# Przykład tokenizacji prostej
print(f"\nPrzykład tokenizacji split():")
print(f"  Oryginał: '{raw_sentences[0][:80]}...'")
print(f"  Tokeny:   {tokenized_sentences[0][:10]}...")

# Przygotowanie danych dla Doc2Vec
tagged_data = [
    TaggedDocument(words=tokenized_sentences[i], tags=[str(i)])
    for i in range(len(tokenized_sentences))
]
print(f"\n✓ Przygotowano {len(tagged_data)} sekwencji TaggedDocument do treningu.")

# --- ETAP 2: Trening Doc2Vec ---
print("\n--- Rozpoczynanie Treningu Doc2Vec (SIMPLE) ---")
start_time = time.time()
model_d2v = Doc2Vec(
    tagged_data,
    vector_size=VECTOR_LENGTH,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    workers=WORKERS,
    epochs=EPOCHS,
    # negative=10,
    # ns_exponent=0.75,
    # sample=1e-5,
    dm=0 # Distributed Memory (PV-DM)
)
end_time = time.time()
print(f"✓ Trening zakończony pomyślnie. Czas trwania: {end_time - start_time:.2f}s")

# --- ETAP 3: Zapisywanie Wytrenowanego Modelu i Mapy ---
try:
    model_d2v.save(OUTPUT_MODEL_FILE)
    print(f"\n✓ Pełny model Doc2Vec (SIMPLE) zapisany jako: '{OUTPUT_MODEL_FILE}'.")

    with open(OUTPUT_SENTENCE_MAP, "w", encoding="utf-8") as f:
        json.dump(raw_sentences, f, ensure_ascii=False, indent=4)
    print(f"✓ Mapa zdań do ID zapisana jako: '{OUTPUT_SENTENCE_MAP}'.")

except Exception as e:
    print(f"BŁĄD podczas zapisu modelu/mapy: {e}")
    raise

print("\n" + "="*80)
print("  TRENING ZAKOŃCZONY POMYŚLNIE (SIMPLE)")
print("="*80)

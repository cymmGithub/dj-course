import numpy as np
import json
import logging
from gensim.models import Word2Vec
from tokenizers import Tokenizer
import os
import glob
# import z corpora (zakładam, że jest to plik pomocniczy)
from corpora import CORPORA_FILES # type: ignore

# Ustawienie logowania dla gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- KONFIGURACJA ŚCIEŻEK I PARAMETRÓW ---
# files = CORPORA_FILES["WOLNELEKTURY"]
# files = CORPORA_FILES["PAN_TADEUSZ"]
files = CORPORA_FILES["ALL"]

# TOKENIZER_FILE = "../tokenizer/tokenizers/custom_bpe_tokenizer.json"
# TOKENIZER_FILE = "../tokenizer/tokenizers/bielik-v1-tokenizer.json"
TOKENIZER_FILE = "../tokenizer/tokenizers/all-tokenizer.json"

OUTPUT_TENSOR_FILE = "embedding_tensor_cbow.npy"
OUTPUT_MAP_FILE = "embedding_token_to_index_map.json"
OUTPUT_MODEL_FILE = "embedding_word2vec_cbow_model.model"

# --- PARAMETRY TRENINGU WORD2VEC (CBOW) ---
# Poniższe parametry kontrolują proces uczenia modelu embeddingów słów.
# Dostosowanie tych wartości wpływa na jakość i charakterystykę wynikowych wektorów.

# VECTOR_LENGTH (wymiar wektora): Liczba wymiarów w przestrzeni wektorowej
# - Większe wartości (np. 100-300) mogą uchwycić więcej niuansów semantycznych
# - Mniejsze wartości (np. 20-50) są szybsze do trenowania i wymagają mniej pamięci
# - Dla małych korpusów (jak tutaj) lepiej użyć mniejszych wartości
VECTOR_LENGTH = 20

# WINDOW_SIZE (okno kontekstu): Maksymalna odległość między słowem a jego kontekstem
# - Określa ile słów po lewej i prawej stronie jest branych pod uwagę
# - WINDOW_SIZE=6 oznacza, że model patrzy na 6 słów przed i 6 po danym słowie
# - Większe okno (8-10) uchwytuje szerszy kontekst i ogólniejsze znaczenia
# - Mniejsze okno (2-4) koncentruje się na bezpośrednim sąsiedztwie i syntaktyce
WINDOW_SIZE = 5

# MIN_COUNT (minimalna częstość): Ignoruje słowa występujące rzadziej niż ta wartość
# - Filtruje rzadkie tokeny, które mogą być szumem lub błędami
# - MIN_COUNT=2 oznacza, że token musi wystąpić przynajmniej 2 razy w korpusie
# - Większe wartości (5-10) dają czystszy model, ale tracą rzadkie słowa
# - Mniejsze wartości (1-2) zachowują więcej słownictwa, ale mogą wprowadzać szum
MIN_COUNT = 2

# WORKERS (liczba wątków): Liczba równoległych procesów do treningu
# - Większa liczba przyspiesza trening na maszynach wielordzeniowych
# - Zazwyczaj ustawia się na liczbę rdzeni CPU (4-8 jest typowe)
WORKERS = 8

# EPOCHS (liczba epok): Ile razy model przechodzi przez cały korpus
# - Więcej epok (20-50) daje lepiej wytrenowany model, ale trwa dłużej
# - Za mało epok (1-5) może nie pozwolić modelowi nauczyć się wzorców
# - Za dużo epok (>100) może prowadzić do przeuczenia (overfitting)
EPOCHS = 40

# SAMPLE_RATE (próbkowanie): Częstość downsamplingu popularnych słów
# - Zmniejsza wpływ bardzo częstych słów (np. "i", "a", "w")
# - 1e-2 (0.01) to typowa wartość - około 1% najczęstszych słów jest pomijanych
# - Większe wartości (1e-1) agresywniej redukują częste słowa
# - Mniejsze wartości (1e-5) prawie nie filtrują
SAMPLE_RATE = 1e-2

# SG_MODE (tryb algorytmu): Wybór między CBOW a Skip-gram
# - 0 = CBOW (Continuous Bag of Words): przewiduje słowo na podstawie kontekstu
#   * Lepszy dla częstych słów
#   * Szybszy w treningu
#   * Dobry dla mniejszych korpusów
# - 1 = Skip-gram: przewiduje kontekst na podstawie słowa
#   * Lepszy dla rzadkich słów i małych korpusów
#   * Wolniejszy, ale często daje lepsze wyniki dla semantyki
SG_MODE = 0

try:
    print(f"Ładowanie tokenizera z pliku: {TOKENIZER_FILE}")
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku '{TOKENIZER_FILE}'. Upewnij się, że plik istnieje.")
    raise

# loading r& aggregating aw sentences from files
def aggregate_raw_sentences(files):
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

    if not raw_sentences:
        print("BŁĄD: Pliki wejściowe są puste lub nie zostały wczytane.")
        exit()
    return raw_sentences

raw_sentences = aggregate_raw_sentences(files)

# --- ETAP 1: Tokenizacja ---

print(f"\n{'='*80}")
print("  TOKENIZACJA KORPUSU")
print(f"{'='*80}")
print(f"\n📝 Przetwarzanie {len(raw_sentences):,} zdań...")
encodings = tokenizer.encode_batch(raw_sentences)

# Konwersja obiektów Encoding na listę list stringów (tokenów)
tokenized_sentences = [
    encoding.tokens for encoding in encodings
]

# Statystyki tokenizacji
total_tokens = sum(len(tokens) for tokens in tokenized_sentences)
avg_tokens = total_tokens / len(tokenized_sentences) if tokenized_sentences else 0

print(f"\n✓ Tokenizacja zakończona:")
print(f"  ├─ Liczba sekwencji: {len(tokenized_sentences):,}")
print(f"  ├─ Łączna liczba tokenów: {total_tokens:,}")
print(f"  └─ Średnia długość sekwencji: {avg_tokens:.1f} tokenów")
print(f"{'='*80}")

# --- ETAP 2: Trening Word2Vec (CBOW) ---

print("\n" + "="*80)
print("  TRENING MODELU WORD2VEC (CBOW)")
print("="*80)
print(f"\n⚙️  PARAMETRY TRENINGU:")
print(f"  ├─ Wymiar wektora: {VECTOR_LENGTH}")
print(f"  ├─ Okno kontekstu: {WINDOW_SIZE} (słów w każdą stronę)")
print(f"  ├─ Min. liczba wystąpień: {MIN_COUNT}")
print(f"  ├─ Liczba epok: {EPOCHS}")
print(f"  ├─ Tryb: {'CBOW' if SG_MODE == 0 else 'Skip-gram'}")
print(f"  └─ Liczba wątków: {WORKERS}")
print(f"\n🔄 Rozpoczynanie treningu...")
print(f"{'─'*80}\n")

model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=VECTOR_LENGTH,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    workers=WORKERS,
    sg=SG_MODE,  # 0: CBOW
    epochs=EPOCHS,
    sample=SAMPLE_RATE,
)

print(f"\n{'─'*80}")
print("✓ Trening zakończony pomyślnie!")
print(f"{'='*80}")

# --- ETAP 3: Eksport i Zapis Wyników ---

print("\n" + "="*80)
print("  EKSPORT WYNIKÓW TRENINGU")
print("="*80)

# Eksport tensora embeddingowego
embedding_matrix_np = model.wv.vectors
embedding_matrix_tensor = np.array(embedding_matrix_np, dtype=np.float32)

print(f"\n📊 STATYSTYKI MODELU:")
print(f"  ├─ Liczba unikalnych tokenów: {embedding_matrix_tensor.shape[0]:,}")
print(f"  ├─ Wymiar wektorów: {embedding_matrix_tensor.shape[1]}")
print(f"  └─ Rozmiar tensora: {embedding_matrix_tensor.shape} (Tokeny × Wymiar)")

print(f"\n💾 ZAPISYWANIE PLIKÓW:")

# 1. Zapisanie tensora NumPy (.npy)
np.save(OUTPUT_TENSOR_FILE, embedding_matrix_tensor)
print(f"  ✓ Tensor embeddingowy: '{OUTPUT_TENSOR_FILE}'")
print(f"    (format NumPy, rozmiar: {embedding_matrix_tensor.nbytes / 1024:.2f} KB)")

# 2. Zapisanie mapowania tokenów na indeksy
token_to_index = {token: model.wv.get_index(token) for token in model.wv.index_to_key}
with open(OUTPUT_MAP_FILE, "w", encoding="utf-8") as f:
    json.dump(token_to_index, f, ensure_ascii=False, indent=4)
print(f"  ✓ Mapa token→indeks: '{OUTPUT_MAP_FILE}'")
print(f"    (format JSON, {len(token_to_index):,} wpisów)")

# 3. Zapisanie całego modelu gensim (opcjonalne, ale zalecane)
model.save(OUTPUT_MODEL_FILE)
print(f"  ✓ Pełny model Word2Vec: '{OUTPUT_MODEL_FILE}'")
print(f"    (format Gensim, zawiera wszystkie dane treningu)")

print(f"\n{'='*80}")
print("  TRENING ZAKOŃCZONY")
print(f"{'='*80}")
print(f"\n💡 Aby wizualizować wyniki, uruchom: python visualize-cbow.py")

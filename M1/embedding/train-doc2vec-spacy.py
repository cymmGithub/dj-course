import numpy as np
import json
import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import glob
import time
import pickle
import hashlib
from corpora import CORPORA_FILES
import spacy

# Ustawienie logowania dla gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# files = CORPORA_FILES["NKJP"]
files = CORPORA_FILES["ALL"]
# files = CORPORA_FILES["PAN_TADEUSZ"]

OUTPUT_MODEL_FILE = "doc2vec_model_spacy.model"
OUTPUT_SENTENCE_MAP = "doc2vec_model_sentence_map_spacy.json"

# Parametry treningu Doc2Vec
VECTOR_LENGTH = 100
WINDOW_SIZE = 5
MIN_COUNT = 4
WORKERS = 10
EPOCHS = 100
SG_MODE = 0

print("\n" + "="*80)
print("  TRENING DOC2VEC Z TOKENIZACJĄ SPACY + LEMMATYZACJA")
print("="*80)

# --- ETAP 0: Wczytanie modelu spaCy ---
print("\nWczytywanie modelu spaCy dla języka polskiego...")
try:
    nlp = spacy.load("pl_core_news_sm")
    print("✓ Załadowano model: pl_core_news_sm")
except OSError:
    print("\n⚠️  BŁĄD: Nie znaleziono modelu spaCy dla języka polskiego.")
    print("\nAby zainstalować, uruchom:")
    print("  python -m spacy download pl_core_news_sm")
    print("\nLub dla lepszej jakości (większy model):")
    print("  python -m spacy download pl_core_news_md")
    print("  python -m spacy download pl_core_news_lg")
    raise

# --- ETAP 1: Wczytanie, Tokenizacja i Przygotowanie Danych ---

# Wczytywanie i agregacja tekstu
raw_sentences = []
print("\nWczytywanie tekstu z plików...")
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

# Funkcja tokenizacji spaCy z lemmatyzacją
def tokenize_with_spacy(text):
    """
    Tokenizacja używając spaCy z lemmatyzacją dla języka polskiego.

    - Konwertuje słowa do formy podstawowej (lemma)
    - Usuwa interpunkcję i białe znaki
    - Konwertuje do małych liter
    """
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_space and token.text.strip()
    ]
    return tokens

# --- CACHE MECHANISM ---
def generate_corpus_hash(sentences):
    """Generuje hash SHA256 korpusu do wykrywania zmian."""
    # Użyj pierwsze 1000 zdań jako reprezentacja + liczba zdań
    corpus_str = "\n".join(sentences[:1000]) + f"\n__TOTAL__{len(sentences)}"
    return hashlib.sha256(corpus_str.encode('utf-8')).hexdigest()[:16]

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

corpus_hash = generate_corpus_hash(raw_sentences)
cache_file = os.path.join(CACHE_DIR, f"lemmatized_spacy_{corpus_hash}.pkl")
cache_meta_file = os.path.join(CACHE_DIR, f"lemmatized_spacy_{corpus_hash}_meta.json")

# Konwersja na listę tokenów UŻYWAJĄC SPACY + LEMMATYZACJI (z CACHE)
print(f"\nTokenizacja {len(raw_sentences)} zdań używając spaCy + lemmatyzacja...")

tokenized_sentences = None

# Sprawdź cache
if os.path.exists(cache_file) and os.path.exists(cache_meta_file):
    try:
        # Wczytaj metadata
        with open(cache_meta_file, 'r') as f:
            meta = json.load(f)

        # Sprawdź czy liczba zdań się zgadza
        if meta.get('num_sentences') == len(raw_sentences):
            print(f"✓ Znaleziono cache lemmatyzacji: {os.path.basename(cache_file)}")
            print("  Ładowanie z cache...")

            start_load = time.time()
            with open(cache_file, 'rb') as f:
                tokenized_sentences = pickle.load(f)
            end_load = time.time()

            print(f"✓ Załadowano {len(tokenized_sentences):,} zdań z cache w {end_load - start_load:.2f}s")
            print(f"  💡 Oszczędność czasu: ~{meta.get('tokenization_time', 0):.0f}s!")
        else:
            print(f"⚠️  Cache nieaktualny (inna liczba zdań: {meta.get('num_sentences')} vs {len(raw_sentences)})")
            print("  Re-tokenizacja...")
    except Exception as e:
        print(f"⚠️  Błąd podczas ładowania cache: {e}")
        print("  Re-tokenizacja...")
else:
    print("  Brak cache. Wykonuję lemmatyzację...")

# Jeśli brak cache lub nieaktualny - lemmatyzuj
if tokenized_sentences is None:
    print("(To może potrwać kilka minut...)")
    start_tokenization = time.time()

    tokenized_sentences = []
    batch_size = 1000  # Przetwarzanie w partiach dla szybkości

    for i in range(0, len(raw_sentences), batch_size):
        batch = raw_sentences[i:i+batch_size]
        # Wykorzystanie pipe() dla wydajności
        for doc in nlp.pipe(batch, n_process=1, batch_size=50):
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_punct and not token.is_space and token.text.strip()
            ]
            tokenized_sentences.append(tokens)

        # Postęp
        if (i + batch_size) % 10000 == 0:
            print(f"  Przetworzone: {min(i + batch_size, len(raw_sentences)):,} / {len(raw_sentences):,} zdań")

    end_tokenization = time.time()
    tokenization_time = end_tokenization - start_tokenization

    # Zapisz do cache
    print(f"\n💾 Zapisywanie cache lemmatyzacji...")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(tokenized_sentences, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Zapisz metadata
        meta = {
            'num_sentences': len(raw_sentences),
            'tokenization_time': tokenization_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'corpus_hash': corpus_hash,
            'spacy_model': 'pl_core_news_sm'
        }
        with open(cache_meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

        cache_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
        print(f"✓ Cache zapisany: {os.path.basename(cache_file)} ({cache_size_mb:.1f} MB)")
        print(f"  💡 Następne uruchomienia będą ~{tokenization_time:.0f}s szybsze!")
    except Exception as e:
        print(f"⚠️  Nie udało się zapisać cache: {e}")
        print("  (Nie wpływa to na trening - kontynuuję)")

# Statystyki tokenizacji spaCy
total_tokens = sum(len(tokens) for tokens in tokenized_sentences)
avg_tokens = total_tokens / len(tokenized_sentences) if tokenized_sentences else 0
print(f"\n✓ Tokenizacja spaCy zakończona:")
print(f"  ├─ Łączna liczba tokenów (po lemmatyzacji): {total_tokens:,}")
print(f"  └─ Średnia długość sekwencji: {avg_tokens:.1f} tokenów")

# Przykład tokenizacji spaCy
print(f"\nPrzykład tokenizacji spaCy + lemmatyzacja:")
print(f"  Oryginał:     '{raw_sentences[0][:80]}...'")
print(f"  Tokeny:       {tokenized_sentences[0][:10]}...")

# Dla porównania - tokenizacja split()
split_example = raw_sentences[0].split()[:10]
print(f"\n  (porównaj z split(): {split_example}...)")

# Przygotowanie danych dla Doc2Vec
tagged_data = [
    TaggedDocument(words=tokenized_sentences[i], tags=[str(i)])
    for i in range(len(tokenized_sentences))
]
print(f"\n✓ Przygotowano {len(tagged_data)} sekwencji TaggedDocument do treningu.")

# --- ETAP 2: Trening Doc2Vec ---
print("\n--- Rozpoczynanie Treningu Doc2Vec (SPACY) ---")
start_time = time.time()
model_d2v = Doc2Vec(
    tagged_data,
    vector_size=VECTOR_LENGTH,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    workers=WORKERS,
    epochs=EPOCHS,
    negative=10,
    ns_exponent=0.75,
    sample=1e-5,
    dm=0  # Distributed Memory (PV-DM)
)
end_time = time.time()
print(f"✓ Trening zakończony pomyślnie. Czas trwania: {end_time - start_time:.2f}s")

# Statystyki modelu
print(f"\n📊 Statystyki wytrenowanego modelu (SPACY):")
print(f"  ├─ Rozmiar słownika: {len(model_d2v.wv):,} unikalnych tokenów")
print(f"  ├─ Wymiar wektora: {model_d2v.vector_size}")
print(f"  └─ Liczba epok: {model_d2v.epochs}")

# --- ETAP 3: Zapisywanie Wytrenowanego Modelu i Mapy ---
try:
    model_d2v.save(OUTPUT_MODEL_FILE)
    print(f"\n✓ Pełny model Doc2Vec (SPACY) zapisany jako: '{OUTPUT_MODEL_FILE}'.")

    with open(OUTPUT_SENTENCE_MAP, "w", encoding="utf-8") as f:
        json.dump(raw_sentences, f, ensure_ascii=False, indent=4)
    print(f"✓ Mapa zdań do ID zapisana jako: '{OUTPUT_SENTENCE_MAP}'.")

except Exception as e:
    print(f"BŁĄD podczas zapisu modelu/mapy: {e}")
    raise

print("\n" + "="*80)
print("  TRENING ZAKOŃCZONY POMYŚLNIE (SPACY)")
print("="*80)
print("\nℹ️  Model używa tokenizacji spaCy z lemmatyzacją, co oznacza:")
print("   • Różne formy tego samego słowa są redukowane do formy podstawowej")
print("   • Przykład: 'książki', 'książką', 'książek' → 'książka'")
print("   • Interpunkcja jest usuwana automatycznie")
print("   • Wszystkie tokeny są konwertowane do małych liter")
print("\n💾 CACHE INFO:")
print(f"   • Cache zapisany w: {CACHE_DIR}/")
print(f"   • Aby wyczyścić cache: usuń folder {CACHE_DIR}/")
print(f"   • Następne uruchomienie: ~instant lemmatization! 🚀")

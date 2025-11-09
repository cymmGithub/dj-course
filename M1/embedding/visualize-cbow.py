import numpy as np
from gensim.models import Word2Vec
from tokenizers import Tokenizer
import os

# --- KONFIGURACJA ŚCIEŻEK ---

# TOKENIZER_FILE = "../tokenizer/tokenizers/custom_bpe_tokenizer.json"
TOKENIZER_FILE = "../tokenizer/tokenizers/all-tokenizer.json"
# TOKENIZER_FILE = "../tokenizer/tokenizers/bielik-v3-tokenizer.json"

MODEL_FILE = "embedding_word2vec_cbow_model.model"

# Parametr używany w funkcji get_word_vector_and_similar (dla komunikatów błędów)
# Powinien być taki sam jak MIN_COUNT użyty podczas treningu
MIN_COUNT = 2

# --- WCZYTYWANIE MODELU I TOKENIZERA ---

print("="*80)
print("  WCZYTYWANIE MODELU I TOKENIZERA")
print("="*80)

try:
    print(f"\n📂 Wczytywanie tokenizera z: '{TOKENIZER_FILE}'")
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    print("   ✓ Tokenizer wczytany pomyślnie")
except FileNotFoundError:
    print(f"\n❌ BŁĄD: Nie znaleziono pliku '{TOKENIZER_FILE}'")
    print("   Upewnij się, że plik istnieje.")
    exit(1)

try:
    print(f"\n📂 Wczytywanie modelu Word2Vec z: '{MODEL_FILE}'")
    model = Word2Vec.load(MODEL_FILE)
    print("   ✓ Model wczytany pomyślnie")
except FileNotFoundError:
    print(f"\n❌ BŁĄD: Nie znaleziono pliku '{MODEL_FILE}'")
    print("   Najpierw uruchom skrypt treningu: python train-cbow.py")
    exit(1)

# Informacje o modelu
print(f"\n📊 INFORMACJE O MODELU:")
print(f"  ├─ Liczba tokenów w słowniku: {len(model.wv.index_to_key):,}")
print(f"  ├─ Wymiar wektorów: {model.wv.vector_size}")
print(f"  └─ Algorytm: {'CBOW' if model.sg == 0 else 'Skip-gram'}")
print("="*80)

# --- FUNKCJA: OBLICZANIE WEKTORA DLA CAŁEGO SŁOWA ---

def get_word_vector_and_similar(word: str, tokenizer: Tokenizer, model: Word2Vec, topn: int = 20):
    """
    Oblicza wektor dla całego słowa poprzez uśrednienie wektorów jego tokenów składowych.

    Args:
        word: Słowo do analizy
        tokenizer: Tokenizer do rozbicia słowa na tokeny
        model: Wytrenowany model Word2Vec
        topn: Liczba najbardziej podobnych tokenów do zwrócenia

    Returns:
        tuple: (word_vector, similar_tokens) lub (None, None) w przypadku błędu
    """
    # Tokenizacja słowa na tokeny podwyrazowe
    # Używamy .encode(), aby otoczyć słowo spacjami, co imituje kontekst w zdaniu
    # Ważne: tokenizator BPE/SentencePiece musi widzieć spację, by dodać prefiks '_'
    encoding = tokenizer.encode(" " + word + " ")
    word_tokens = [t.strip() for t in encoding.tokens if t.strip()] # Usuń puste tokeny

    # Usuwamy tokeny początku/końca sekwencji, jeśli zostały dodane przez tokenizator
    if word_tokens and word_tokens[0] in ['[CLS]', '<s>', '<s>', 'Ġ']:
        word_tokens = word_tokens[1:]
    if word_tokens and word_tokens[-1] in ['[SEP]', '</s>', '</s>']:
        word_tokens = word_tokens[:-1]

    valid_vectors = []
    missing_tokens = []

    # 1. Zbieranie wektorów dla każdego tokenu
    for token in word_tokens:
        if token in model.wv:
            # Użycie tokenu ze spacją (np. '_ryż') lub bez (np. 'szlach')
            valid_vectors.append(model.wv[token])
        else:
            # W tym miejscu token może być zbyt rzadki i pominięty przez MIN_COUNT
            missing_tokens.append(token)

    if not valid_vectors:
        # Kod do obsługi, gdy żaden token nie ma wektora
        if missing_tokens:
            print(f"BŁĄD: Żaden z tokenów składowych ('{word_tokens}') nie znajduje się w słowniku (MIN_COUNT={MIN_COUNT}).")
        else:
            print(f"BŁĄD: Słowo '{word}' nie zostało przetworzone na wektory (sprawdź tokenizację).")
        return None, None

    # 2. Uśrednianie wektorów
    # Wektor dla całego słowa to średnia wektorów jego tokenów składowych
    word_vector = np.mean(valid_vectors, axis=0)

    # 3. Znalezienie najbardziej podobnych tokenów
    similar_words = model.wv.most_similar(
        positive=[word_vector],
        topn=topn
    )

    return word_vector, similar_words

# --- ANALIZA PODOBIEŃSTWA SŁÓW ---

print("\n" + "="*80)
print("  ANALIZA PODOBIEŃSTWA SŁÓW - WYNIKI MODELU WORD2VEC (CBOW)")
print("="*80)
print("\nModel analizuje podobieństwo semantyczne słów na podstawie ich kontekstu.")
print("Im wyższa wartość podobieństwa (0.0 - 1.0), tym bardziej słowa są związane.\n")

# Przykłady słów do testowania
words_to_test = ['wojsko', 'szlachta', 'choroba', 'król']

for i, word in enumerate(words_to_test, 1):
    word_vector, similar_tokens = get_word_vector_and_similar(word, tokenizer, model, topn=10)

    if word_vector is not None:
        print(f"\n{'─'*80}")
        print(f"  [{i}/{len(words_to_test)}] SŁOWO TESTOWE: '{word.upper()}'")
        print(f"{'─'*80}")

        # Informacja o tokenizacji
        tokens = tokenizer.encode(" " + word + " ").tokens
        tokens_clean = [t.strip() for t in tokens if t.strip() and t not in ['[CLS]', '[SEP]', '<s>', '</s>']]
        print(f"  📝 Tokenizacja: {' + '.join(tokens_clean)}")

        # Wyświetlanie wektora (pierwsze 5 elementów)
        print(f"  🔢 Wektor (początek): [{', '.join([f'{v:.3f}' for v in word_vector[:5]])}...]")

        print(f"\n  🎯 TOP 10 NAJBARDZIEJ PODOBNYCH TOKENÓW:")
        print(f"  {'─'*76}")
        print(f"  {'  Pozycja':<12} {'Token':<35} {'Podobieństwo':<15}")
        print(f"  {'─'*76}")

        for rank, (token, similarity) in enumerate(similar_tokens, 1):
            # Wizualizacja podobieństwa za pomocą paska
            bar_length = int(similarity * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)

            print(f"  {rank:>2}. {token:<35} {similarity:.4f}  {bar}")

        print(f"  {'─'*76}")

# --- ANALIZA ANALOGII WEKTOROWYCH ---

print(f"\n\n{'='*80}")
print("  ANALIZA ANALOGII WEKTOROWYCH")
print("="*80)
print("\nAnalogie wektorowe pokazują związki semantyczne między słowami.")
print("Model łączy wektory słów, aby znaleźć koncepcje powiązane z ich kombinacją.\n")

tokens_analogy = ['mężczyzna', 'zabawa']

# Używamy uśredniania wektorów dla tokenów
if tokens_analogy[0] in model.wv and tokens_analogy[1] in model.wv:
    similar_to_combined = model.wv.most_similar(
        positive=tokens_analogy,
        topn=10
    )

    print(f"{'─'*80}")
    print(f"  🔗 KOMBINACJA TOKENÓW: {' + '.join(tokens_analogy)}")
    print(f"{'─'*80}")
    print(f"  Interpretacja: Szukamy tokenów semantycznie powiązanych")
    print(f"  z koncepcją łączącą oba słowa wejściowe.\n")

    print(f"  🎯 TOP 10 NAJBARDZIEJ PODOBNYCH TOKENÓW:")
    print(f"  {'─'*76}")
    print(f"  {'  Pozycja':<12} {'Token':<35} {'Podobieństwo':<15}")
    print(f"  {'─'*76}")

    for rank, (token, similarity) in enumerate(similar_to_combined, 1):
        # Wizualizacja podobieństwa za pomocą paska
        bar_length = int(similarity * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)

        print(f"  {rank:>2}. {token:<35} {similarity:.4f}  {bar}")

    print(f"  {'─'*76}")
else:
    print(f"\n⚠️  OSTRZEŻENIE: Co najmniej jeden z tokenów '{tokens_analogy}' nie znajduje się w słowniku.")
    print(f"    Może to być spowodowane zbyt rzadkim występowaniem (MIN_COUNT={MIN_COUNT}).")

print(f"\n{'='*80}")
print("  KONIEC ANALIZY")
print("="*80)

# --- DODATKOWE OPCJE ---

print(f"\n\n💡 WSKAZÓWKA:")
print(f"  Możesz modyfikować ten skrypt, aby testować własne słowa lub analogie.")
print(f"  Zmień listę 'words_to_test' lub 'tokens_analogy', aby eksperymentować!")

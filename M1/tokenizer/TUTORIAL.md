# Jak zbudować własny tokenizer - Tutorial krok po kroku

## Co to jest tokenizer?

Tokenizer to narzędzie, które rozbija tekst na mniejsze jednostki (tokeny) zrozumiałe dla modeli językowych.

**Przykład:**
- Tekst: "Litwo! Ojczyzno moja!"
- Tokeny: ["Litwo", "!", "Ojczyzno", "moja", "!"]

## Dlaczego to ważne?

Modele językowe (LLM) nie rozumieją tekstu - rozumieją liczby. Tokenizer:
1. Zamienia tekst na tokeny
2. Każdy token ma swój unikalny ID (liczbę)
3. Model przetwarza te liczby

**Im lepszy tokenizer, tym:**
- Mniej tokenów potrzeba (tańsze API)
- Model lepiej rozumie język (wyższa jakość)

---

## Rodzaje tokenizerów

### 1. BPE (Byte Pair Encoding)
- Najpopularniejszy dla języków o bogatej fleksji (jak polski)
- Uczy się najczęstszych par znaków i łączy je
- Używany w: GPT, Bielik

### 2. WordPiece
- Podobny do BPE
- Używany w: BERT

### 3. SentencePiece
- Nie wymaga białych znaków (dobry dla języków azjatyckich)
- Używany w: T5, mBERT

**W tym tutorialu używamy BPE - najprostszy i najskuteczniejszy.**

---

## Krok 1: Instalacja środowiska

### Wymagania:
- Python 3.7+
- pip

### Zainstaluj bibliotekę:

```bash
cd M1/tokenizer
pip install tokenizers
```

---

## Krok 2: Przygotowanie danych treningowych

Tokenizer musi "nauczyć się" języka na podstawie korpusu tekstów.

### Dostępne korpusy:

**A) WOLNELEKTURY** - Literatura polska (zalecane na początek)
- Pan Tadeusz, Lalka, Quo Vadis, etc.
- Lokalizacja: `../korpus-wolnelektury/*.txt`

**B) NKJP** - Narodowy Korpus Języka Polskiego
- Oficjalny korpus naukowy
- Lokalizacja: `../korpus-nkjp/output/*.txt`

### Sprawdź dostępne pliki:

```bash
python corpora.py
```

---

## Krok 3: Podstawowy tokenizer (minimalny przykład)

Stwórz plik `moj-tokenizer.py`:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. Inicjalizacja modelu BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. Pre-tokenizer - dzieli tekst na spacje
tokenizer.pre_tokenizer = Whitespace()

# 3. Konfiguracja trenera
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[PAD]"],  # Tokeny specjalne
    vocab_size=5000,                     # Rozmiar słownika
    min_frequency=2                      # Min. wystąpień
)

# 4. Trening na jednym pliku
FILES = ["../korpus-wolnelektury/latarnik.txt"]
tokenizer.train(FILES, trainer=trainer)

# 5. Zapis
tokenizer.save("tokenizers/moj_tokenizer.json")

# 6. Test
encoded = tokenizer.encode("Witaj świecie!")
print("Tokeny:", encoded.tokens)
print("IDs:", encoded.ids)
```

### Uruchom:

```bash
python moj-tokenizer.py
```

---

## Krok 4: Zaawansowana konfiguracja

### Parametry trenera:

```python
trainer = BpeTrainer(
    # Tokeny specjalne dla różnych zadań NLP
    special_tokens=[
        "[UNK]",   # Nieznany token
        "[CLS]",   # Początek sekwencji (classification)
        "[SEP]",   # Separator
        "[PAD]",   # Padding (wyrównanie długości)
        "[MASK]"   # Maskowanie (BERT-style)
    ],

    # Rozmiar słownika
    # - Mały (5000): szybki, mało pamięci, gorsze pokrycie
    # - Średni (30000): dobry balans
    # - Duży (50000+): wolny, dużo pamięci, najlepsze pokrycie
    vocab_size=30000,

    # Minimalna częstość występowania
    # - 1: wszystkie słowa (ryzyko overfitting)
    # - 2-3: dobry balans (zalecane)
    # - 5+: tylko częste słowa
    min_frequency=2,

    # Limit sekwencji (opcjonalne)
    # limit_alphabet=1000,  # Max unikalnych znaków
)
```

### Wybór korpusu:

```python
from corpora import CORPORA_FILES

# Opcja 1: Pojedynczy plik
FILES = ["../korpus-wolnelektury/latarnik.txt"]

# Opcja 2: Wszystkie pliki Pan Tadeusz
FILES = [str(f) for f in CORPORA_FILES["PAN_TADEUSZ"]]

# Opcja 3: Całe Wolne Lektury (ZALECANE)
FILES = [str(f) for f in CORPORA_FILES["WOLNELEKTURY"]]

# Opcja 4: NKJP (korpus naukowy)
FILES = [str(f) for f in CORPORA_FILES["NKJP"]]

# Opcja 5: Wszystko
FILES = [str(f) for f in CORPORA_FILES["ALL"]]
```

---

## Krok 5: Kompletny przykład - tokenizer na całym korpusie

Plik: `tokenizer-build-complete.py`

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from corpora import CORPORA_FILES

# Nazwa pliku wyjściowego
TOKENIZER_OUTPUT = "tokenizers/polski_tokenizer.json"

# 1. Inicjalizacja
print("Inicjalizacja tokenizera...")
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 2. Konfiguracja
print("Konfiguracja trenera...")
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=30000,
    min_frequency=2
)

# 3. Wybór korpusu
FILES = [str(f) for f in CORPORA_FILES["WOLNELEKTURY"]]
print(f"Trening na {len(FILES)} plikach...")

# 4. Trening
tokenizer.train(FILES, trainer=trainer)

# 5. Zapis
tokenizer.save(TOKENIZER_OUTPUT)
print(f"Zapisano: {TOKENIZER_OUTPUT}")

# 6. Testy
test_texts = [
    "Litwo! Ojczyzno moja!",
    "Sztuczna inteligencja",
    "Witaj świecie!"
]

print("\n=== TESTY ===")
for text in test_texts:
    encoded = tokenizer.encode(text)
    print(f"\nTekst: {text}")
    print(f"Tokeny: {encoded.tokens}")
    print(f"Liczba: {len(encoded.tokens)}")
```

---

## Krok 6: Testowanie tokenizera

### Użyj gotowego skryptu:

```bash
# Zmień w pliku tokenize-pan-tadeusz.py linię:
# TOKENIZER = "bielik-v3"
# na:
# TOKENIZER = "bpe"  # lub nazwę Twojego tokenizera

python tokenize-pan-tadeusz.py
```

### Wizualizacja:

```bash
# W tokenize-visualize.py zmień:
# TOKENIZER_PATH = "tokenizers/custom_bpe_tokenizer.json"
# na:
# TOKENIZER_PATH = "tokenizers/moj_tokenizer.json"

python tokenize-visualize.py
```

---

## Krok 7: Porównanie z innymi tokenizerami

Użyj profesjonalnych tokenizerów jako punktu odniesienia:

```python
from tokenizers import Tokenizer

# Załaduj różne tokenizery
twoj = Tokenizer.from_file("tokenizers/moj_tokenizer.json")
bielik = Tokenizer.from_file("tokenizers/bielik-v3-tokenizer.json")

text = "Litwo! Ojczyzno moja! ty jesteś jak zdrowie."

# Porównaj
print("Twój tokenizer:", len(twoj.encode(text).tokens), "tokenów")
print("Bielik:", len(bielik.encode(text).tokens), "tokenów")
```

**Cel:** Twój tokenizer powinien mieć podobną liczbę tokenów jak Bielik.

---

## FAQ - Częste problemy

### Problem 1: Tokenizer dzieli słowa na pojedyncze litery

**Przyczyna:** Za mały korpus treningowy

**Rozwiązanie:**
```python
# Zamiast:
FILES = ["../korpus-wolnelektury/latarnik.txt"]

# Użyj:
FILES = [str(f) for f in CORPORA_FILES["WOLNELEKTURY"]]
```

### Problem 2: Trening trwa bardzo długo

**Przyczyna:** Za duży korpus + za duży vocab_size

**Rozwiązanie:**
```python
# Zmniejsz rozmiar słownika
vocab_size=10000  # zamiast 50000

# Lub użyj mniejszego korpusu
FILES = [str(f) for f in CORPORA_FILES["PAN_TADEUSZ"]]
```

### Problem 3: "Unknown token" dla polskich znaków

**Przyczyna:** Brak obsługi UTF-8

**Rozwiązanie:** Sprawdź encoding plików:
```python
with open(file, 'r', encoding='utf-8') as f:
    content = f.read()
```

---

## Porady i Best Practices

### 1. Rozmiar korpusu

| Korpus | Pliki | Rozmiar | Vocab Size | Czas treningu |
|--------|-------|---------|------------|---------------|
| latarnik.txt | 1 | ~20KB | 5,000 | 1s |
| Pan Tadeusz | 12 | ~500KB | 15,000 | 5s |
| Wolne Lektury | ~30 | ~50MB | 30,000 | 30s |
| NKJP | ~1000 | ~500MB | 50,000 | 5min |

**Zalecenie:** Zacznij od Wolnych Lektur.

### 2. Vocabulary Size

- **5,000 - 10,000:** Prototypy, testy
- **30,000:** Produkcja (balans)
- **50,000+:** Profesjonalne modele

**Zasada:** vocab_size ≈ 0.1% liczby słów w korpusie

### 3. Min Frequency

- **1:** Wszystko (ryzyko noise)
- **2-3:** Standardowe (zalecane)
- **5+:** Tylko popularne

---

## Następne kroki

### 1. Budowa własnego tokenizera
```bash
# Edytuj parametry w tokenizer-build.py
python tokenizer-build.py
```

### 2. Test na różnych tekstach
```bash
python tokenize-visualize.py
```

### 3. Porównanie z Bielik
```bash
python tokenize-pan-tadeusz.py
```

### 4. Eksperymentuj!
- Zmień vocab_size
- Użyj innego korpusu
- Dodaj własne special tokens
- Przetestuj na różnych tekstach

---

## Dodatkowe zasoby

### Dokumentacja
- HuggingFace Tokenizers: https://huggingface.co/docs/tokenizers
- BPE Paper: https://arxiv.org/abs/1508.07909

### Korpusy
- Wolne Lektury: https://wolnelektury.pl
- NKJP: http://nkjp.pl

### Modele używające tych tokenizerów
- Bielik (Polski GPT): https://huggingface.co/speakleash
- GPT-2, GPT-3: OpenAI
- LLaMA: Meta

---

## Podsumowanie

### Podstawowy przepływ pracy:

```
1. Wybierz korpus → CORPORA_FILES["WOLNELEKTURY"]
2. Ustaw parametry → vocab_size=30000, min_frequency=2
3. Wytrenuj → tokenizer.train(FILES, trainer)
4. Zapisz → tokenizer.save("moj_tokenizer.json")
5. Testuj → tokenizer.encode(text)
```

### Kluczowe parametry:

| Parametr | Wartość zalecana | Efekt |
|----------|------------------|-------|
| vocab_size | 30,000 | Balans jakość/szybkość |
| min_frequency | 2 | Ignoruje rzadkie słowa |
| special_tokens | [UNK, PAD, CLS, SEP, MASK] | Standardowe tokeny NLP |

**Gotowy do budowy? Otwórz `tokenizer-build.py` i zacznij eksperymentować!**

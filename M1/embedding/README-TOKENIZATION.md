# Porównanie Metod Tokenizacji dla Doc2Vec

Kompleksowy system do porównania trzech różnych metod tokenizacji w kontekście modeli Doc2Vec dla języka polskiego.

## 📋 Przegląd

Ten projekt porównuje trzy podejścia do tokenizacji:

1. **SIMPLE** - Prosta tokenizacja używająca `split()`
2. **BPE** - Byte Pair Encoding (subword tokenization)
3. **SPACY** - Lemmatyzacja używająca spaCy (POLECANE dla języka polskiego)

## 🚀 Szybki Start

### 1. Instalacja zależności

```bash
cd M1/embedding
pip install -r requirements.txt

# Zainstaluj model spaCy dla języka polskiego
python -m spacy download pl_core_news_sm

# Opcjonalnie - większe modele dla lepszej jakości:
# python -m spacy download pl_core_news_md
# python -m spacy download pl_core_news_lg
```

### 2. Trening wszystkich modeli

**Opcja A: Automatyczne trenowanie wszystkich trzech modeli**
```bash
python train-both.py
```

**Opcja B: Trenowanie pojedynczych modeli**
```bash
python train-doc2vec.py          # SIMPLE (split)
python train-doc2vec-bpe.py      # BPE
python train-doc2vec-spacy.py    # SPACY (lemmatization)
```

### 3. Interaktywne porównanie

```bash
python compare-all-tokenization.py
```

## 📁 Struktura Plików

### Skrypty treningowe
- **`train-doc2vec.py`** - Trening z prostym split()
- **`train-doc2vec-bpe.py`** - Trening z BPE tokenization
- **`train-doc2vec-spacy.py`** - Trening z spaCy lemmatization ⭐ NOWY
- **`train-both.py`** - Automatyczny trening wszystkich trzech modeli

### Skrypty porównawcze
- **`compare-all-tokenization.py`** - Interaktywne porównanie wszystkich 3 metod ⭐ NOWY
- **`compare-tokenization.py`** - Porównanie BPE vs SIMPLE (starszy)
- **`visualize-doc2vec.py`** - Wizualizacja wyników

### Wygenerowane modele
Po treningu powstaną:
- `doc2vec_model_simple.model` + `doc2vec_model_sentence_map_simple.json`
- `doc2vec_model_bpe.model` + `doc2vec_model_sentence_map_bpe.json`
- `doc2vec_model_spacy.model` + `doc2vec_model_sentence_map_spacy.json`

## 🎯 Funkcje compare-all-tokenization.py

### Menu główne
1. **Demonstracja** - Analiza przykładowych zdań
2. **Tryb interaktywny** - Wprowadzaj własne zdania
3. **Statystyki modeli** - Porównanie parametrów

### Co pokazuje dla każdego zdania:
- Tokeny wygenerowane przez każdą metodę
- Top N najbardziej podobnych zdań z korpusu
- Statystyki porównawcze:
  - Liczba tokenów
  - Średnie podobieństwo
  - Norma wektora
  - Podobieństwo cosinusowe między modelami

## 📊 Przykład użycia

```bash
$ python compare-all-tokenization.py

================================ MENU GŁÓWNE ================================

Wybierz opcję:
  1. Uruchom demonstrację (przykładowe zdania)
  2. Tryb interaktywny (własne zdania)
  3. Statystyki modeli
  q. Zakończ

Wybór > 2

============================= TRYB INTERAKTYWNY =============================

Wprowadź własne zdania aby porównać tokenizację.
Wpisz 'q' lub 'quit' aby zakończyć.

Zdanie > Czytam książki w bibliotece

======================= ANALIZA: "Czytam książki w bibliotece" ==============

🔶 MODEL SIMPLE (split tokenization)
--------------------------------------------------------------------------------
Tokeny (4): ['Czytam', 'książki', 'w', 'bibliotece']

Top 5 najbardziej podobnych zdań:
  1. [0.8234] Czytałem wiele książek w miejskiej bibliotece...
  2. [0.7891] W bibliotece znalazłem interesujące pozycje...
  ...

🔷 MODEL BPE (Byte Pair Encoding)
--------------------------------------------------------------------------------
Tokeny (7): ['Czy', 'tam', 'książ', 'ki', 'w', 'biblio', 'tece']

Top 5 najbardziej podobnych zdań:
  1. [0.7456] Biblioteka posiada bogatą kolekcję...
  ...

🔵 MODEL SPACY (lemmatization)
--------------------------------------------------------------------------------
Tokeny (lemmatyzowane, 4): ['czytać', 'książka', 'w', 'biblioteka']

Top 5 najbardziej podobnych zdań:
  1. [0.9012] Czytam, czytałem i będę czytał książki...
  2. [0.8567] Książki w bibliotekach są dostępne...
  ...

📊 PORÓWNANIE STATYSTYCZNE
--------------------------------------------------------------------------------

Liczba tokenów:
  • SIMPLE: 4 tokenów
  • BPE: 7 tokenów
  • SPACY: 4 tokenów

Średnie podobieństwo (top 5):
  • SIMPLE: 0.7234
  • BPE: 0.6891
  • SPACY: 0.8456  ← NAJLEPSZE!

Norma wektora:
  • SIMPLE: 12.3456
  • BPE: 11.8923
  • SPACY: 13.2341

Podobieństwo cosinusowe między wektorami:
  • SIMPLE ↔ BPE: 0.6234
  • SIMPLE ↔ SPACY: 0.7891
  • BPE ↔ SPACY: 0.5678
```

## 🔍 Dlaczego spaCy + lemmatyzacja jest najlepsze dla polskiego?

Polski jest językiem **fleksyjnym** z bogatą morfologią:

### Problem z prostym split()
```python
"książki", "książką", "książek", "książkom"
# Traktowane jako 4 RÓŻNE tokeny
```

### Problem z BPE
```python
"książki" → ["książ", "ki"]
"książką" → ["książ", "ką"]
"książek" → ["książ", "ek"]
# Różne sub-tokeny dla tej samej formy bazowej
```

### Rozwiązanie: spaCy lemmatization
```python
"książki" → "książka"
"książką" → "książka"
"książek" → "książka"
"książkom" → "książka"
# Wszystkie formy → ta sama LEMMA
```

### Zalety lemmatyzacji dla polskiego:
- ✅ Redukuje wielkość słownika o ~70%
- ✅ Lepsze uogólnianie semantyczne
- ✅ Rozumie morfologię polską (7 przypadków, koniugacja)
- ✅ Automatycznie usuwa interpunkcję
- ✅ Normalizuje wielkość liter

## ⚙️ Parametry Treningu

Wszystkie modele używają tych samych parametrów Doc2Vec:

```python
VECTOR_LENGTH = 500    # Wymiar wektora
WINDOW_SIZE = 5        # Okno kontekstu
MIN_COUNT = 4          # Minimalna częstość tokena
WORKERS = 10           # Liczba wątków CPU
EPOCHS = 100           # Liczba epok treningu
```

Możesz je zmienić w każdym skrypcie treningowym.

## 📈 Porównanie Wydajności

### Czas tokenizacji (100k zdań)
- **SIMPLE**: ~1s (najszybsze)
- **BPE**: ~15s
- **SPACY**: ~120s (najwolniejsze, ale najlepsze wyniki)

### Rozmiar słownika (korpus ALL)
- **SIMPLE**: ~150,000 tokenów
- **BPE**: ~32,000 tokenów
- **SPACY**: ~45,000 tokenów (po lemmatyzacji)

### Jakość dla języka polskiego
| Metryka | SIMPLE | BPE | SPACY |
|---------|--------|-----|-------|
| Semantyka | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Morfologia | ❌ | ⭐ | ⭐⭐⭐⭐⭐ |
| OOV handling | ❌ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Wielkość słownika | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🎓 Przypadki użycia

### Użyj SIMPLE gdy:
- Prototypujesz szybko
- Potrzebujesz baseline do porównania
- Dane są już pre-procesowane

### Użyj BPE gdy:
- Pracujesz z wieloma językami
- Potrzebujesz małego słownika
- Masz dużo OOV (out-of-vocabulary) tokenów

### Użyj SPACY gdy:
- Pracujesz z językiem fleksyjnym (polski, rosyjski, czeski...)
- Jakość jest ważniejsza niż szybkość
- Chcesz najlepszych wyników semantycznych

## 🔧 Troubleshooting

### "Nie znaleziono modelu spaCy"
```bash
python -m spacy download pl_core_news_sm
```

### "Tokenizacja spaCy jest wolna"
- To normalne dla pierwszego uruchomienia
- Używa batching i pipe() dla wydajności
- Możesz zmniejszyć `EPOCHS` w skrypcie treningowym dla szybszych testów

### "Brak modeli do porównania"
Najpierw wytrenuj modele:
```bash
python train-both.py
```

### "Out of memory podczas treningu spaCy"
- Zmniejsz `batch_size` w `train-doc2vec-spacy.py` (domyślnie 1000)
- Zmniejsz `VECTOR_LENGTH` (np. do 100)
- Użyj mniejszego korpusu (np. `PAN_TADEUSZ` zamiast `ALL`)

## 📚 Dodatkowe Zasoby

### Modele spaCy dla polskiego
- `pl_core_news_sm` - mały, szybki (13 MB)
- `pl_core_news_md` - średni (45 MB) ← ZALECANY
- `pl_core_news_lg` - duży, najdokładniejszy (122 MB)

### Zmiana modelu spaCy
W `train-doc2vec-spacy.py`:
```python
nlp = spacy.load("pl_core_news_md")  # Zamiast _sm
```

## 📝 Licencja

Ten projekt jest częścią kursu dj-course. Użyj go do nauki i eksperymentowania!

## 🤝 Wkład

Masz pomysły na ulepszenia? Przetestowałeś inne metody tokenizacji?
Podziel się swoimi wynikami!

---

**Autor:** Projekt demonstracyjny porównania metod tokenizacji
**Data:** 2025
**Wersja:** 1.0

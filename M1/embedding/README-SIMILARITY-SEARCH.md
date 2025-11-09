# Wyszukiwanie Podobnych Zdań - Dokumentacja

## 📚 Przegląd

Ten katalog zawiera kompletny zestaw narzędzi do wyszukiwania semantycznie podobnych zdań w korpusie polskiego tekstu. Obsługuje dwa podejścia:

1. **Doc2Vec z tokenizacją spaCy** - własny model wytrenowany na korpusie
2. **Sentence-BERT** - gotowe modele transformer z HuggingFace

## 🚀 Szybki Start

### Wariant 1: Doc2Vec (spaCy)

```bash
# 1. Wytrenuj model (jeśli jeszcze nie masz)
python train-doc2vec-spacy.py

# 2. Zakoduj korpus (wygeneruj embeddingi)
python encode-corpus-doc2vec-spacy.py

# 3. Odpytaj korpus
python query-doc2vec-spacy.py
```

### Wariant 2: Sentence-BERT

```bash
# 1. Zakoduj korpus (używa gotowego modelu)
python encode-corpus-sbert.py

# 2. Odpytuj korpus
python query-sbert.py
```

## 📝 Szczegółowy Opis Skryptów

### Doc2Vec (spaCy)

#### `train-doc2vec-spacy.py`
- **Cel**: Trenowanie modelu Doc2Vec na polskim korpusie
- **Tokenizacja**: spaCy z lemmatyzacją (pl_core_news_sm)
- **Wyjście**:
  - `doc2vec_model_spacy.model` - wytrenowany model
  - `doc2vec_model_sentence_map_spacy.json` - mapa zdań
- **Parametry treningu**:
  - Vector size: 100
  - Window: 5
  - Min count: 4
  - Epochs: 100
  - Workers: 10

#### `encode-corpus-doc2vec-spacy.py`
- **Cel**: Generowanie embeddingów dla całego korpusu
- **Wejście**:
  - `doc2vec_model_spacy.model`
  - `doc2vec_model_sentence_map_spacy.json`
- **Wyjście**:
  - `doc2vec_spacy_corpus_embeddings.npy` - macierz embeddingów
- **Czas**: ~kilka sekund (embeddingi są już w modelu)

#### `query-doc2vec-spacy.py`
- **Cel**: Wyszukiwanie podobnych zdań
- **Funkcje**:
  - Test na zdaniach wymyślonych (spoza korpusu)
  - Test na zdaniach z korpusu
  - Tryb interaktywny
- **Wejście**:
  - `doc2vec_model_spacy.model`
  - `doc2vec_spacy_corpus_embeddings.npy`
  - `doc2vec_model_sentence_map_spacy.json`

### Sentence-BERT

#### `encode-corpus-sbert.py`
- **Cel**: Kodowanie korpusu przy użyciu gotowych modeli SBERT
- **Dostępne modele**:
  - `sdadas/stella-pl` ⭐ **REKOMENDOWANY dla polskiego**
    - NDCG@10: 60.52 na PIRB
    - Najlepszy model dla polskiego tekstu
  - `intfloat/multilingual-e5-small` - uniwersalny multilingual
  - `radlab/polish-sts-v2` - polski model STS
- **Wyjście**:
  - `sbert_<model_slug>_embeddings.npy` - macierz embeddingów
  - `sbert_sentence_map.json` - mapa zdań
- **Czas**: ~kilka minut (zależy od rozmiaru korpusu i modelu)

#### `query-sbert.py`
- **Cel**: Wyszukiwanie podobnych zdań używając SBERT
- **Funkcje**:
  - Test na zdaniach wymyślonych (spoza korpusu)
  - Test na zdaniach z korpusu
  - Tryb interaktywny
- **WAŻNE**: Użyj tego samego modelu co w `encode-corpus-sbert.py`!

## 🆚 Porównanie Podejść

| Aspekt | Doc2Vec (spaCy) | Sentence-BERT |
|--------|----------------|---------------|
| **Trening** | Wymagany (~10-20 min) | Gotowy model |
| **Kodowanie korpusu** | Bardzo szybkie (~sek) | Wolniejsze (~min) |
| **Jakość dla polskiego** | Dobra (zależy od korpusu) | ⭐ Doskonała (stella-pl) |
| **Wymiar wektora** | 100 | 1024 (stella-pl) |
| **Elastyczność** | Można dostosować parametry | Gotowy model |
| **Rozmiar modelu** | Mały (~10-50 MB) | Duży (~300-1500 MB) |

## 📊 Przykładowe Użycie

### Przykład 1: Wyszukiwanie podobnych zdań (Doc2Vec)

```python
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import spacy

# Wczytaj model i embeddingi
model = Doc2Vec.load("doc2vec_model_spacy.model")
corpus_embeddings = np.load("doc2vec_spacy_corpus_embeddings.npy")

# Wczytaj spaCy
nlp = spacy.load("pl_core_news_sm")

# Tokenizuj zapytanie
query = "Jestem głodny."
tokens = [token.lemma_.lower() for token in nlp(query)
          if not token.is_punct and not token.is_space]

# Wygeneruj embedding
query_embedding = model.infer_vector(tokens, epochs=model.epochs)

# Znajdź podobne (używając cosine similarity)
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
top_5 = np.argsort(similarities)[::-1][:5]
```

### Przykład 2: Wyszukiwanie podobnych zdań (SBERT)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Wczytaj model i embeddingi
model = SentenceTransformer('sdadas/stella-pl')
corpus_embeddings = np.load("sbert_sdadas_stella_pl_embeddings.npy")

# Wygeneruj embedding dla zapytania
query = "Jestem głodny."
query_embedding = model.encode([query])

# Znajdź podobne
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
top_5 = np.argsort(similarities)[::-1][:5]
```

## 🔧 Konfiguracja

### Zmiana modelu SBERT

W plikach `encode-corpus-sbert.py` i `query-sbert.py`, zmień:

```python
MODEL_NAME = 'sdadas/stella-pl'  # Zmień na inny model
```

Dostępne opcje:
- `sdadas/stella-pl` - ⭐ najlepszy dla polskiego
- `radlab/polish-sts-v2` - polski model STS
- `intfloat/multilingual-e5-small` - uniwersalny multilingual

**UWAGA**: Zawsze używaj tego samego modelu w `encode-corpus-sbert.py` i `query-sbert.py`!

### Zmiana korpusu

W plikach `train-doc2vec-spacy.py` i `encode-corpus-sbert.py`, zmień:

```python
files = CORPORA_FILES["ALL"]  # Zmień na inny korpus
# files = CORPORA_FILES["PAN_TADEUSZ"]
# files = CORPORA_FILES["NKJP"]
```

## 📈 Wyniki i Testy

Każdy skrypt `query-*.py` zawiera automatyczne testy:

1. **Test 1: Zdania wymyślone (spoza korpusu)**
   - Sprawdza, jak model radzi sobie z nowymi zdaniami
   - 5 przykładowych zapytań

2. **Test 2: Zdania z korpusu treningowego**
   - Sprawdza, czy model poprawnie odtwarza podobieństwa
   - 3 losowe zdania z korpusu

3. **Tryb interaktywny**
   - Możliwość wpisywania własnych zapytań
   - Wpisz `quit` lub `exit` aby zakończyć

## 🎯 Najlepsze Praktyki

1. **Dla produkcji**: Użyj SBERT z modelem `sdadas/stella-pl`
   - Najlepsza jakość dla polskiego
   - Nie wymaga treningu
   - Rozmiar: ~1.5 GB

2. **Dla eksperymentów**: Użyj Doc2Vec
   - Szybkie trenowanie i wnioskowanie
   - Możliwość dostosowania parametrów
   - Mały rozmiar

3. **Zawsze używaj tego samego modelu** do kodowania i odpytywania!

4. **Cache embeddingów**: Raz zakodowany korpus można wielokrotnie odpytywać

## 📚 Zasoby

- [HuggingFace Polish Embedding Models](https://huggingface.co/collections/sdadas/polish-embedding-models-66e69fe67240b605c9348ea7)
- [PIRB Leaderboard](https://github.com/sdadas/pirb) - Polish Information Retrieval Benchmark
- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Gensim Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html)

## ❓ FAQ

**Q: Który model wybrać?**
A: Dla najlepszej jakości na polskim tekście, użyj `sdadas/stella-pl`.

**Q: Czy mogę użyć Doc2Vec modelu w SBERT skrypcie?**
A: Nie bezpośrednio. To są różne architektury. Użyj dedykowanych skryptów.

**Q: Ile czasu zajmuje kodowanie korpusu?**
A: Doc2Vec: ~kilka sekund. SBERT: ~kilka minut (zależy od rozmiaru korpusu).

**Q: Jak duży jest plik embeddingów?**
A: Dla ~100k zdań:
- Doc2Vec (dim=100): ~40 MB
- SBERT stella-pl (dim=1024): ~400 MB

**Q: Czy potrzebuję GPU?**
A: Nie jest wymagane, ale przyspiesza kodowanie SBERT (~2-5x).

## 🐛 Rozwiązywanie Problemów

### Błąd: "No module named 'spacy'"
```bash
pip install spacy
python -m spacy download pl_core_news_sm
```

### Błąd: "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Błąd: "File not found: doc2vec_model_spacy.model"
Najpierw wytrenuj model:
```bash
python train-doc2vec-spacy.py
```

### Błąd: "File not found: sbert_*_embeddings.npy"
Najpierw zakoduj korpus:
```bash
python encode-corpus-sbert.py

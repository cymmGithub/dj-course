# Porównanie Tokenizacji: BPE vs spaCy + Lematyzacja

## Diagram Porównawczy

```mermaid
graph TB
    subgraph "TEKST WEJŚCIOWY"
        INPUT["'Książki leżały na półkach. Czytałem tę książkę wczoraj.'"]
    end

    subgraph "BPE - Byte Pair Encoding"
        BPE_START["📥 KROK 1: Wczytanie tekstu"]
        BPE_TRAIN["🔧 KROK 2: Trenowanie słownika BPE<br/>Zliczanie par znaków w korpusie"]
        BPE_MERGE["🔗 KROK 3: Łączenie najczęstszych par<br/>np. 'k'+'s' → 'ks', 'książ'+'ka' → 'książka'"]
        BPE_SPLIT["✂️ KROK 4: Podział na subwords<br/>według nauczonego słownika"]
        BPE_OUTPUT["📤 WYNIK BPE:<br/>['Ksi', 'ąż', 'ki', 'le', 'ża', 'ły', 'na',<br/>'pół', 'kach', 'Czy', 'ta', 'łem',<br/>'tę', 'ksi', 'ąż', 'kę', 'wczo', 'raj']"]
        BPE_PROBLEM["⚠️ PROBLEM:<br/>Różne formy tego samego słowa<br/>są tokenizowane różnie:<br/>'książki' ≠ 'książkę' ≠ 'książką'"]

        BPE_START --> BPE_TRAIN
        BPE_TRAIN --> BPE_MERGE
        BPE_MERGE --> BPE_SPLIT
        BPE_SPLIT --> BPE_OUTPUT
        BPE_OUTPUT --> BPE_PROBLEM
    end

    subgraph "spaCy + Lematyzacja"
        SPACY_START["📥 KROK 1: Wczytanie tekstu"]
        SPACY_LOAD["🧠 KROK 2: Załadowanie modelu NLP<br/>pl_core_news_sm (trenowany na polskim)"]
        SPACY_PARSE["🔍 KROK 3: Analiza morfologiczna<br/>Rozpoznanie: część mowy, przypadek, liczba, osoba"]
        SPACY_LEMMA["📖 KROK 4: Lematyzacja<br/>Sprowadzenie do formy podstawowej:<br/>książki → książka<br/>książkę → książka<br/>leżały → leżeć"]
        SPACY_FILTER["🧹 KROK 5: Filtrowanie<br/>Usunięcie: interpunkcja, spacje"]
        SPACY_OUTPUT["📤 WYNIK spaCy:<br/>['książka', 'leżeć', 'półka',<br/>'czytać', 'ten', 'książka', 'wczoraj']"]
        SPACY_BENEFIT["✅ KORZYŚĆ:<br/>Wszystkie formy → jedna lemma<br/>'książki', 'książkę', 'książką' → 'książka'<br/>Model rozumie 7 przypadków polskich"]

        SPACY_START --> SPACY_LOAD
        SPACY_LOAD --> SPACY_PARSE
        SPACY_PARSE --> SPACY_LEMMA
        SPACY_LEMMA --> SPACY_FILTER
        SPACY_FILTER --> SPACY_OUTPUT
        SPACY_OUTPUT --> SPACY_BENEFIT
    end

    INPUT --> BPE_START
    INPUT --> SPACY_START

    subgraph "PORÓWNANIE WYNIKÓW"
        COMP_BPE["BPE: 18 tokenów<br/>książki ≠ książkę (różne tokeny)"]
        COMP_SPACY["spaCy: 7 tokenów<br/>książki = książkę = książka (ta sama lemma)"]
        COMP_WINNER["🏆 Dla języka polskiego:<br/>spaCy + lematyzacja WYGRYWA<br/>bo redukuje 14+ form słowa do jednej"]

        COMP_BPE --> COMP_WINNER
        COMP_SPACY --> COMP_WINNER
    end

    BPE_PROBLEM --> COMP_BPE
    SPACY_BENEFIT --> COMP_SPACY

    style BPE_PROBLEM fill:#ffcccc
    style SPACY_BENEFIT fill:#ccffcc
    style COMP_WINNER fill:#ffffcc
    style INPUT fill:#e1f5ff
```

## Szczegółowe Wyjaśnienie

### BPE (Byte Pair Encoding)

**Zalety:**
- ✅ Uniwersalny - działa dla każdego języka
- ✅ Szybki trening i wykonanie
- ✅ Radzi sobie z rzadkimi słowami przez subwords

**Wady dla języka polskiego:**
- ❌ Nie rozumie gramatyki
- ❌ Każda forma fleksyjna → inne tokeny
- ❌ "książka" (mianownik) ≠ "książki" (dopełniacz) ≠ "książką" (narzędnik)
- ❌ Dla języka fleksyjnego (7 przypadków) = ogromny słownik

**Przykład:**
```
Tekst:      "Mam książkę. Czytam książkę. To jest książka."
BPE tokens: ['Ma', 'm', 'ksi', 'ąż', 'kę', 'Czy', 'ta', 'm', 'ksi', 'ąż', 'kę', 'To', 'jest', 'ksi', 'ąż', 'ka']
Problem:    'książkę' i 'książka' → różne tokeny!
```

### spaCy + Lematyzacja

**Zalety:**
- ✅ Rozumie morfologię polskiego
- ✅ Wszystkie 14+ form słowa → jedna lemma
- ✅ Lepsze embeddingi (podobne znaczenie → podobne wektory)
- ✅ Mniejszy słownik (książka, książki, książkę → książka)

**Wady:**
- ❌ Wymaga zainstalowania modelu językowego (pl_core_news_sm)
- ❌ Wolniejszy (ale cache rozwiązuje problem!)

**Przykład:**
```
Tekst:         "Mam książkę. Czytam książkę. To jest książka."
spaCy tokens:  ['mieć', 'książka', 'czytać', 'książka', 'to', 'być', 'książka']
Korzyść:       wszystkie formy 'książka' → jedna lemma 'książka'!
```

## Dlaczego spaCy Wygrywa dla Języka Polskiego?

Polski to **język fleksyjny** z 7 przypadkami gramatycznymi:

| Przypadek | Liczba pojedyncza | Liczba mnoga |
|-----------|-------------------|--------------|
| Mianownik | książka           | książki      |
| Dopełniacz| książki           | książek      |
| Celownik  | książce           | książkom     |
| Biernik   | książkę           | książki      |
| Narzędnik | książką           | książkami    |
| Miejscownik| książce          | książkach    |
| Wołacz    | książko           | książki      |

**Dla BPE:** 14 różnych form = 14+ różnych zestawów tokenów
**Dla spaCy:** 14 różnych form = 1 lemma (`książka`)

## Wydajność

### BPE
- ⚡ Szybki: ~15 sekund na korpus
- 🗂️ Duży słownik przez formy fleksyjne

### spaCy + Cache
- 🐌 Bez cache: ~120 sekund
- ⚡ Z cache: ~2 sekundy (60x szybciej!)
- 🗂️ Mały słownik przez lematyzację

## Podsumowanie

| Kryterium | BPE | spaCy + Lematyzacja |
|-----------|-----|---------------------|
| Rozumienie polskiego | ❌ Nie | ✅ Tak |
| Formy fleksyjne | ❌ Różne tokeny | ✅ Jedna lemma |
| Szybkość (z cache) | ⚡⚡⚡ | ⚡⚡⚡ |
| Jakość embeddingów | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Rekomendacja dla polskiego** | ❌ | ✅ **WYBIERZ TO!** |

## Pliki w Projekcie

- `train-doc2vec-bpe.py` - Trenowanie z BPE
- `train-doc2vec-spacy.py` - Trenowanie z spaCy + cache
- `compare-all-tokenization.py` - Interaktywne porównanie
- `visualize-doc2vec-spacy.py` - Wizualizacja wyników spaCy

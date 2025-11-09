#!/usr/bin/env python3
"""
Interaktywny skrypt do wizualizacji wyników modelu Doc2Vec z tokenizacją spaCy.
Pozwala na:
- Wprowadzanie własnych zdań i znajdowanie podobnych
- Przeglądanie losowych przykładów z korpusu
- Wizualizację embeddingów (t-SNE, PCA)
- Analizę statystyk modelu
"""

import numpy as np
import json
import logging
from gensim.models.doc2vec import Doc2Vec
import spacy
import sys
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Wyłącz verbose logging
logging.basicConfig(level=logging.ERROR)

# Kolory ANSI dla terminala
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text, width=80):
    """Wyświetla nagłówek z kolorami."""
    print("\n" + Colors.BOLD + Colors.CYAN + "="*width + Colors.ENDC)
    print(Colors.BOLD + Colors.CYAN + text.center(width) + Colors.ENDC)
    print(Colors.BOLD + Colors.CYAN + "="*width + Colors.ENDC)

def print_subheader(text):
    """Wyświetla podtytuł."""
    print("\n" + Colors.BOLD + Colors.BLUE + text + Colors.ENDC)
    print(Colors.BLUE + "-"*80 + Colors.ENDC)

def print_error(text):
    """Wyświetla komunikat błędu."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_success(text):
    """Wyświetla komunikat sukcesu."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_info(text):
    """Wyświetla informację."""
    print(f"{Colors.YELLOW}ℹ {text}{Colors.ENDC}")

# Pliki modelu
MODEL_FILE = "doc2vec_model_spacy.model"
SENTENCE_MAP_FILE = "doc2vec_model_sentence_map_spacy.json"

print_header("WIZUALIZACJA DOC2VEC - MODEL SPACY")

# --- Wczytanie modelu i danych ---
print("\nWczytywanie modelu i danych...")

try:
    model = Doc2Vec.load(MODEL_FILE)
    print_success(f"Model załadowany: {MODEL_FILE}")
except FileNotFoundError:
    print_error(f"Nie znaleziono modelu: {MODEL_FILE}")
    print_info("Uruchom najpierw: python train-doc2vec-spacy.py")
    sys.exit(1)

try:
    with open(SENTENCE_MAP_FILE, "r", encoding="utf-8") as f:
        sentence_map = json.load(f)
    print_success(f"Mapa zdań załadowana: {len(sentence_map):,} zdań")
except FileNotFoundError:
    print_error(f"Nie znaleziono mapy zdań: {SENTENCE_MAP_FILE}")
    sys.exit(1)

# Wczytanie modelu spaCy
try:
    nlp = spacy.load("pl_core_news_sm")
    print_success("Model spaCy załadowany: pl_core_news_sm")
except OSError:
    print_error("Nie znaleziono modelu spaCy")
    print_info("Zainstaluj: python -m spacy download pl_core_news_sm")
    sys.exit(1)

# --- Funkcje pomocnicze ---

def tokenize_spacy(text):
    """Tokenizacja z lemmatyzacją (tak samo jak w treningu)."""
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_space and token.text.strip()
    ]
    return tokens

def find_similar(query_text, topn=10):
    """Znajduje podobne zdania do zapytania."""
    # Tokenizacja
    tokens = tokenize_spacy(query_text)

    if not tokens:
        print_error("Brak tokenów po lemmatyzacji")
        return []

    # Wnioskowanie wektora
    vector = model.infer_vector(tokens, epochs=model.epochs)

    # Znajdź podobne
    similar = model.dv.most_similar([vector], topn=topn)

    return tokens, vector, similar

def display_similar_results(query_text, tokens, similar):
    """Wyświetla wyniki wyszukiwania podobnych zdań."""
    print_subheader(f'WYNIKI DLA: "{query_text}"')

    print(f"\n{Colors.BOLD}Tokeny (lemmatyzowane):{Colors.ENDC}")
    print(f"  {tokens}")
    print(f"  Liczba tokenów: {len(tokens)}")

    print(f"\n{Colors.BOLD}Top {len(similar)} najbardziej podobnych zdań:{Colors.ENDC}")
    for rank, (doc_id, similarity) in enumerate(similar, 1):
        sentence = sentence_map[int(doc_id)]
        # Podświetl wysokie podobieństwa
        if similarity > 0.8:
            color = Colors.GREEN
        elif similarity > 0.6:
            color = Colors.YELLOW
        else:
            color = ""

        print(f"  {color}{rank:2d}. [{similarity:.4f}] {sentence[:100]}{Colors.ENDC}")
        if len(sentence) > 100:
            print(f"      {sentence[100:200]}...")

# --- Funkcje menu ---

def interactive_search():
    """Tryb interaktywnego wyszukiwania."""
    print_header("TRYB INTERAKTYWNY - Wyszukiwanie Podobnych Zdań")
    print(f"\n{Colors.BOLD}Wprowadź własne zdania aby znaleźć podobne.{Colors.ENDC}")
    print(f"{Colors.YELLOW}Wpisz 'q' lub 'quit' aby zakończyć.{Colors.ENDC}\n")

    while True:
        try:
            user_input = input(f"{Colors.BOLD}{Colors.GREEN}Zdanie > {Colors.ENDC}").strip()

            if user_input.lower() in ['q', 'quit', 'exit']:
                print(f"\n{Colors.YELLOW}Zamykam tryb interaktywny.{Colors.ENDC}")
                break

            if not user_input:
                print_info("Zdanie nie może być puste.")
                continue

            # Znajdź podobne
            tokens, vector, similar = find_similar(user_input, topn=10)

            # Wyświetl wyniki
            display_similar_results(user_input, tokens, similar)

            # Statystyki wektora
            print(f"\n{Colors.BOLD}Statystyki wektora:{Colors.ENDC}")
            print(f"  Norma L2: {np.linalg.norm(vector):.4f}")
            print(f"  Średnia wartość: {np.mean(vector):.4f}")
            print(f"  Odchylenie std: {np.std(vector):.4f}")
            print()

        except EOFError:
            print(f"\n{Colors.YELLOW}Zakończono.{Colors.ENDC}")
            break
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Przerwano przez użytkownika.{Colors.ENDC}")
            break

def random_examples():
    """Pokazuje losowe przykłady z korpusu."""
    print_header("LOSOWE PRZYKŁADY Z KORPUSU")

    num_examples = 5
    print(f"\nWylosuję {num_examples} zdań i znajdę dla nich najbardziej podobne...\n")

    indices = random.sample(range(len(sentence_map)), num_examples)

    for i, idx in enumerate(indices, 1):
        sentence = sentence_map[idx]
        print(f"\n{Colors.BOLD}{Colors.YELLOW}═══ Przykład {i}/{num_examples} ═══{Colors.ENDC}")
        print(f"{Colors.BOLD}Zdanie źródłowe:{Colors.ENDC}")
        print(f"  [{idx}] {sentence}")

        # Pobierz wektor dla tego dokumentu
        vector = model.dv[str(idx)]

        # Znajdź podobne (pomijając siebie)
        similar = model.dv.most_similar([vector], topn=6)
        similar = [(doc_id, sim) for doc_id, sim in similar if int(doc_id) != idx][:5]

        print(f"\n{Colors.BOLD}Top 5 podobnych:{Colors.ENDC}")
        for rank, (doc_id, similarity) in enumerate(similar, 1):
            sent = sentence_map[int(doc_id)]
            print(f"  {rank}. [{similarity:.4f}] {sent[:80]}")

        if i < num_examples:
            input(f"\n{Colors.CYAN}Naciśnij Enter aby kontynuować...{Colors.ENDC}")

def visualize_embeddings():
    """Wizualizacja embeddingów używając t-SNE lub PCA."""
    print_header("WIZUALIZACJA EMBEDDINGÓW")

    print(f"\n{Colors.BOLD}Wybierz metodę wizualizacji:{Colors.ENDC}")
    print(f"  {Colors.CYAN}1.{Colors.ENDC} t-SNE (wolniejsze, lepsze dla struktur nieliniowych)")
    print(f"  {Colors.CYAN}2.{Colors.ENDC} PCA (szybsze, linearne)")
    print(f"  {Colors.CYAN}q.{Colors.ENDC} Powrót do menu")

    choice = input(f"\n{Colors.BOLD}{Colors.GREEN}Wybór > {Colors.ENDC}").strip()

    if choice == 'q':
        return

    # Parametry wizualizacji
    num_samples = min(1000, len(sentence_map))  # Max 1000 punktów dla czytelności
    print(f"\n{Colors.YELLOW}Próbuję {num_samples} losowych dokumentów...{Colors.ENDC}")

    # Wylosuj indeksy
    indices = random.sample(range(len(sentence_map)), num_samples)

    # Pobierz wektory
    vectors = np.array([model.dv[str(i)] for i in indices])

    print(f"Rozmiar macierzy wektorów: {vectors.shape}")

    # Redukuj wymiarowość
    if choice == '1':
        print(f"{Colors.YELLOW}Obliczam t-SNE (może potrwać ~30s)...{Colors.ENDC}")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        print(f"{Colors.YELLOW}Obliczam PCA...{Colors.ENDC}")
        reducer = PCA(n_components=2, random_state=42)

    coords_2d = reducer.fit_transform(vectors)

    print_success("Redukcja wymiarowości zakończona")

    # Wizualizacja
    print(f"{Colors.YELLOW}Tworzę wykres...{Colors.ENDC}")

    plt.figure(figsize=(12, 8))
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.5, s=10)

    method_name = "t-SNE" if choice == '1' else "PCA"
    plt.title(f"Wizualizacja {num_samples} embeddingów Doc2Vec (spaCy) - {method_name}", fontsize=14, fontweight='bold')
    plt.xlabel(f"{method_name} Dimension 1")
    plt.ylabel(f"{method_name} Dimension 2")
    plt.grid(True, alpha=0.3)

    # Opcjonalnie: zaznacz kilka losowych punktów z etykietami
    num_labels = min(20, num_samples // 50)
    labeled_indices = random.sample(range(num_samples), num_labels)

    for i in labeled_indices:
        idx = indices[i]
        sentence = sentence_map[idx][:30] + "..."
        plt.annotate(sentence,
                    xy=(coords_2d[i, 0], coords_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.tight_layout()

    # Zapisz i pokaż
    output_file = f"doc2vec_spacy_visualization_{method_name.lower()}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print_success(f"Wykres zapisany: {output_file}")

    print(f"\n{Colors.YELLOW}Czy otworzyć wykres? (t/n): {Colors.ENDC}", end="")
    show = input().strip().lower()

    if show in ['t', 'y', 'yes', 'tak']:
        plt.show()
    else:
        plt.close()

def show_statistics():
    """Wyświetla statystyki modelu."""
    print_header("STATYSTYKI MODELU")

    print(f"\n{Colors.BOLD}{Colors.BLUE}📊 Parametry modelu:{Colors.ENDC}")
    print(f"  ├─ Rozmiar słownika: {len(model.wv):,} unikalnych tokenów")
    print(f"  ├─ Wymiar wektora: {model.vector_size}")
    print(f"  ├─ Liczba dokumentów: {len(model.dv):,}")
    print(f"  ├─ Liczba epok treningu: {model.epochs}")
    print(f"  ├─ Rozmiar okna: {model.window}")
    print(f"  ├─ Minimalna liczność: {model.min_count}")
    print(f"  └─ Algorytm: PV-DBOW (dm=0)")

    # Analiza wektorów
    print(f"\n{Colors.BOLD}{Colors.BLUE}📈 Analiza wektorów dokumentów:{Colors.ENDC}")

    # Pobierz próbkę wektorów
    sample_size = min(1000, len(model.dv))
    sample_indices = random.sample(range(len(model.dv)), sample_size)
    sample_vectors = np.array([model.dv[str(i)] for i in sample_indices])

    norms = np.linalg.norm(sample_vectors, axis=1)
    means = np.mean(sample_vectors, axis=1)
    stds = np.std(sample_vectors, axis=1)

    print(f"  (Analiza na próbce {sample_size} wektorów)")
    print(f"  ├─ Średnia norma L2: {np.mean(norms):.4f} ± {np.std(norms):.4f}")
    print(f"  ├─ Min/Max norma: {np.min(norms):.4f} / {np.max(norms):.4f}")
    print(f"  ├─ Średnia wartość: {np.mean(means):.6f}")
    print(f"  └─ Średnie odchylenie std: {np.mean(stds):.4f}")

    # Top słowa w słowniku
    print(f"\n{Colors.BOLD}{Colors.BLUE}📝 Statystyki słownika:{Colors.ENDC}")

    # Sortuj słowa według częstości (frequency)
    vocab_items = [(word, model.wv.get_vecattr(word, 'count'))
                   for word in list(model.wv.index_to_key)[:100]]
    vocab_items.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Top 20 najczęstszych tokenów:")
    for i, (word, count) in enumerate(vocab_items[:20], 1):
        print(f"    {i:2d}. '{word}' ({count:,} wystąpień)")

    # Długości zdań
    print(f"\n{Colors.BOLD}{Colors.BLUE}📏 Statystyki korpusu:{Colors.ENDC}")
    sentence_lengths = [len(sent.split()) for sent in sentence_map]
    print(f"  ├─ Liczba zdań: {len(sentence_map):,}")
    print(f"  ├─ Średnia długość zdania: {np.mean(sentence_lengths):.1f} słów")
    print(f"  ├─ Min/Max długość: {np.min(sentence_lengths)} / {np.max(sentence_lengths)} słów")
    print(f"  └─ Mediana długości: {np.median(sentence_lengths):.1f} słów")

    input(f"\n{Colors.CYAN}Naciśnij Enter aby wrócić do menu...{Colors.ENDC}")

def semantic_search_demo():
    """Demonstracja wyszukiwania semantycznego z przykładami."""
    print_header("DEMONSTRACJA WYSZUKIWANIA SEMANTYCZNEGO")

    demo_queries = [
        "Król przybył do miasta",
        "Wojna między narodami",
        "Piękna dziewczyna tańczy",
        "Jestem bardzo głodny",
        "Pogoda jest wspaniała"
    ]

    print(f"\n{Colors.BOLD}Przetestuję {len(demo_queries)} przykładowych zapytań:{Colors.ENDC}\n")

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{Colors.BOLD}{Colors.YELLOW}═══ Query {i}/{len(demo_queries)} ═══{Colors.ENDC}")

        tokens, vector, similar = find_similar(query, topn=5)
        display_similar_results(query, tokens, similar[:5])

        if i < len(demo_queries):
            input(f"\n{Colors.CYAN}Naciśnij Enter aby kontynuować...{Colors.ENDC}")

# --- Menu główne ---

def main_menu():
    """Wyświetla menu główne i obsługuje wybór użytkownika."""
    while True:
        print_header("MENU GŁÓWNE")
        print(f"\n{Colors.BOLD}Wybierz opcję:{Colors.ENDC}")
        print(f"  {Colors.CYAN}1.{Colors.ENDC} Interaktywne wyszukiwanie (wprowadź własne zdania)")
        print(f"  {Colors.CYAN}2.{Colors.ENDC} Losowe przykłady z korpusu")
        print(f"  {Colors.CYAN}3.{Colors.ENDC} Demonstracja wyszukiwania semantycznego")
        print(f"  {Colors.CYAN}4.{Colors.ENDC} Wizualizacja embeddingów (t-SNE/PCA)")
        print(f"  {Colors.CYAN}5.{Colors.ENDC} Statystyki modelu")
        print(f"  {Colors.CYAN}q.{Colors.ENDC} Zakończ")

        try:
            choice = input(f"\n{Colors.BOLD}{Colors.GREEN}Wybór > {Colors.ENDC}").strip().lower()

            if choice == '1':
                interactive_search()
            elif choice == '2':
                random_examples()
            elif choice == '3':
                semantic_search_demo()
            elif choice == '4':
                visualize_embeddings()
            elif choice == '5':
                show_statistics()
            elif choice in ['q', 'quit', 'exit']:
                print(f"\n{Colors.GREEN}Do widzenia!{Colors.ENDC}\n")
                break
            else:
                print_error("Nieprawidłowy wybór. Spróbuj ponownie.")

        except EOFError:
            print(f"\n{Colors.YELLOW}Zakończono.{Colors.ENDC}")
            break
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Przerwano przez użytkownika.{Colors.ENDC}")
            break

# --- Uruchomienie programu ---

if __name__ == "__main__":
    try:
        main_menu()
    except Exception as e:
        print_error(f"Nieoczekiwany błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

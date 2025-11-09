#!/usr/bin/env python3
"""
Interaktywny skrypt do porównywania WSZYSTKICH trzech metod tokenizacji:
- SIMPLE (split)
- BPE (Byte Pair Encoding)
- SPACY (lemmatization)

Pozwala na wprowadzanie własnych zdań i porównywanie wyników.
"""

import numpy as np
import json
import logging
from gensim.models.doc2vec import Doc2Vec
from tokenizers import Tokenizer
import spacy
import sys

# Ustawienie logowania dla gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

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

def print_subheader(text, width=80):
    """Wyświetla podtytuł."""
    print("\n" + Colors.BOLD + Colors.BLUE + text + Colors.ENDC)
    print(Colors.BLUE + "-"*width + Colors.ENDC)

def print_error(text):
    """Wyświetla komunikat błędu."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_success(text):
    """Wyświetla komunikat sukcesu."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_info(text):
    """Wyświetla informację."""
    print(f"{Colors.YELLOW}ℹ {text}{Colors.ENDC}")

# Pliki dla modelu SIMPLE
MODEL_FILE_SIMPLE = "doc2vec_model_simple.model"
SENTENCE_MAP_FILE_SIMPLE = "doc2vec_model_sentence_map_simple.json"

# Pliki dla modelu BPE
TOKENIZER_FILE_BPE = "../tokenizer/tokenizers/nkjp-tokenizer.json"
MODEL_FILE_BPE = "doc2vec_model_bpe.model"
SENTENCE_MAP_FILE_BPE = "doc2vec_model_sentence_map_bpe.json"

# Pliki dla modelu SPACY
MODEL_FILE_SPACY = "doc2vec_model_spacy.model"
SENTENCE_MAP_FILE_SPACY = "doc2vec_model_sentence_map_spacy.json"

print_header("INTERAKTYWNE PORÓWNANIE TOKENIZACJI")
print(f"\n{Colors.BOLD}Wczytywanie modeli i tokenizatorów...{Colors.ENDC}\n")

# --- Wczytanie wszystkich modeli ---

models = {}
sentence_maps = {}
tokenizers = {}

# 1. MODEL SIMPLE
try:
    models['simple'] = Doc2Vec.load(MODEL_FILE_SIMPLE)
    with open(SENTENCE_MAP_FILE_SIMPLE, "r", encoding="utf-8") as f:
        sentence_maps['simple'] = json.load(f)
    print_success(f"Model SIMPLE załadowany ({MODEL_FILE_SIMPLE})")
except FileNotFoundError:
    print_error(f"Nie znaleziono modelu SIMPLE '{MODEL_FILE_SIMPLE}'")
    print_info("Uruchom najpierw: python train-doc2vec.py")
    models['simple'] = None

# 2. MODEL BPE
try:
    tokenizers['bpe'] = Tokenizer.from_file(TOKENIZER_FILE_BPE)
    models['bpe'] = Doc2Vec.load(MODEL_FILE_BPE)
    with open(SENTENCE_MAP_FILE_BPE, "r", encoding="utf-8") as f:
        sentence_maps['bpe'] = json.load(f)
    print_success(f"Model BPE załadowany ({MODEL_FILE_BPE})")
except FileNotFoundError as e:
    print_error(f"Nie znaleziono modelu BPE lub tokenizera")
    print_info("Uruchom najpierw: python train-doc2vec-bpe.py")
    models['bpe'] = None

# 3. MODEL SPACY
try:
    nlp = spacy.load("pl_core_news_sm")
    models['spacy'] = Doc2Vec.load(MODEL_FILE_SPACY)
    with open(SENTENCE_MAP_FILE_SPACY, "r", encoding="utf-8") as f:
        sentence_maps['spacy'] = json.load(f)
    print_success(f"Model SPACY załadowany ({MODEL_FILE_SPACY})")
except (FileNotFoundError, OSError) as e:
    print_error(f"Nie znaleziono modelu SPACY lub modelu językowego spaCy")
    print_info("Uruchom najpierw:")
    print_info("  1. python -m spacy download pl_core_news_sm")
    print_info("  2. python train-doc2vec-spacy.py")
    models['spacy'] = None

# Sprawdź czy wszystkie modele zostały załadowane
available_models = [name for name, model in models.items() if model is not None]

if not available_models:
    print_error("\nBrak dostępnych modeli. Nie można kontynuować.")
    sys.exit(1)

print(f"\n{Colors.GREEN}{Colors.BOLD}Dostępne modele: {', '.join(available_models).upper()}{Colors.ENDC}")

# --- Funkcje tokenizacji ---

def tokenize_simple(text):
    """Tokenizacja prostym split()"""
    return text.split()

def tokenize_bpe(text):
    """Tokenizacja BPE"""
    return tokenizers['bpe'].encode(text).tokens

def tokenize_spacy(text):
    """Tokenizacja spaCy z lemmatyzacją"""
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_space and token.text.strip()
    ]

# --- Funkcja głównego porównania ---

def compare_sentence(sentence, topn=5):
    """
    Porównuje tokenizację i wyniki dla danego zdania we wszystkich dostępnych modelach.

    Args:
        sentence: Zdanie do przetestowania
        topn: Liczba najbardziej podobnych zdań do wyświetlenia
    """
    print_header(f'ANALIZA: "{sentence}"')

    # Przechowywanie wyników
    results = {}

    # --- SIMPLE ---
    if models['simple']:
        print_subheader("🔶 MODEL SIMPLE (split tokenization)")
        tokens = tokenize_simple(sentence)
        print(f"Tokeny ({len(tokens)}): {tokens}")

        vector = models['simple'].infer_vector(tokens, epochs=models['simple'].epochs)
        similar = models['simple'].dv.most_similar([vector], topn=topn)

        print(f"\nTop {topn} najbardziej podobnych zdań:")
        for rank, (doc_id, sim) in enumerate(similar, 1):
            sent = sentence_maps['simple'][int(doc_id)]
            print(f"  {rank}. [{sim:.4f}] {sent[:70]}")

        results['simple'] = {'tokens': tokens, 'vector': vector, 'similar': similar}

    # --- BPE ---
    if models['bpe']:
        print_subheader("🔷 MODEL BPE (Byte Pair Encoding)")
        tokens = tokenize_bpe(sentence)
        print(f"Tokeny ({len(tokens)}): {tokens}")

        vector = models['bpe'].infer_vector(tokens, epochs=models['bpe'].epochs)
        similar = models['bpe'].dv.most_similar([vector], topn=topn)

        print(f"\nTop {topn} najbardziej podobnych zdań:")
        for rank, (doc_id, sim) in enumerate(similar, 1):
            sent = sentence_maps['bpe'][int(doc_id)]
            print(f"  {rank}. [{sim:.4f}] {sent[:70]}")

        results['bpe'] = {'tokens': tokens, 'vector': vector, 'similar': similar}

    # --- SPACY ---
    if models['spacy']:
        print_subheader("🔵 MODEL SPACY (lemmatization)")
        tokens = tokenize_spacy(sentence)
        print(f"Tokeny (lemmatyzowane, {len(tokens)}): {tokens}")

        vector = models['spacy'].infer_vector(tokens, epochs=models['spacy'].epochs)
        similar = models['spacy'].dv.most_similar([vector], topn=topn)

        print(f"\nTop {topn} najbardziej podobnych zdań:")
        for rank, (doc_id, sim) in enumerate(similar, 1):
            sent = sentence_maps['spacy'][int(doc_id)]
            print(f"  {rank}. [{sim:.4f}] {sent[:70]}")

        results['spacy'] = {'tokens': tokens, 'vector': vector, 'similar': similar}

    # --- Porównanie statystyczne ---
    if len(results) > 1:
        print_subheader("📊 PORÓWNANIE STATYSTYCZNE")

        print(f"\n{Colors.BOLD}Liczba tokenów:{Colors.ENDC}")
        for model_name, data in results.items():
            print(f"  • {model_name.upper()}: {len(data['tokens'])} tokenów")

        print(f"\n{Colors.BOLD}Średnie podobieństwo (top {topn}):{Colors.ENDC}")
        for model_name, data in results.items():
            avg_sim = np.mean([s for _, s in data['similar']])
            print(f"  • {model_name.upper()}: {avg_sim:.4f}")

        print(f"\n{Colors.BOLD}Norma wektora:{Colors.ENDC}")
        for model_name, data in results.items():
            norm = np.linalg.norm(data['vector'])
            print(f"  • {model_name.upper()}: {norm:.4f}")

        # Podobieństwo między wektorami
        if len(results) >= 2:
            print(f"\n{Colors.BOLD}Podobieństwo cosinusowe między wektorami:{Colors.ENDC}")
            model_names = list(results.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    name1, name2 = model_names[i], model_names[j]
                    vec1, vec2 = results[name1]['vector'], results[name2]['vector']
                    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    print(f"  • {name1.upper()} ↔ {name2.upper()}: {cos_sim:.4f}")

# --- Przykłady demonstracyjne ---

def run_demo():
    """Uruchamia demonstrację z przykładowymi zdaniami."""
    demo_sentences = [
        "Jestem głodny.",
        "Kot siedzi na macie.",
        "Piękna pogoda dzisiaj.",
        "Król Polski przyjechał do Warszawy.",
        "Czytam książki w bibliotece."
    ]

    print_header("DEMONSTRACJA - Przykładowe zdania")
    print(f"\nUruchamiam analizę {len(demo_sentences)} przykładowych zdań...\n")

    for i, sentence in enumerate(demo_sentences, 1):
        print(f"\n{Colors.BOLD}{Colors.YELLOW}═══ Przykład {i}/{len(demo_sentences)} ═══{Colors.ENDC}")
        compare_sentence(sentence, topn=3)

        if i < len(demo_sentences):
            input(f"\n{Colors.CYAN}Naciśnij Enter aby kontynuować...{Colors.ENDC}")

# --- Tryb interaktywny ---

def interactive_mode():
    """Tryb interaktywny - użytkownik wprowadza własne zdania."""
    print_header("TRYB INTERAKTYWNY")
    print(f"\n{Colors.BOLD}Wprowadź własne zdania aby porównać tokenizację.{Colors.ENDC}")
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

            compare_sentence(user_input, topn=5)
            print()  # Dodatkowa linia dla czytelności

        except EOFError:
            print(f"\n{Colors.YELLOW}Zakończono.{Colors.ENDC}")
            break
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Przerwano przez użytkownika.{Colors.ENDC}")
            break

# --- Menu główne ---

def main_menu():
    """Wyświetla menu główne i obsługuje wybór użytkownika."""
    while True:
        print_header("MENU GŁÓWNE")
        print(f"\n{Colors.BOLD}Wybierz opcję:{Colors.ENDC}")
        print(f"  {Colors.CYAN}1.{Colors.ENDC} Uruchom demonstrację (przykładowe zdania)")
        print(f"  {Colors.CYAN}2.{Colors.ENDC} Tryb interaktywny (własne zdania)")
        print(f"  {Colors.CYAN}3.{Colors.ENDC} Statystyki modeli")
        print(f"  {Colors.CYAN}q.{Colors.ENDC} Zakończ")

        try:
            choice = input(f"\n{Colors.BOLD}{Colors.GREEN}Wybór > {Colors.ENDC}").strip().lower()

            if choice == '1':
                run_demo()
            elif choice == '2':
                interactive_mode()
            elif choice == '3':
                show_model_statistics()
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

def show_model_statistics():
    """Wyświetla statystyki załadowanych modeli."""
    print_header("STATYSTYKI MODELI")

    for model_name in available_models:
        model = models[model_name]
        print(f"\n{Colors.BOLD}{Colors.BLUE}Model: {model_name.upper()}{Colors.ENDC}")
        print(f"  ├─ Rozmiar słownika: {len(model.wv):,} unikalnych tokenów")
        print(f"  ├─ Wymiar wektora: {model.vector_size}")
        print(f"  ├─ Liczba epok treningu: {model.epochs}")
        print(f"  ├─ Liczba dokumentów: {len(model.dv):,}")
        print(f"  └─ Okno kontekstu: {model.window}")

    input(f"\n{Colors.CYAN}Naciśnij Enter aby wrócić do menu...{Colors.ENDC}")

# --- Uruchomienie programu ---

if __name__ == "__main__":
    try:
        main_menu()
    except Exception as e:
        print_error(f"Nieoczekiwany błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

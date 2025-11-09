#!/usr/bin/env python3
"""
Wrapper do trenowania obu modeli Doc2Vec (BPE i SIMPLE) z ładnym wyświetlaniem postępu.
"""

import subprocess
import sys
import time
from datetime import datetime

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

def print_header(text):
    """Wyświetla nagłówek z kolorami."""
    width = 80
    print("\n" + Colors.BOLD + Colors.CYAN + "="*width + Colors.ENDC)
    print(Colors.BOLD + Colors.CYAN + text.center(width) + Colors.ENDC)
    print(Colors.BOLD + Colors.CYAN + "="*width + Colors.ENDC + "\n")

def print_step(step_num, total_steps, description):
    """Wyświetla krok z numerem."""
    print(f"{Colors.BOLD}[{step_num}/{total_steps}]{Colors.ENDC} {Colors.BLUE}{description}{Colors.ENDC}")

def print_success(text):
    """Wyświetla komunikat sukcesu."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    """Wyświetla komunikat błędu."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_info(text):
    """Wyświetla informację."""
    print(f"{Colors.YELLOW}ℹ {text}{Colors.ENDC}")

def run_training(script_name, model_type):
    """
    Uruchamia skrypt treningowy i wyświetla jego output w czasie rzeczywistym.

    Args:
        script_name: Nazwa skryptu do uruchomienia
        model_type: Typ modelu (do wyświetlania)

    Returns:
        True jeśli sukces, False jeśli błąd
    """
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}Rozpoczynam trening: {model_type}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.ENDC}\n")

    start_time = time.time()

    try:
        # Uruchom skrypt i wyświetlaj output w czasie rzeczywistym
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Wyświetlaj output linia po linii
        for line in process.stdout:
            print(line, end='')

        # Poczekaj na zakończenie procesu
        return_code = process.wait()

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.ENDC}")

        if return_code == 0:
            print_success(f"{model_type} - trening zakończony pomyślnie!")
            print_info(f"Czas trwania: {duration:.2f}s ({duration/60:.2f} min)")
            print(f"{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.ENDC}\n")
            return True
        else:
            print_error(f"{model_type} - trening zakończony z błędem (kod: {return_code})")
            print(f"{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.ENDC}\n")
            return False

    except Exception as e:
        print_error(f"Błąd podczas uruchamiania {script_name}: {e}")
        return False

def main():
    """Główna funkcja wrappera."""
    print_header("TRENING MODELI DOC2VEC - BPE vs SIMPLE")

    print(f"{Colors.BOLD}Rozpoczęto:{Colors.ENDC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Lista modeli do wytrenowania
    models = [
        {
            'script': 'train-doc2vec.py',
            'name': 'MODEL SIMPLE (split tokenization)',
            'number': 1
        },
        {
            'script': 'train-doc2vec-bpe.py',
            'name': 'MODEL BPE (Byte Pair Encoding)',
            'number': 2
        },
        {
            'script': 'train-doc2vec-spacy.py',
            'name': 'MODEL SPACY (lemmatization)',
            'number': 3
        }
    ]

    total_models = len(models)
    successful_models = []
    failed_models = []

    overall_start = time.time()

    # Trenuj każdy model
    for model in models:
        print_step(model['number'], total_models, model['name'])

        success = run_training(model['script'], model['name'])

        if success:
            successful_models.append(model['name'])
        else:
            failed_models.append(model['name'])
            print_info("Kontynuuję z następnym modelem...\n")

    overall_end = time.time()
    total_duration = overall_end - overall_start

    # Podsumowanie
    print_header("PODSUMOWANIE TRENINGU")

    print(f"{Colors.BOLD}Całkowity czas:{Colors.ENDC} {total_duration:.2f}s ({total_duration/60:.2f} min)\n")

    if successful_models:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Pomyślnie wytrenowane modele ({len(successful_models)}/{total_models}):{Colors.ENDC}")
        for model_name in successful_models:
            print(f"  • {model_name}")
        print()

    if failed_models:
        print(f"{Colors.RED}{Colors.BOLD}✗ Nieudane treningi ({len(failed_models)}/{total_models}):{Colors.ENDC}")
        for model_name in failed_models:
            print(f"  • {model_name}")
        print()

    # Następne kroki
    if len(successful_models) == total_models:
        print_header("NASTĘPNE KROKI")
        print(f"{Colors.GREEN}Wszystkie modele zostały pomyślnie wytrenowane!{Colors.ENDC}\n")
        print(f"{Colors.BOLD}Uruchom porównanie:{Colors.ENDC}")
        print(f"  python compare-tokenization.py\n")
        return 0
    elif successful_models:
        print_info("Niektóre modele zostały wytrenowane. Możesz uruchomić porównanie dla dostępnych modeli.")
        return 1
    else:
        print_error("Żaden model nie został wytrenowany pomyślnie.")
        return 2

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Przerwano przez użytkownika.{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Nieoczekiwany błąd: {e}")
        sys.exit(1)

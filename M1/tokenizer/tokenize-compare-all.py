import os
import glob
from pathlib import Path
from tokenizers import Tokenizer
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

# Teksty testowe
TEST_TEXTS = [
    "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
    "Jakże mi wesoło!",
    "Pan Tadeusz czyli ostatni zajazd na Litwie",
    "Sztuczna inteligencja i uczenie maszynowe",
    "To jest przykładowy tekst do tokenizacji.",
    "This is some random text in english, and i like cats!"
]

TOKENIZERS_DIR = "tokenizers"
console = Console()

def load_all_tokenizers() -> Dict[str, Tokenizer]:
    """Ładuje wszystkie tokenizery z folderu tokenizers/"""
    tokenizers = {}
    json_files = glob.glob(f"{TOKENIZERS_DIR}/*.json")

    for filepath in sorted(json_files):
        name = Path(filepath).stem
        try:
            tokenizers[name] = Tokenizer.from_file(filepath)
            console.print(f"✓ Załadowano: [bold green]{name}[/bold green]")
        except Exception as e:
            console.print(f"✗ Błąd ładowania {name}: {e}", style="red")

    return tokenizers

def get_tokenizer_stats(tokenizer: Tokenizer) -> Dict[str, any]:
    """Zwraca statystyki tokenizera"""
    try:
        vocab_size = tokenizer.get_vocab_size()
        return {"vocab_size": vocab_size}
    except:
        return {"vocab_size": "N/A"}

def tokenize_and_analyze(text: str, tokenizer: Tokenizer) -> Dict[str, any]:
    """Tokenizuje tekst i zwraca analizę"""
    encoding = tokenizer.encode(text)

    return {
        "tokens": encoding.tokens,
        "ids": encoding.ids,
        "token_count": len(encoding.tokens),
        "avg_token_length": sum(len(t) for t in encoding.tokens) / len(encoding.tokens) if encoding.tokens else 0,
    }

def create_bar_chart(values: List[int], labels: List[str], title: str, max_width: int = 50):
    """Tworzy prosty wykres słupkowy ASCII"""
    if not values:
        return

    max_value = max(values)
    console.print(f"\n[bold cyan]{title}[/bold cyan]")

    for label, value in zip(labels, values):
        bar_length = int((value / max_value) * max_width) if max_value > 0 else 0
        bar = "█" * bar_length
        color = "green" if value == min(values) else "yellow" if value == max(values) else "blue"
        console.print(f"{label:<30} [{color}]{bar}[/{color}] {value}")

def visualize_tokens_colorful(tokens: List[str], tokenizer_name: str):
    """Wizualizuje tokeny z kolorowym formatowaniem"""
    text = Text()

    colors = ["cyan", "yellow", "green", "magenta", "blue", "red"]

    for i, token in enumerate(tokens[:30]):  # Pierwsze 30 tokenów
        color = colors[i % len(colors)]
        text.append(f"[{token}]", style=f"bold {color}")
        if i < len(tokens) - 1:
            text.append(" ", style="dim")

    if len(tokens) > 30:
        text.append(f" ... (+{len(tokens) - 30} więcej)", style="dim italic")

    panel = Panel(
        text,
        title=f"[bold]{tokenizer_name}[/bold]",
        border_style="blue",
        box=box.ROUNDED,
    )
    console.print(panel)

def compare_tokenizers_on_text(text: str, tokenizers: Dict[str, Tokenizer]):
    """Porównuje wszystkie tokenizery na jednym tekście"""
    console.print("\n")
    console.rule(f"[bold magenta]TEKST: \"{text}\"[/bold magenta]")

    results = []
    for name, tokenizer in tokenizers.items():
        analysis = tokenize_and_analyze(text, tokenizer)
        results.append((name, analysis))

    # Sortuj wyniki według liczby tokenów (rosnąco - mniej tokenów = lepiej)
    results.sort(key=lambda x: x[1]['token_count'])

    # Tabela porównawcza
    table = Table(
        title="📊 Statystyki Tokenizacji",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Tokenizer", style="bold yellow", width=30)
    table.add_column("Liczba tokenów", justify="right", style="green")
    table.add_column("Śr. długość tokenu", justify="right", style="blue")
    table.add_column("Efektywność", justify="center", style="magenta")

    # Znajdź najlepszy (najmniej tokenów)
    min_tokens = min(r[1]['token_count'] for r in results)

    for name, analysis in results:
        token_count = analysis['token_count']
        avg_len = f"{analysis['avg_token_length']:.2f}"

        # Ocena efektywności
        if token_count == min_tokens:
            efficiency = "⭐⭐⭐ Najlepszy"
            style = "bold green"
        elif token_count <= min_tokens * 1.2:
            efficiency = "⭐⭐ Dobry"
            style = "green"
        elif token_count <= min_tokens * 1.5:
            efficiency = "⭐ Średni"
            style = "yellow"
        else:
            efficiency = "❌ Słaby"
            style = "red"

        table.add_row(
            name,
            str(token_count),
            avg_len,
            efficiency,
            style=style if token_count == min_tokens else None
        )

    console.print(table)

    # Wykres słupkowy liczby tokenów
    token_counts = [r[1]['token_count'] for r in results]
    labels = [r[0] for r in results]
    create_bar_chart(token_counts, labels, "📈 Liczba tokenów (mniej = lepiej)")

    # Wizualizacja tokenów
    console.print("\n[bold cyan]🔍 WIZUALIZACJA TOKENÓW:[/bold cyan]\n")
    for name, analysis in results:
        visualize_tokens_colorful(analysis['tokens'], name)

def print_summary_table(tokenizers: Dict[str, Tokenizer]):
    """Drukuje tabelę podsumowującą wszystkie tokenizery"""
    table = Table(
        title="📚 Podsumowanie Tokenizerów",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Nazwa tokenizera", style="bold cyan", width=30)
    table.add_column("Rozmiar słownika", justify="right", style="yellow")
    table.add_column("Status", justify="center", style="green")

    for name, tokenizer in tokenizers.items():
        stats = get_tokenizer_stats(tokenizer)
        vocab = stats['vocab_size']

        # Określ typ na podstawie nazwy
        if 'bielik' in name.lower():
            status = "🎯 Profesjonalny"
        elif 'custom' in name.lower() or 'bpe' in name.lower():
            status = "🔧 Własny"
        else:
            status = "📝 Standard"

        vocab_str = f"{vocab:,}" if isinstance(vocab, int) else str(vocab)
        table.add_row(name, vocab_str, status)

    console.print(table)

def main():
    console.clear()

    # Header
    console.print(Panel.fit(
        "[bold yellow]🔬 PORÓWNANIE WSZYSTKICH TOKENIZERÓW 🔬[/bold yellow]\n"
        "[dim]Analiza wydajności i jakości tokenizacji[/dim]",
        border_style="bold blue",
        box=box.DOUBLE
    ))

    console.print()

    # Ładowanie tokenizerów
    with console.status("[bold green]Ładowanie tokenizerów...", spinner="dots"):
        tokenizers = load_all_tokenizers()

    if not tokenizers:
        console.print("❌ [bold red]Nie znaleziono żadnych tokenizerów w folderze 'tokenizers/'[/bold red]")
        return

    console.print(f"\n✅ [bold green]Załadowano {len(tokenizers)} tokenizerów[/bold green]\n")

    # Podsumowanie
    print_summary_table(tokenizers)

    # Porównanie na każdym tekście testowym
    for text in TEST_TEXTS:
        compare_tokenizers_on_text(text, tokenizers)

    # Footer
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]✅ KONIEC PORÓWNANIA[/bold green]\n"
        "[dim]Najlepszy tokenizer to ten z najmniejszą liczbą tokenów dla danego tekstu[/dim]",
        border_style="bold green"
    ))

if __name__ == "__main__":
    main()

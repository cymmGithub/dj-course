import numpy as np
import json
import logging
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Ustawienie logowania dla gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Pliki dla modelu SIMPLE
MODEL_FILE_SIMPLE = "doc2vec_model_simple.model"
SENTENCE_MAP_FILE_SIMPLE = "doc2vec_model_sentence_map_simple.json"

print("\n" + "="*80)
print("  WIZUALIZACJA DOC2VEC (SIMPLE TOKENIZATION)")
print("="*80)

# --- ETAP 1: Wczytanie Modelu ---
print("\n--- Wczytywanie modelu ---")

try:
    model_simple = Doc2Vec.load(MODEL_FILE_SIMPLE)
    print(f"✓ Model SIMPLE wczytany z: {MODEL_FILE_SIMPLE}")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku modelu SIMPLE '{MODEL_FILE_SIMPLE}'.")
    print("Uruchom najpierw: python train-doc2vec.py")
    raise

try:
    with open(SENTENCE_MAP_FILE_SIMPLE, "r", encoding="utf-8") as f:
        sentence_lookup_simple = json.load(f)
    print(f"✓ Mapa zdań SIMPLE wczytana z: {SENTENCE_MAP_FILE_SIMPLE}")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku mapy zdań SIMPLE '{SENTENCE_MAP_FILE_SIMPLE}'.")
    raise

# --- ETAP 2: Statystyki Modelu ---
print("\n" + "="*80)
print("  STATYSTYKI MODELU")
print("="*80)

print(f"\n📊 Model SIMPLE Doc2Vec:")
print(f"  ├─ Liczba wektorów zdań: {len(model_simple.dv):,}")
print(f"  ├─ Wymiar wektora: {model_simple.vector_size}")
print(f"  ├─ Rozmiar słownika: {len(model_simple.wv):,}")
print(f"  ├─ Rozmiar okna: {model_simple.window}")
print(f"  ├─ Min. liczba wystąpień: {model_simple.min_count}")
print(f"  └─ Liczba epok treningu: {model_simple.epochs}")

# --- ETAP 3: Przykładowe Wnioskowanie ---
print("\n" + "="*80)
print("  PRZYKŁADOWE WNIOSKOWANIE")
print("="*80)

test_sentences = [
    "Jestem głodny.",
    "Kot siedzi na macie.",
    "Piękna pogoda dzisiaj.",
    "Król Polski przyjechał do Warszawy."
]

for i, sentence in enumerate(test_sentences, 1):
    print(f"\n--- Przykład {i} ---")
    print(f"Zdanie: \"{sentence}\"")

    # Tokenizacja SIMPLE
    tokens_simple = sentence.split()
    print(f"Tokeny ({len(tokens_simple)}): {tokens_simple}")

    # Wnioskowanie wektora
    inferred_vector = model_simple.infer_vector(tokens_simple, epochs=model_simple.epochs)

    # Znalezienie podobnych zdań
    similar_docs = model_simple.dv.most_similar([inferred_vector], topn=5)

    print(f"\nTop 5 najbardziej podobnych zdań:")
    for rank, (doc_id_str, similarity) in enumerate(similar_docs, 1):
        doc_index = int(doc_id_str)
        try:
            original_sentence = sentence_lookup_simple[doc_index]
            print(f"  {rank}. [{similarity:.4f}] {original_sentence[:70]}")
        except IndexError:
            print(f"  {rank}. [{similarity:.4f}] BŁĄD: Nie znaleziono zdania")

# --- ETAP 4: Wizualizacja PCA ---
print("\n" + "="*80)
print("  WIZUALIZACJA PCA (2D)")
print("="*80)

# Pobierz próbkę wektorów (dla lepszej wydajności)
num_samples = min(1000, len(model_simple.dv))
print(f"\nGenerowanie wizualizacji PCA dla {num_samples} wektorów...")

# Pobierz wektory
doc_vectors = []
doc_labels = []
for i in range(num_samples):
    doc_vectors.append(model_simple.dv[str(i)])
    # Pobierz pierwsze 30 znaków zdania jako etykietę
    doc_labels.append(sentence_lookup_simple[i][:30])

doc_vectors = np.array(doc_vectors)

# Wykonaj PCA
pca = PCA(n_components=2)
vectors_2d_pca = pca.fit_transform(doc_vectors)

print(f"✓ PCA zakończone. Wyjaśniona wariancja: {sum(pca.explained_variance_ratio_):.2%}")

# Stwórz wykres PCA
plt.figure(figsize=(14, 10))
plt.scatter(vectors_2d_pca[:, 0], vectors_2d_pca[:, 1], alpha=0.5, s=10)
plt.title(f'Wizualizacja Doc2Vec (PCA) - {num_samples} wektorów zdań', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} wariancji)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} wariancji)')
plt.grid(True, alpha=0.3)

# Dodaj etykiety dla pierwszych 20 punktów
for i in range(min(20, num_samples)):
    plt.annotate(doc_labels[i],
                xy=(vectors_2d_pca[i, 0], vectors_2d_pca[i, 1]),
                fontsize=7, alpha=0.7,
                xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
pca_filename = "results/doc2vec_pca_visualization.png"
plt.savefig(pca_filename, dpi=150, bbox_inches='tight')
print(f"✓ Wykres PCA zapisany jako: {pca_filename}")
plt.close()

# --- ETAP 5: Wizualizacja t-SNE ---
print("\n" + "="*80)
print("  WIZUALIZACJA t-SNE (2D)")
print("="*80)

# Użyj mniejszej próbki dla t-SNE (jest wolniejsze)
num_samples_tsne = min(500, len(model_simple.dv))
print(f"\nGenerowanie wizualizacji t-SNE dla {num_samples_tsne} wektorów...")
print("(To może potrwać kilka sekund...)")

# Pobierz wektory
tsne_vectors = []
tsne_labels = []
for i in range(num_samples_tsne):
    tsne_vectors.append(model_simple.dv[str(i)])
    tsne_labels.append(sentence_lookup_simple[i][:30])

tsne_vectors = np.array(tsne_vectors)

# Wykonaj t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
vectors_2d_tsne = tsne.fit_transform(tsne_vectors)

print(f"✓ t-SNE zakończone")

# Stwórz wykres t-SNE
plt.figure(figsize=(14, 10))
plt.scatter(vectors_2d_tsne[:, 0], vectors_2d_tsne[:, 1], alpha=0.5, s=10, c=range(num_samples_tsne), cmap='viridis')
plt.title(f'Wizualizacja Doc2Vec (t-SNE) - {num_samples_tsne} wektorów zdań', fontsize=14)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(label='Indeks zdania')
plt.grid(True, alpha=0.3)

# Dodaj etykiety dla pierwszych 15 punktów
for i in range(min(15, num_samples_tsne)):
    plt.annotate(tsne_labels[i],
                xy=(vectors_2d_tsne[i, 0], vectors_2d_tsne[i, 1]),
                fontsize=7, alpha=0.7,
                xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
tsne_filename = "results/doc2vec_tsne_visualization.png"
plt.savefig(tsne_filename, dpi=150, bbox_inches='tight')
print(f"✓ Wykres t-SNE zapisany jako: {tsne_filename}")
plt.close()

# --- ETAP 6: Analiza Podobieństwa Słów ---
print("\n" + "="*80)
print("  ANALIZA PODOBIEŃSTWA SŁÓW")
print("="*80)

test_words = ["król", "kot", "dom", "wojna", "miłość"]
print(f"\nAnalizowanie podobieństwa dla słów: {test_words}")

for word in test_words:
    if word in model_simple.wv:
        print(f"\n🔍 Słowo: '{word}'")
        similar_words = model_simple.wv.most_similar(word, topn=5)
        print(f"  Top 5 najbardziej podobnych słów:")
        for rank, (similar_word, similarity) in enumerate(similar_words, 1):
            print(f"    {rank}. [{similarity:.4f}] {similar_word}")
    else:
        print(f"\n⚠️  Słowo '{word}' nie znajduje się w słowniku modelu")

# --- ETAP 7: Interaktywny Tryb ---
print("\n" + "="*80)
print("  INTERAKTYWNY TRYB WNIOSKOWANIA")
print("="*80)

print("\n\nCzy chcesz przetestować własne zdania? (t/n): ", end="")
try:
    choice = input().strip().lower()

    if choice == 't' or choice == 'y':
        while True:
            print("\n" + "-"*80)
            user_sentence = input("Wprowadź zdanie (lub 'q' aby zakończyć): ").strip()

            if user_sentence.lower() == 'q':
                print("Zakończono tryb interaktywny.")
                break

            if not user_sentence:
                print("Zdanie nie może być puste.")
                continue

            print(f"\n🔍 Analiza zdania: \"{user_sentence}\"")

            # Tokenizacja
            tokens_simple = user_sentence.split()
            print(f"  Tokeny ({len(tokens_simple)}): {tokens_simple}")

            # Wnioskowanie wektora
            vector_simple = model_simple.infer_vector(tokens_simple, epochs=model_simple.epochs)

            # Statystyki wektora
            print(f"  Norma wektora: {np.linalg.norm(vector_simple):.4f}")
            print(f"  Średnia wartość: {np.mean(vector_simple):.4f}")
            print(f"  Odchylenie standardowe: {np.std(vector_simple):.4f}")

            # Znalezienie podobnych zdań
            similar_simple = model_simple.dv.most_similar([vector_simple], topn=5)

            print(f"\n  📋 Top 5 najbardziej podobnych zdań:")
            for rank, (doc_id, sim) in enumerate(similar_simple, 1):
                sent = sentence_lookup_simple[int(doc_id)]
                print(f"    {rank:2d}. [{sim:.4f}] {sent[:70]}")

            # Analiza słów z wprowadzonego zdania
            print(f"\n  🔤 Analiza słów z wprowadzonego zdania:")
            for token in tokens_simple[:5]:  # Analiza pierwszych 5 słów
                if token in model_simple.wv:
                    similar_words = model_simple.wv.most_similar(token, topn=3)
                    similar_words_str = ", ".join([f"{w} ({s:.3f})" for w, s in similar_words])
                    print(f"    '{token}' → {similar_words_str}")
                else:
                    print(f"    '{token}' → (nie ma w słowniku)")

except EOFError:
    print("\n\nZakończono.")
except KeyboardInterrupt:
    print("\n\nPrzerwano przez użytkownika.")

print("\n" + "="*80)
print("  WIZUALIZACJA ZAKOŃCZONA")
print("="*80)
print(f"\nWygenerowane pliki:")
print(f"  ├─ {pca_filename}")
print(f"  └─ {tsne_filename}")
print("\n")

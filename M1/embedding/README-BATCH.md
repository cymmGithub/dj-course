# Batch Training System for Word2Vec CBOW

This system allows you to train and compare Word2Vec CBOW models across multiple tokenizer × dataset combinations automatically.

## Overview

- **12 tokenizers** available in `M1/tokenizer/tokenizers/`
- **4 datasets** defined in `corpora.py`: NKJP, WOLNELEKTURY, PAN_TADEUSZ, ALL
- **48 total combinations** to train and compare

## Files

### Training Scripts

- **`train-cbow-batch.py`** - Full batch training of all combinations
- **`train-cbow-batch-test.py`** - Test script with reduced parameters (3 tokenizers × 2 datasets = 6 combinations)

### Visualization Scripts

- **`visualize-cbow-batch.py`** - Complete visualization of all trained models
- **`visualize-cbow-batch-test.py`** - Visualization for test results

### Original Scripts (unchanged)

- `train-cbow.py` - Original single model training
- `visualize-cbow.py` - Original single model visualization
- `corpora.py` - Dataset definitions

## Quick Start

### 1. Test Run (Recommended First)

Test with a small subset to verify everything works:

```bash
cd M1/embedding
python train-cbow-batch-test.py
python visualize-cbow-batch-test.py
```

This trains 6 models with reduced parameters (10 epochs, vector_length=20) and takes ~5-10 minutes.

### 2. Full Batch Training

Train all 48 combinations:

```bash
python train-cbow-batch.py
```

This will take significantly longer (~2-4 hours depending on your hardware).

### 3. Visualize Results

Generate comprehensive comparison charts:

```bash
python visualize-cbow-batch.py
```

## Output Structure

```
results/
├── models/
│   ├── all-tokenizer_NKJP_cbow_model.model
│   ├── all-tokenizer_WOLNELEKTURY_cbow_model.model
│   ├── bielik-v1-tokenizer_NKJP_cbow_model.model
│   └── ... (48 models total)
└── metrics_summary.csv

results-test/  (from test run)
├── models/
│   └── ... (6 models)
└── metrics_summary.csv
```

## Customization

### Select Specific Combinations

Edit the scripts to train only specific combinations:

```python
# In train-cbow-batch.py or train-cbow-batch-test.py

TOKENIZERS_TO_USE = ["all-tokenizer.json", "bielik-v1-tokenizer.json"]
DATASETS_TO_USE = ["PAN_TADEUSZ", "WOLNELEKTURY"]
```

Set to `None` to use all available options.

### Adjust Training Parameters

```python
VECTOR_LENGTH = 40      # Embedding dimension
WINDOW_SIZE = 5         # Context window size
MIN_COUNT = 1           # Minimum token frequency
WORKERS = 8             # CPU threads
EPOCHS = 50             # Training epochs
SAMPLE_RATE = 1e-2      # Downsampling rate
SG_MODE = 0             # 0=CBOW, 1=Skip-gram
```

## Visualization Features

The batch visualization provides:

1. **Metrics Summary Table**
   - Vocabulary size
   - Total tokens processed
   - Training time
   - Model file size

2. **Dataset Comparison**
   - Average metrics per dataset
   - Performance across tokenizers

3. **Tokenizer Comparison**
   - Average metrics per tokenizer
   - Performance across datasets

4. **Word Similarity Grid**
   - Side-by-side comparison of similar words
   - Test words: 'wojsko', 'szlachta', 'choroba', 'król'
   - Grouped by dataset for easy comparison

## Resume Capability

The training scripts automatically skip already-trained combinations, so you can:
- Stop and resume training at any time
- Re-run the script to train only missing combinations
- Delete specific model files to retrain those combinations

## Example Output

### Training Progress
```
📍 Kombinacja 5/48
════════════════════════════════════════════════════════════════════════════════════════════════════
  🔄 TRENING: bielik-v1-tokenizer_WOLNELEKTURY
════════════════════════════════════════════════════════════════════════════════════════════════════
  Tokenizer: bielik-v1-tokenizer.json
  Dataset: WOLNELEKTURY (28 plików)
  📝 Wczytano 124,562 zdań
  🔢 Tokenizacja: 3,456,789 tokenów
  🏋️  Rozpoczynam trening...
  ✅ Zakończono w 45.3s
     Słownik: 12,345 tokenów | Rozmiar: 2.34 MB
```

### Visualization Grid
```
┌─────────────────────────┬─────────────┬──────────┬────────────┬──────────┬──────────┐
│       Tokenizer         │   Dataset   │   Vocab  │   Tokens   │ Time(s)  │ Size(MB) │
├─────────────────────────┼─────────────┼──────────┼────────────┼──────────┼──────────┤
│ all-tokenizer           │ PAN_TADEUSZ │   8,234  │   456,789  │   12.3   │   1.45   │
│ bielik-v1               │ PAN_TADEUSZ │   7,891  │   467,234  │   13.1   │   1.38   │
│ custom_bpe              │ PAN_TADEUSZ │   9,012  │   445,123  │   11.8   │   1.56   │
└─────────────────────────┴─────────────┴──────────┴────────────┴──────────┴──────────┘
```

## Tips

- **Start with test run** to verify setup before full training
- **Monitor disk space** - 48 models will take ~100-200 MB
- **Use powerful hardware** - More CPU cores = faster training (adjust WORKERS)
- **Customize test words** in visualization scripts to test domain-specific vocabulary

## Troubleshooting

### "File not found" errors
- Verify corpus files exist in `../korpus-nkjp/output/` and `../korpus-wolnelektury/`
- Check that tokenizer files exist in `../tokenizer/tokenizers/`

### Out of memory
- Reduce `VECTOR_LENGTH` and `WORKERS`
- Train fewer combinations at once
- Use test script with reduced parameters

### Slow training
- Reduce `EPOCHS`
- Reduce `VECTOR_LENGTH`
- Increase `MIN_COUNT` (filters rare tokens)
- Ensure `WORKERS` matches your CPU core count

## Next Steps

After training and visualization, you can:
- Compare which tokenizer works best for your use case
- Identify which dataset provides better semantic representations
- Use the best-performing model in production
- Export embeddings for downstream tasks

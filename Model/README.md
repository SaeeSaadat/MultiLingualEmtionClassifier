# Multilingual Emotion Classifier

A multilingual emotion classifier fine-tuned on [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-large) using the [BRIGHTER dataset](https://brighter-dataset.github.io/). Given a sentence in a supported language, the model predicts one or more emotions from: **anger, disgust, fear, joy, sadness, surprise**.


## Installation

```bash
pip install -r requirements.txt
```

## Prediction (`predict.py`)

### Single sentence

```bash
python predict.py --model_dir output/xlmr_large_improved_seed42 \
    --text "I can't believe how amazing this turned out!"
```

### Specify language

```bash
python predict.py --model_dir output/xlmr_large_improved_seed42 \
    --lang deu --text "Ich habe große Angst vor der Prüfung."
```

### Multiple sentences

```bash
python predict.py --model_dir output/xlmr_large_improved_seed42 \
    --text "I'm so happy!" "This is terrifying."
```

### From a file (one sentence per line)

```bash
python predict.py --model_dir output/xlmr_large_improved_seed42 \
    --file test_sentences/English_texts.txt
```

### Interactive mode

```bash
python predict.py --model_dir output/xlmr_large_improved_seed42 --lang eng
```

Type a sentence and press Enter. Blank line or Ctrl-D to quit.

### JSON output

```bash
python predict.py --model_dir output/xlmr_large_improved_seed42 \
    --text "I'm so done with everything." --json
```

### Override decision threshold

```bash
python predict.py --model_dir output/xlmr_large_improved_seed42 \
    --text "..." --threshold 0.4
```

## Evaluation (`evaluate.py`)

Runs the classifier on three parallel test sets (English, Spanish, Persian) and reports per-label and macro F1 against the expected results.

```bash
python evaluate.py --model_dir output/xlmr_large_improved_seed42
```

Optional threshold override:

```bash
python evaluate.py --model_dir output/xlmr_large_improved_seed42 --threshold 0.4
```

## Test sentences (`test_sentences/`)

| File | Description |
|---|---|
| `English_texts.txt` | 50 English sentences |
| `spanish_texts.txt` | Same 50 sentences translated to Spanish |
| `persian_texts.txt` | Same 50 sentences translated to Persian (Farsi) |
| `expected_results.txt` | Gold-standard emotion labels, one line per sentence, space-separated |

All three language files share the same line order, so line *N* in each file is the same sentence and corresponds to line *N* in `expected_results.txt`.
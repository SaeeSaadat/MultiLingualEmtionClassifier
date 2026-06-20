#!/usr/bin/env python3
"""
evaluate.py — measure macro F1 of the emotion classifier on multilingual test sets.

Usage:
    python evaluate.py --model_dir output/xlmr_large_improved_seed42
    python evaluate.py --model_dir output/xlmr_large_improved_seed42 --threshold 0.4
"""

import os
import sys
import argparse

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import EmotionPredictor, EVAL_LABELS

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_sentences")

LANGUAGE_FILES = [
    ("eng", "English_texts.txt",  "English"),
    ("spa", "spanish_texts.txt",  "Spanish"),
    ("fas", "persian_texts.txt",  "Persian"),
]
EXPECTED_FILE = os.path.join(TEST_DIR, "expected_results.txt")


def load_expected(path):
    with open(path, encoding="utf-8") as f:
        return [set(line.strip().split()) for line in f if line.strip()]


def load_texts(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def to_binary(label_sets, labels):
    mat = np.zeros((len(label_sets), len(labels)), dtype=int)
    for i, s in enumerate(label_sets):
        for j, label in enumerate(labels):
            if label in s:
                mat[i, j] = 1
    return mat


def evaluate(model_dir, threshold=None):
    expected = load_expected(EXPECTED_FILE)

    # Only evaluate on labels that actually appear in the reference file
    referenced_labels = set().union(*expected)
    eval_labels = [l for l in EVAL_LABELS if l in referenced_labels]

    predictor = EmotionPredictor(model_dir)

    print(f"Model : {model_dir}")
    print(f"Labels: {eval_labels}")
    print(f"Sentences per language: {len(expected)}\n")

    summary = {}
    for lang_code, filename, lang_name in LANGUAGE_FILES:
        texts_path = os.path.join(TEST_DIR, filename)
        texts = load_texts(texts_path)

        if len(texts) != len(expected):
            print(f"[warn] {filename} has {len(texts)} lines but expected "
                  f"{len(expected)} — skipping {lang_name}.", file=sys.stderr)
            continue

        results = predictor.predict(texts, lang=lang_code, threshold=threshold)
        predicted_sets = [set(r["predicted"]) for r in results]

        y_true = to_binary(expected,       eval_labels)
        y_pred = to_binary(predicted_sets, eval_labels)

        macro_f1      = f1_score(y_true, y_pred, average="macro",  zero_division=0)
        per_label_f1  = f1_score(y_true, y_pred, average=None,     zero_division=0)

        summary[lang_name] = macro_f1

        print(f"=== {lang_name} ({lang_code}) ===")
        print(f"Macro F1 : {macro_f1:.4f}")
        print("Per-label F1:")
        for label, score in zip(eval_labels, per_label_f1):
            print(f"  {label:10s}: {score:.4f}")
        print()

    print("=== Summary ===")
    for lang_name, macro_f1 in summary.items():
        print(f"  {lang_name:10s}: {macro_f1:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate emotion classifier macro F1.")
    ap.add_argument("--model_dir", required=True,
                    help="Folder with best_model.pt, run_config.json, tokenizer/, thresholds")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override decision threshold for all labels (e.g. 0.4)")
    args = ap.parse_args()
    evaluate(args.model_dir, args.threshold)


if __name__ == "__main__":
    main()

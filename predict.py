#!/usr/bin/env python3
"""
predict.py — run the trained multilingual emotion model on sentences.

Self-contained: the model class is inlined, so to share this all someone needs is
this file + a trained model folder (best_model.pt, run_config.json, tokenizer/,
and a thresholds file). No web UI — command line / file / interactive / importable.

Dependencies:  pip install torch transformers numpy

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
# one sentence (English is the default language)
python src/predict.py --model_dir outputs/xlmr_large_per_lang_clear \
       --text "I am so happy we won, but I'm exhausted from the long match!"

# pick the language (must be one the model was trained on: eng / deu / arq)
python src/predict.py --model_dir outputs/xlmr_large_per_lang_clear \
       --lang deu --text "Ich habe große Angst vor der Prüfung."

# several sentences, or a file with one sentence per line
python src/predict.py --model_dir <dir> --text "Sentence one." "Sentence two."
python src/predict.py --model_dir <dir> --file my_sentences.txt

# interactive: type sentences, blank line / Ctrl-D to quit
python src/predict.py --model_dir <dir> --lang eng

# machine-readable output
python src/predict.py --model_dir <dir> --text "..." --json

------------------------------------------------------------------------------
AS A LIBRARY
------------------------------------------------------------------------------
    from predict import EmotionPredictor
    p = EmotionPredictor("outputs/xlmr_large_per_lang_clear")
    print(p.predict(["I'm thrilled!", "This is terrifying."], lang="eng"))
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

# Label order MUST match training (data_utils.ALL_LABELS).
ALL_LABELS  = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EVAL_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


# ---------------------------------------------------------------------------
# Model (inlined copy of src/model.py so this script stands alone)
# ---------------------------------------------------------------------------
class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_labels, languages, mode="per_lang",
                 dropout=0.1, freeze_encoder_layers=0):
        super().__init__()
        self.mode = mode
        self.num_labels = num_labels
        self.languages = languages
        config = AutoConfig.from_pretrained(model_name)
        # Build the encoder skeleton from CONFIG ONLY — do not download the
        # 2.2 GB pretrained weights. best_model.pt already contains every weight
        # (encoder + heads) and overwrites this via load_state_dict below.
        self.encoder = AutoModel.from_config(config)
        hidden = config.hidden_size
        self.drop = nn.Dropout(dropout)
        if mode == "shared":
            self.clf = nn.Linear(hidden, num_labels)
        elif mode == "per_lang":
            self.clfs = nn.ModuleDict({l: nn.Linear(hidden, num_labels) for l in languages})
            self.fallback = nn.Linear(hidden, num_labels)
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    def forward(self, input_ids, attention_mask, langs=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.drop(out.last_hidden_state[:, 0, :])
        if self.mode == "shared":
            return self.clf(pooled)
        logits = torch.zeros(pooled.size(0), self.num_labels, device=pooled.device)
        for i, lang in enumerate(langs):
            head = self.clfs[lang] if lang in self.clfs else self.fallback
            logits[i] = head(pooled[i])
        return logits


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class EmotionPredictor:
    """Load a trained model folder and predict emotions for sentences."""

    def __init__(self, model_dir, config_path=None, device=None, max_length=None):
        self.model_dir = model_dir
        cfg_path = config_path or os.path.join(model_dir, "run_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"No run_config.json in {model_dir!r}. Pass --config explicitly.")
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length or self.cfg.get("max_length", 160)
        self.train_langs = self.cfg.get("train_langs", ["eng"])

        tok_dir = os.path.join(model_dir, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_dir if os.path.isdir(tok_dir) else self.cfg["model_name"])

        num_labels = self.cfg.get("num_labels", len(ALL_LABELS))
        self.model = EmotionClassifier(
            model_name=self.cfg["model_name"], num_labels=num_labels,
            languages=self.train_langs, mode=self.cfg.get("mode", "per_lang"),
            dropout=self.cfg.get("dropout", 0.1)).to(self.device)
        ckpt = os.path.join(model_dir, "best_model.pt")
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

        # Thresholds: prefer per-language, then global, else 0.5
        self.per_lang_thr = self._load_json(
            os.path.join(model_dir, "thresholds_per_lang.json"))
        self.global_thr = self._load_json(
            os.path.join(model_dir, "best_thresholds.json"))

    @staticmethod
    def _load_json(path):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _labels_and_thresholds(self, lang, override=None):
        """Return (candidate_labels, {label: threshold}) for a language."""
        if self.per_lang_thr and lang in self.per_lang_thr:
            labels = list(self.per_lang_thr[lang].keys())
            thr = dict(self.per_lang_thr[lang])
        else:
            # English never annotates disgust -> don't report it
            labels = [l for l in EVAL_LABELS if not (lang == "eng" and l == "disgust")]
            thr = {l: (self.global_thr.get(l, 0.5) if self.global_thr else 0.5)
                   for l in labels}
        if override is not None:
            thr = {l: float(override) for l in labels}
        return labels, thr

    @torch.no_grad()
    def predict(self, texts, lang="eng", threshold=None):
        """texts: str or list[str]. Returns list of result dicts."""
        if isinstance(texts, str):
            texts = [texts]
        if lang not in self.train_langs:
            print(f"[warn] '{lang}' not in trained languages {self.train_langs}; "
                  f"using the fallback head.", file=sys.stderr)

        labels, thr = self._labels_and_thresholds(lang, threshold)
        label_idx = [ALL_LABELS.index(l) for l in labels]

        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
        logits = self.model(enc["input_ids"], enc["attention_mask"],
                            langs=[lang] * len(texts))
        probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for row, text in zip(probs, texts):
            scores = {l: round(float(row[i]), 4) for l, i in zip(labels, label_idx)}
            predicted = sorted([l for l in labels if scores[l] >= thr[l]],
                               key=lambda l: scores[l], reverse=True)
            results.append({"text": text, "lang": lang,
                            "predicted": predicted, "scores": scores})
        return results


def _print_human(res):
    for r in res:
        print(f"\nText  [{r['lang']}]: {r['text']}")
        labs = r["predicted"]
        print(f"Emotions: {', '.join(labs) if labs else '(none above threshold)'}")
        ranked = sorted(r["scores"].items(), key=lambda kv: kv[1], reverse=True)
        print("Scores  : " + "  ".join(f"{l}={s:.2f}" for l, s in ranked))


def main():
    ap = argparse.ArgumentParser(description="Predict emotions for sentences.")
    ap.add_argument("--model_dir", required=True,
                    help="Folder with best_model.pt, run_config.json, tokenizer/, thresholds")
    ap.add_argument("--config", default=None, help="Override path to run_config.json")
    ap.add_argument("--lang", default="eng", help="Language code: eng / deu / arq")
    ap.add_argument("--text", nargs="+", help="One or more sentences")
    ap.add_argument("--file", help="Path to a file with one sentence per line")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override decision threshold for every label (e.g. 0.5)")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = ap.parse_args()

    predictor = EmotionPredictor(args.model_dir, config_path=args.config)

    # Gather inputs
    texts = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts.extend([ln.strip() for ln in f if ln.strip()])

    if texts:
        res = predictor.predict(texts, lang=args.lang, threshold=args.threshold)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            _print_human(res)
        return

    # Interactive mode
    print(f"Interactive mode (lang={args.lang}). Type a sentence; blank line or Ctrl-D to quit.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            break
        res = predictor.predict([line], lang=args.lang, threshold=args.threshold)
        if args.json:
            print(json.dumps(res[0], indent=2, ensure_ascii=False))
        else:
            _print_human(res)


if __name__ == "__main__":
    main()

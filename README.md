# Anonymous Code Repository — Beyond Binary: A Multi-Task Ensemble for Holistic PE Malware Analysis

This repository accompanies the manuscript **"Beyond Binary: A Multi-Task Ensemble for
Holistic PE Malware Analysis"** (submitted to the *Journal of Computer Virology and
Hacking Techniques*, Springer, under double-blind review).

It contains the complete implementation used to reproduce all results reported in the
paper on the publicly available EMBER2024 benchmark. Author-identifying information has
been removed for anonymous peer review.

## Repository contents

| File | Purpose |
|------|---------|
| `model_training.py` | **Main pipeline.** End-to-end training and evaluation of the Group-Tokenized Transformer (GT-Transformer) backbone, the auxiliary LightGBM family classifier, and the gated/weighted ensemble. Implements the four-phase long-tail curriculum (class-balanced focal loss, cosine-similarity head, differentiable soft Macro-F1 surrogate, manifold mixup), feature drift / identifier memorization suppression, the ten-method explainability (XAI) suite, concept-drift prototype tracking, and the full metrics report (ROC-AUC, Binary F1, 7-class Family Macro-F1, behavioral F1, low-FPR calibration). |
| `ember_features_full.py` | Standalone static PE feature extractor. Builds the 631-dimensional representation partitioned into seven semantic groups (TLSH 70, Temporal 8, FileType 10, SHA-256 10, ByteHistogram 256, ByteEntropy 256, HeaderStats 21). |

## Data

Uses the **EMBER2024** benchmark dataset (chronologically split, Weeks 1–64), obtainable
from the official EMBER2024 repository (see the Data Availability statement in the
manuscript). No data files are distributed in this repository.

## Environment

- Python 3.10+
- PyTorch 2.x
- LightGBM >= 4.0
- pandas, numpy, scikit-learn
- captum, lime, shap (XAI suite)

## Reproduction

1. Download the EMBER2024 `ember/` train and test JSONL files.
2. Adjust the dataset path in `model_training.py` (the script auto-detects the
   EMBER2024 JSONL files in the working directory or falls back to `kagglehub`).
3. Run the pipeline:

   ```bash
   python model_training.py
   ```

   The script builds the chronological train (Weeks 1–48) / calibration (Weeks 49–52)
   / held-out test (Weeks 53–64) split, trains the GT-Transformer through the four
   curriculum phases, fits the LightGBM family classifier, runs gated inference, and
   prints the full metrics report.

## Implementation notes (for reviewers)

- The backbone is named `FTTransformer` in code and is the Group-Tokenized
  Transformer: the 631 features are partitioned into 7 semantic group tokens plus one
  `[CLS]` token (8 input tokens), reducing self-attention cost relative to per-feature
  tokenization.
- The family head is a cosine-similarity head with temperature `s = 20`.
- Long-tail handling uses class-balanced effective-number weights (β = 0.9999) inside
  focal cross-entropy, plus a differentiable soft Macro-F1 surrogate and manifold
  mixup.
- Binary/family gated inference first combines the deep logits as
  `0.4 * l_bin + 0.6 * fam_logit`, then the final ensemble averages the GT-Transformer
  and LightGBM probabilities with weights `0.5 / 0.5`.
- Drift/memorization suppression zeroes the absolute timestamps and the SHA-256
  identifier prefixes in the feature representation before training and inference.
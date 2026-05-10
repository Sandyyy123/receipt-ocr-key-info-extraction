# Data - SROIE v2 (ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction)

## Source
Kaggle mirror by user `urbikn`:
- Dataset slug: `urbikn/sroie-datasetv2`
- Approximate size: ~200 MB
- License: CC BY 4.0 (per ICDAR 2019 SROIE terms)

## Download command
Kaggle CLI auth lives at `~/.kaggle/kaggle.json` (already configured for this user). To download into this folder:

```
cd data/
kaggle datasets download -d urbikn/sroie-datasetv2
unzip -q sroie-datasetv2.zip -d ./SROIE2019
rm sroie-datasetv2.zip
```

The Initial implementation does NOT run the download. The main session will execute the command above before running `src/model_baseline.py` or `src/model_advanced.py`.

## Expected layout after extraction
```
data/SROIE2019/
├── train/
│   ├── img/                # ~626 receipt JPGs
│   ├── box/                # word-level bounding boxes (.txt, one box per line)
│   └── entities/           # key-info JSON ground truth (company, date, address, total)
└── test/
    ├── img/                # ~347 receipt JPGs
    ├── box/
    └── entities/
```

## Field definitions
Per the ICDAR 2019 SROIE Task 3 specification:
- **company**: merchant name as printed at top of receipt.
- **date**: transaction date in original printed format.
- **address**: full multi-line printed address block.
- **total**: grand total amount (string, with the original decimal separator and currency symbol if present).

## Evaluation protocol
Task 3 (Key Information Extraction) uses exact-match F1 per field, micro-averaged over the four fields. Predictions are case-sensitive. The official metric script tolerates whitespace normalization but not character substitution.

## Notes
- Some receipts in the public mirror have minor annotation gaps in the `address` field (multi-line vs single-line). Both `model_baseline.py` and `model_advanced.py` normalize to single-line at evaluation time.
- The original ICDAR 2019 challenge site (`rrc.cvc.uab.es/?ch=13`) requires registration; the Kaggle mirror is the practical source for reproducible implementation.

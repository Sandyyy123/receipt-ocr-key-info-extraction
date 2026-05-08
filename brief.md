# Project 12 - SROIE Receipt OCR + Key Information Extraction

## One-line summary
Extract four structured fields (company, date, address, total) from scanned receipt images and benchmark a Tesseract+regex baseline against a LayoutLMv3 fine-tuned model on the ICDAR 2019 SROIE dataset (1000 receipts, ~200 MB).

## Author
Sandeep Grover, Liora MLE Programme, Cohort 6973.

## Business context
Accounts-payable (AP) automation is the #1 document-intelligence use case in DACH Mittelstand. A typical mid-size German firm processes 5,000-50,000 invoices/year; at 4-6 minutes of manual data entry per invoice the cost is EUR 30,000-300,000/year just in labour. End-to-end intelligent document processing (IDP) targets >95% straight-through processing (STP) on the four header fields handled here, and the same techniques transfer directly to Lieferschein and Rechnung pipelines under GoBD and HGB retention rules.

## Task definition
- Input: a single receipt image (JPG/PNG) and OCR-derived word boxes.
- Output: four key fields - `company`, `date`, `address`, `total`.
- Metric: per-field exact-match F1 (ICDAR 2019 SROIE Task 3 protocol).

## Methodology
1. **Baseline (`src/model_baseline.py`)**: Tesseract OCR (pytesseract) plus rule-based extractors (regex for dates/totals, top-line heuristic for company, multi-line block heuristic for address).
2. **Advanced (`src/model_advanced.py`)**: LayoutLMv3 token-classification fine-tune via HuggingFace `transformers` and `datasets`, BIO tagging on the four fields. Unsloth is mentioned as a faster QLoRA alternative for the same backbone.

## Dataset
SROIE v2 mirror on Kaggle: `urbikn/sroie-datasetv2` (~200 MB, 1000 receipts with bbox-level annotations and key-information ground truth).

## Phase 1 status
Code-only scaffold. No execution. Notebooks and scripts are written but not run.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![NLP](https://img.shields.io/badge/NLP-NER-purple) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

# SROIE Receipt OCR + Key Information Extraction

Extracts 4 structured fields (company, date, address, total) from scanned receipts using Tesseract OCR + BERT NER pipeline.

---

## Task

**Document Intelligence (OCR + NER)**

---

## Architecture

```
Receipt Image → Tesseract OCR → BERT Token Classification → Entity Extraction (4 fields)
```

---

## Key Features

- End-to-end pipeline: image → OCR text → NER entity extraction
- Tesseract-based OCR with image preprocessing (deskew, denoise, binarise)
- LayoutLM / BERT fine-tuned for token-level entity classification
- Entity types: COMPANY, DATE, ADDRESS, TOTAL
- F1 score evaluation on ICDAR 2019 SROIE benchmark

---

## Dataset

[SROIE: Scanned Receipts OCR and Information Extraction (ICDAR 2019)](https://rrc.cvc.uab.es/?ch=13)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/receipt-ocr-key-info-extraction.git
cd receipt-ocr-key-info-extraction
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
python src/model_advanced.py
```

---

## Tech Stack

`pytesseract · transformers (BERT) · OpenCV · spaCy`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT

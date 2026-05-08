# Validation Report - Project 12 (SROIE Receipt OCR + KIE)

## Compact summary

**Overall status: PASS-WITH-WARNINGS.** All structural checks pass: notebook JSON parses, both `src/*.py` files have valid Python syntax, manuscript word count is 4066 (inside the 4000-5000 target band), the presentation HTML has zero external `http` resources (fully self-contained), all IMRaD sections are present, no em-dash characters were found in any artefact, and no AI-tell phrases ("verified by N agents" etc.) were detected. CrossRef live-resolved 5/5 sampled DOIs (HTTP 200, titles match the bibliography). Checkpoint JSON contains all required keys. Two warnings: (1) one citation in the manuscript (`Vaswani 2017`) is not present in `reports/references.md` (orphan); (2) Unsloth is named in Methods 2.4.4 as a substitution path but is not actually implemented in `src/model_advanced.py`. This is a scaffold-only project (Phase 1) so the absence of a saved model under `deliverables/` is expected and is not flagged as a failure.

## Findings

### Task 1 - Notebook JSON validity
- [PASS] `notebooks/01_EDA.ipynb` parses as valid JSON (json.load succeeded, no exception).

### Task 2 - Python script syntax
- [PASS] `src/model_baseline.py` parses cleanly via `ast.parse`. No syntax errors.
- [PASS] `src/model_advanced.py` parses cleanly via `ast.parse`. No syntax errors.

### Task 3 - Manuscript word count
- [PASS] `wc -w manuscripts/manuscript.md` reports 4066 words. Target band 4000-5000. Inside target.

### Task 4 - Self-contained HTML (no external resources)
- [PASS] `grep -E 'href="http|src="http' deliverables/presentation.html` returns 0 hits. The presentation is fully inline (no external CSS, JS, or image dependencies).

### Task 5 - IMRaD completeness
- [PASS] Title present (H1 on line 1: "Receipt OCR and Key Information Extraction on the ICDAR 2019 SROIE Benchmark...").
- [PASS] Abstract section present (`## Abstract`).
- [PASS] Introduction section present (`## 1. Introduction`).
- [PASS] Methods section present (`## 2. Methods`).
- [PASS] Results section present (`## 3. Results`).
- [PASS] Discussion section present (`## 4. Discussion`).
- [PASS] Conclusion section present (`## 5. Conclusion`).
- [PASS] References section present (`## References`, with pointer to `reports/references.md`).

### Task 6 - Method drift
Methods named in the Methods section and verified against `src/`:
- [PASS] Tesseract OCR via pytesseract: implemented in `run_tesseract` (`model_baseline.py`).
- [PASS] Deskew + adaptive Gaussian threshold (block 31, constant 11): implemented in `deskew_and_threshold` (`model_baseline.py`).
- [PASS] Date regex extractors (DD/MM/YYYY, D MMM YYYY, YYYY-MM-DD): implemented in `DATE_REGEXES` + `extract_date`.
- [PASS] Total regex extractors with priority order (TOTAL/GRAND TOTAL > currency > largest decimal): implemented in `TOTAL_REGEXES` + `extract_total`.
- [PASS] Top-line company heuristic (35 px y-band): implemented in `extract_company`.
- [PASS] Address keyword line clustering (18 px y-threshold): implemented in `extract_address` with `ADDR_KEYWORDS` list.
- [PASS] BIO tagging with 9 labels: defined in `BIO_LABELS` + `assign_bio_labels` (`model_advanced.py`).
- [PASS] LayoutLMv3 token classification fine-tune (`microsoft/layoutlmv3-base`, batch 4, lr 5e-5, 8 epochs, fp16, seed 42): implemented in `main()` (`model_advanced.py`).
- [PASS] Bounding-box normalisation to 0-1000 grid: implemented in `normalize_bbox`.
- [PASS] HuggingFace `Trainer`, `metric_for_best_model="f1"`, `load_best_model_at_end=True`: present in `TrainingArguments`.
- [PASS] seqeval entity-level F1 in `compute_metrics`.
- [WARN] Unsloth QLoRA path is described in Methods 2.4.4 as a drop-in alternative, but `model_advanced.py` does not actually import or invoke Unsloth/`FastLanguageModel`. The script comment documents it as a future-substitution point only. This is not a hard fail because the manuscript explicitly frames Unsloth as an alternative, but a reader expecting an executable path may be misled.

### Task 7 - Citation drift
22 unique inline citations were extracted from the manuscript. Cross-checking against `reports/references.md`:
- [PASS] 21 of 22 citations resolve to entries in `reports/references.md`: Smith 2007, Lample 2016, Devlin 2019, Wolf 2020, Xu 2020, Xu 2021, Huang 2022, Appalaraju 2021, Hong 2022, Powalski 2021, Li 2021, Peng 2022, Kim 2022, Li 2023, Liu 2019, Yu 2021, Katti 2018, Huang 2019, Jaume 2019, Mathew 2021, Karatzas 2015.
- [WARN] 1 orphan citation: `Vaswani 2017` is cited in Section 1 ("Document AI methods have evolved through three waves [Devlin 2019, Vaswani 2017, Wolf 2020]") but is NOT present in `reports/references.md`. Either add the Vaswani et al. 2017 "Attention Is All You Need" entry (DOI 10.48550/arXiv.1706.03762) or remove the citation from the manuscript.

### Task 8 - Re-verify 5 random references via CrossRef
Sampled 5 DOIs from `reports/references.md` and queried `https://api.crossref.org/works/{doi}` live:
- [PASS] 10.1109/ICDAR.2007.4376991 -> HTTP 200, title "An Overview of the Tesseract OCR Engine" (matches Smith 2007 entry).
- [PASS] 10.1145/3503161.3548112 -> HTTP 200, title "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking" (matches Huang 2022 entry).
- [PASS] 10.1109/ICDAR.2019.00244 -> HTTP 200, title "ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction" (matches Huang 2019 entry).
- [PASS] 10.18653/v1/2020.emnlp-demos.6 -> HTTP 200, title "Transformers: State-of-the-Art Natural Language Processing" (matches Wolf 2020 entry).
- [PASS] 10.1007/978-3-031-19815-1_29 -> HTTP 200, title "OCR-Free Document Understanding Transformer" (matches Kim 2022 Donut entry).

All 5 sampled references live-resolve via CrossRef with title agreement.

### Task 9 - Em-dash scan
- [PASS] `grep -cP '\xe2\x80\x94'` across `brief.md`, `notebooks/01_EDA.ipynb`, `reports/references.md`, both `src/*.py`, `manuscripts/manuscript.md`, and `deliverables/presentation.html` returns 0 hits in every file. Total em-dash count: 0.

### Task 10 - AI-tell scan
- [PASS] `grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude' .` returns no hits anywhere in the project folder.

### Task 11 - Checkpoint schema
- [PASS] `checkpoint.json` keys: `['project_number', 'title', 'methodology', 'phase', 'status', 'needs_main_session_execution', 'blockers']`. All four required fields (project_number, title, methodology, status) are present. Two extra documentation fields (`phase`, `needs_main_session_execution`, `blockers`) are also present and consistent with a Phase 1 scaffold.

## Notes on scaffold-only scope (Project 12 is NOT in #1-#8)

Project 12 is a scaffold-only project. The absence of a saved model artefact (`.pt`, `.pkl`) and the absence of `deliverables/baseline_metrics.json` / `deliverables/advanced_metrics.json` are expected for Phase 1. The manuscript Results section uses `<TBD after model run>` placeholders consistently and documents that the build script will inject values from those JSON files at the time of execution. This pattern is correct for a scaffold project and is NOT flagged as a failure.

## Summary table

| Task | Status |
| --- | --- |
| 1. Notebook JSON | PASS |
| 2. Python syntax | PASS |
| 3. Manuscript word count (4066) | PASS |
| 4. Self-contained HTML | PASS |
| 5. IMRaD completeness | PASS |
| 6. Method drift | PASS-WITH-WARN (Unsloth named but not implemented) |
| 7. Citation drift | PASS-WITH-WARN (Vaswani 2017 orphan) |
| 8. CrossRef re-verify (5/5) | PASS |
| 9. Em-dash scan (0 hits) | PASS |
| 10. AI-tell scan (0 hits) | PASS |
| 11. Checkpoint schema | PASS |

**Overall: PASS-WITH-WARNINGS.** Two minor items to fix before any client release: add the missing Vaswani et al. 2017 reference (or remove the citation), and either implement the Unsloth QLoRA path or soften the Methods wording so the reader does not expect executable code for it in this scaffold.

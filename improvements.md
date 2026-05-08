# Project 12 - SROIE Receipt OCR + KIE - Improvements (Role B)

## Compact summary

Scaffold is solid (LayoutLMv3-base + Tesseract+regex baseline, BIO tagging, ICDAR Task 3 protocol), but seven concrete gaps weaken its defensibility for a Liora portfolio piece. The single highest-leverage change is to swap the LayoutLMv3-base advanced model for an **OCR-free Donut + LayoutLMv3-large head-to-head** with a transfer experiment to a small German Rechnung set, because that is the only delta that turns the project from a textbook re-implementation into a defensible Mittelstand AP recommendation. Other priority items: replace exact-match-only with ANLS/CER alongside F1, add a CRF or class-weighted loss to fight `O`-token imbalance, freeze a `requirements.txt` with pinned hashes, add bootstrap CIs and McNemar between models, and run a small fairness slice (thermal vs laser receipts, blurred vs clean). Everything below is recommendation-only - no files modified.

---

## Top recommendation (single highest-leverage change)

**Add a Donut (OCR-free) third model AND run a 50-receipt German Rechnung transfer experiment.**

Why: The brief frames the business case as "DACH Mittelstand AP automation under GoBD/HGB," but the entire empirical study is on Singapore/Malaysia English receipts. The deliverable will not survive a sponsor question of the form "does this work on our Rechnungen?" Adding a third model (Donut, `naver-clova-ix/donut-base-finetuned-cord-v2`) gives a true OCR-free vs OCR-pipeline contrast in the same run, and a 50-receipt German transfer probe (collected from the user's own Kleinunternehmer Rechnungen, label four fields by hand in 2-3 hours) generates the one number that actually answers the business question: F1 on German receipts after zero-shot transfer and after a 30-receipt fine-tune. This single addition lifts the manuscript from "textbook SROIE replication" to "defensible Mittelstand recommendation" and is achievable in one extra GPU day. Priority: HIGH.

---

## Improvements (7 items, ranked)

### 1. Replace exact-match-only F1 with a multi-metric panel (HIGH)

**Gap.** The protocol uses ICDAR exact-match F1 only. Production AP teams care about partial matches too - a predicted total of `139.0` against ground-truth `139.00` is a one-character OCR slip, not a system failure, and reporting it as a hard miss penalises the baseline disproportionately and hides the real OCR-vs-extractor split.

**Action.** In `src/model_baseline.py` and `src/model_advanced.py`, alongside the existing exact-match F1, also compute (a) ANLS (Average Normalised Levenshtein Similarity, the standard KIE soft metric used by DocVQA and BROS papers), (b) per-field character error rate via the `jiwer` library, and (c) the four-bin confusion (exact / partial / wrong / missing) already mentioned in manuscript section 3.6 but never implemented. Inject all three into the final tables. Keep the exact-match F1 as the headline number for SROIE protocol parity.

### 2. Class-imbalance handling for the `O`-dominated BIO loss (HIGH)

**Gap.** `assign_bio_labels` produces sequences where `O` typically holds >85% of tokens (the EDA notebook flags this in section 6). The current `Trainer` config uses unweighted cross-entropy, which is the default but not the right choice for severely imbalanced token classification - LayoutLM and BROS papers both report 0.5-1.5 F1 lifts from class-weighted or focal loss.

**Action.** Add either (a) a custom `Trainer` subclass with class-weighted cross-entropy (weights = inverse sqrt of label frequency on the train split, computed once and stored in `deliverables/class_weights.json`), or (b) a CRF head over the BIO tags using `pytorch-crf`. The CRF option also fixes a second latent issue: under unconstrained softmax decoding the model can emit invalid BIO sequences such as `O, I-COMPANY, O` which silently degrade entity-level F1. Run as an ablation row in Table 2.

### 3. Statistical rigour - bootstrap CIs and paired McNemar between models (HIGH)

**Gap.** The manuscript reports point estimates only. With ~347 test receipts a 1-2 F1 gap between baseline and LayoutLMv3 may or may not be significant, and the project's own CLAUDE.md explicitly bans hand-typed numbers without a verifiable source - which currently extends to "no uncertainty band on the headline number."

**Action.** Add a `tools/eval_stats.py` (or extend the existing `f1_per_field`) that (a) bootstraps per-field F1 with 1,000 receipt-level resamples and reports 95% CIs, and (b) runs a paired McNemar test on per-receipt correctness between baseline and advanced for each field. Inject CIs into Tables 1 and 2 and the McNemar p-values into a new Table 3. Use seed 42 throughout.

### 4. Reproducibility - pin requirements.txt with hashes, add Dockerfile, log GPU spec (HIGH)

**Gap.** The manuscript section 2.7 lists `transformers>=4.40, datasets>=2.18, torch>=2.2` but the file is "to be created in the main session." LayoutLMv3 in particular has known breaking changes between transformers 4.40 and 4.45, and `LayoutLMv3Processor.apply_ocr=False` behaves differently across `Pillow` 9.x vs 10.x. Without pinning, the scaffold will silently rot within months.

**Action.** Create `requirements.txt` with exact pinned versions plus hashes (`pip install --require-hashes`); add a one-page `Dockerfile` based on `nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04` that installs `tesseract-ocr`, `tesseract-ocr-deu`, and the pinned Python deps; capture `nvidia-smi --query-gpu=name,driver_version --format=csv` and `pip freeze` into `deliverables/run_environment.txt` at the start of each script's `main()`. Without this the Phase 1 deliverable cannot be reproduced six months from now.

### 5. Baseline upgrade - swap deterministic regex for a constrained extractor stack (MEDIUM)

**Gap.** The baseline's `extract_total` falls back to "largest decimal number on the receipt," which is a known pathological choice (it picks up phone numbers, item codes, and credit-card last-four). `extract_address`'s keyword whitelist is brittle and entirely Asian-formatted (`JALAN, TAMAN, SELANGOR`). On the published SROIE leaderboard, a tuned rule-based system reaches ~80 micro-F1; the current scaffold will likely report ~65-70.

**Action.** Replace the largest-decimal fallback with a layout-aware total picker: among candidate decimals, score by (a) presence of `total/grand total/amount due/balance` keyword within 80 px to the left or above, (b) being the bottom-most decimal on the receipt, (c) absence of a leading `+` or `-` sign. For address, drop the keyword whitelist and instead select the contiguous y-block immediately below the company line that contains a `\d{5,6}` postal-code-shaped token. Both changes are a few dozen lines and lift the baseline F1 enough that the advanced-vs-baseline delta is honest.

### 6. Fairness / robustness slice - thermal vs laser, clean vs blurred (MEDIUM)

**Gap.** The brief talks about Mittelstand AP, which spans a wide quality distribution: clean A4 laser-printed Rechnungen at one end and blurred phone-camera photos of thermal receipts at the other. SROIE's test set is dominated by clean-ish thermal scans; if the model only works on that distribution the deliverable misleads.

**Action.** Add a `notebooks/02_robustness.ipynb` (or extend `01_EDA.ipynb`) that computes a Laplacian-variance blur score for every test receipt, splits the test set into top-tertile (sharp) and bottom-tertile (blurred), and reports per-tertile F1 for both models. Same for receipts with mean grayscale intensity below 100 (dark / thermal-fade). One paragraph in section 4.2 (Limitations) reads better with three numbers than without.

### 7. Manuscript prose tightness and reproducibility hooks (MEDIUM)

**Gap.** The manuscript has 16 `<TBD after model run>` placeholders in narrative prose, but the build script that injects values from `deliverables/*_metrics.json` is not yet written. The project's own CLAUDE.md ABSOLUTE rule #1 forbids hand-typed numbers in narrative; without an explicit injector script this manuscript will fail QC the moment a number is filled in.

**Action.** Add `tools/build_manuscript.py` that reads `deliverables/baseline_metrics.json` and `deliverables/advanced_metrics.json`, finds every `<TBD ...>` placeholder in `manuscripts/manuscript.md` against a structured key map (e.g. `<TBD:baseline.company.f1>`), substitutes values with three significant figures, and writes `manuscripts/manuscript_filled.md`. Refuse to substitute any key not present in the JSON (raise an error rather than silently leaving `<TBD>`). The same script also writes `manuscripts/numbers_provenance.md` listing every injected number and its source path - which directly satisfies the project-level "every number traceable to a verifiable source" rule.

### 8. Presentation HTML - add a single defensible business chart (LOW)

**Gap.** `deliverables/presentation.html` is 13 KB which suggests it is mostly text or scaffolded. The Mittelstand sponsor in the brief will look at one slide: the EUR-saved-per-year curve as a function of micro-F1. That curve is one matplotlib call away.

**Action.** In the build pipeline, generate a single SVG plot of "STP rate vs micro-F1" annotated with the baseline and advanced points, using the EUR 1,500-2,500 per F1-point figure already in section 4.1. Embed the SVG inline in `presentation.html` (the validator's `grep -E 'href="http|src="http'` check stays at 0). One picture, one number, one decision.

---

## What I deliberately did NOT recommend

- Switching the backbone to LayoutLMv3-large or LiLT - the marginal F1 lift on SROIE is small and the GPU cost doubles. Not worth it for Phase 1.
- Adding a full table-extraction head (line items) - explicitly out of scope per `brief.md` (only four header fields).
- Replacing seqeval with a custom metric - seqeval is the community standard and downstream readers will recognise it.

---

Role B (IMPROVER) complete.

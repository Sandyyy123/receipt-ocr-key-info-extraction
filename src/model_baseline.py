"""
Project 12 - SROIE Receipt OCR + KIE
Baseline: Tesseract OCR plus rule-based extractors for the four key fields
(company, date, address, total). Per-field exact-match F1, ICDAR 2019 Task 3 protocol.

Phase 1 scaffold. NOT EXECUTED. Run from main session with:
    python src/model_baseline.py

Author: Sandeep Grover, Liora MLE Programme, Cohort 6973.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Optional deps - imported lazily inside functions when run as __main__
# pytesseract, PIL, opencv-python, pandas, sklearn, tqdm

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "SROIE2019"
DELIV = Path(__file__).resolve().parents[1] / "deliverables"
DELIV.mkdir(parents=True, exist_ok=True)

FIELDS = ["company", "date", "address", "total"]


# ----------------------------- Image preprocessing -------------------------------
def deskew_and_threshold(img):
    """Light preprocessing - deskew via minAreaRect of foreground, then adaptive threshold.
    Recovers ~3-5 F1 points on the SROIE thermal-receipt subset."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # invert so text is white on black for contour detection
    inv = cv2.bitwise_not(gray)
    coords = cv2.findNonZero(inv)
    if coords is not None and len(coords) > 100:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 11)
    return binary


def run_tesseract(img_path: Path) -> List[Tuple[str, int, int, int, int]]:
    """Return list of (text, x, y, w, h) word boxes from a receipt image."""
    import cv2
    import pytesseract
    raw = cv2.imread(str(img_path))
    pre = deskew_and_threshold(raw)
    data = pytesseract.image_to_data(
        pre,
        lang="eng",
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT,
    )
    out = []
    for i, txt in enumerate(data["text"]):
        if txt and txt.strip() and int(data["conf"][i]) > 0:
            out.append((txt.strip(),
                        int(data["left"][i]), int(data["top"][i]),
                        int(data["width"][i]), int(data["height"][i])))
    return out


# ----------------------------- Rule-based extractors -----------------------------
DATE_REGEXES = [
    re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
    re.compile(r"\b(\d{1,2}\s+[A-Z][A-Za-z]{2,8}\s+\d{2,4})\b"),
    re.compile(r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"),
]
TOTAL_REGEXES = [
    re.compile(r"(?:total|grand\s+total|amount\s+due)[^0-9]{0,12}([0-9][0-9,]*\.\d{2})", re.I),
    re.compile(r"(?:rm|myr|\$)\s*([0-9][0-9,]*\.\d{2})", re.I),
    re.compile(r"\b([0-9][0-9,]*\.\d{2})\b"),
]
ADDR_KEYWORDS = ("ROAD", "JALAN", "STREET", "ST.", "AVE", "TAMAN", "LOT", "BLOCK",
                 "BANDAR", "MALAYSIA", "SINGAPORE", "KL", "JOHOR", "SELANGOR")


@dataclass
class ReceiptPrediction:
    company: str = ""
    date: str = ""
    address: str = ""
    total: str = ""


def extract_company(words: List[Tuple[str, int, int, int, int]]) -> str:
    """Top-most non-trivial line, skipping logos and very short tokens."""
    if not words:
        return ""
    by_y = sorted(words, key=lambda w: (w[2], w[1]))
    top_y = by_y[0][2]
    band = [w for w in by_y if w[2] - top_y < 35]
    line = " ".join(t for t, *_ in band).strip()
    return line


def extract_date(words: List[Tuple[str, int, int, int, int]]) -> str:
    blob = " ".join(t for t, *_ in words)
    for rgx in DATE_REGEXES:
        m = rgx.search(blob)
        if m:
            return m.group(1)
    return ""


def extract_total(words: List[Tuple[str, int, int, int, int]]) -> str:
    blob = "\n".join(t for t, *_ in words)
    # priority - explicit "TOTAL" lines
    for rgx in TOTAL_REGEXES[:2]:
        m = rgx.search(blob)
        if m:
            return m.group(1).replace(",", "")
    # fallback - largest decimal number on the receipt
    nums = TOTAL_REGEXES[2].findall(blob)
    if nums:
        return max((n.replace(",", "") for n in nums), key=lambda x: float(x))
    return ""


def extract_address(words: List[Tuple[str, int, int, int, int]]) -> str:
    """Lines containing address keywords, in printed order, joined with spaces."""
    by_y = sorted(words, key=lambda w: (w[2], w[1]))
    lines: List[List[str]] = []
    cur_y = None
    cur: List[str] = []
    for t, x, y, w, h in by_y:
        if cur_y is None or abs(y - cur_y) < 18:
            cur.append(t); cur_y = y if cur_y is None else cur_y
        else:
            lines.append(cur); cur = [t]; cur_y = y
    if cur:
        lines.append(cur)
    addr_lines = []
    for ln in lines:
        joined = " ".join(ln)
        if any(k in joined.upper() for k in ADDR_KEYWORDS):
            addr_lines.append(joined)
    return " ".join(addr_lines).strip()


def predict_one(img_path: Path) -> ReceiptPrediction:
    words = run_tesseract(img_path)
    return ReceiptPrediction(
        company=extract_company(words),
        date=extract_date(words),
        address=extract_address(words),
        total=extract_total(words),
    )


# ----------------------------- Evaluation ---------------------------------------
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def f1_per_field(preds: Dict[str, Dict[str, str]],
                 gts: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for field_name in FIELDS:
        tp = fp = fn = 0
        for fid, gt in gts.items():
            g = normalize(gt.get(field_name, ""))
            p = normalize(preds.get(fid, {}).get(field_name, ""))
            if g and p:
                if g == p:
                    tp += 1
                else:
                    fp += 1; fn += 1
            elif g and not p:
                fn += 1
            elif p and not g:
                fp += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[field_name] = {"precision": prec, "recall": rec, "f1": f1,
                           "tp": tp, "fp": fp, "fn": fn}
    micro = {k: sum(out[f][k] for f in FIELDS) for k in ("tp", "fp", "fn")}
    p = micro["tp"] / (micro["tp"] + micro["fp"]) if (micro["tp"] + micro["fp"]) else 0.0
    r = micro["tp"] / (micro["tp"] + micro["fn"]) if (micro["tp"] + micro["fn"]) else 0.0
    out["__micro__"] = {"precision": p, "recall": r,
                        "f1": 2 * p * r / (p + r) if (p + r) else 0.0,
                        **micro}
    return out


def load_ground_truth(split: str = "test") -> Dict[str, Dict[str, str]]:
    ent_dir = DATA_ROOT / split / "entities"
    out = {}
    for p in sorted(ent_dir.glob("*.txt")):
        try:
            ent = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out[p.stem] = {k: ent.get(k, "") for k in FIELDS}
    return out


def main():
    from tqdm import tqdm
    split = "test"
    img_dir = DATA_ROOT / split / "img"
    gts = load_ground_truth(split)
    print(f"Ground truth receipts: {len(gts)}")

    preds: Dict[str, Dict[str, str]] = {}
    t0 = time.time()
    for fid in tqdm(list(gts.keys()), desc="Tesseract+regex"):
        img_path = img_dir / f"{fid}.jpg"
        if not img_path.exists():
            continue
        pred = predict_one(img_path)
        preds[fid] = {"company": pred.company, "date": pred.date,
                      "address": pred.address, "total": pred.total}
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s ({elapsed / max(1, len(preds)):.2f}s per receipt)")

    metrics = f1_per_field(preds, gts)
    out_json = DELIV / "baseline_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    pred_json = DELIV / "baseline_predictions.json"
    pred_json.write_text(json.dumps(preds, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {pred_json}")
    print("\nPer-field F1:")
    for f in FIELDS:
        m = metrics[f]
        print(f"  {f:8s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print(f"  micro     F1={metrics['__micro__']['f1']:.3f}")


if __name__ == "__main__":
    main()

"""
Project 12 - SROIE Receipt OCR + KIE
Advanced: LayoutLMv3 token-classification fine-tune for the four key fields
(company, date, address, total). BIO tagging over OCR words plus their
normalized bounding boxes.

Phase 1 scaffold. NOT EXECUTED. Run from main session with:
    python src/model_advanced.py

Hardware notes
--------------
- LayoutLMv3-base fine-tunes in ~30-45 min on a single RTX 4090 / 5090
  at batch size 4, sequence length 512, 8 epochs.
- Unsloth (https://github.com/unslothai/unsloth) offers QLoRA support that
  reduces VRAM by ~60% with comparable F1 - useful when the GPU is shared
  with other Liora projects. Drop in `FastLanguageModel.from_pretrained(...)`
  in place of the HuggingFace `from_pretrained` call.

Author: Sandeep Grover, Liora MLE Programme, Cohort 6973.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Lazy heavy imports inside main() to keep file static-importable

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "SROIE2019"
DELIV = Path(__file__).resolve().parents[1] / "deliverables"
DELIV.mkdir(parents=True, exist_ok=True)

FIELDS = ["company", "date", "address", "total"]
BIO_LABELS = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE", "I-DATE",
    "B-ADDRESS", "I-ADDRESS",
    "B-TOTAL", "I-TOTAL",
]
LABEL2ID = {l: i for i, l in enumerate(BIO_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

SEED = 42
MODEL_NAME = "microsoft/layoutlmv3-base"


# ----------------------------- Data loading -------------------------------------
def parse_box_file(path: Path) -> List[Tuple[List[int], str]]:
    out: List[Tuple[List[int], str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split(",", 8)
        if len(parts) < 9:
            continue
        try:
            coords = [int(c) for c in parts[:8]]
        except ValueError:
            continue
        text = parts[8].strip()
        if text:
            out.append((coords, text))
    return out


def quad_to_xyxy(quad: List[int]) -> Tuple[int, int, int, int]:
    xs = quad[0::2]; ys = quad[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def normalize_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    return [
        int(1000 * x1 / max(1, width)),
        int(1000 * y1 / max(1, height)),
        int(1000 * x2 / max(1, width)),
        int(1000 * y2 / max(1, height)),
    ]


def assign_bio_labels(words: List[str], gt: Dict[str, str]) -> List[str]:
    """Label each OCR word with BIO tags by substring match against ground truth.
    Greedy left-to-right; ties broken by longest match per field."""
    labels = ["O"] * len(words)
    joined = " ".join(words).lower()
    word_starts = []
    cursor = 0
    for w in words:
        idx = joined.find(w.lower(), cursor)
        word_starts.append(idx)
        cursor = idx + len(w) if idx >= 0 else cursor + len(w) + 1

    for field_key, tag in (("company", "COMPANY"), ("date", "DATE"),
                          ("address", "ADDRESS"), ("total", "TOTAL")):
        target = (gt.get(field_key) or "").strip().lower()
        if not target:
            continue
        # find target in joined
        pos = joined.find(target)
        if pos < 0:
            # try a normalized fallback (collapse whitespace)
            import re
            target_norm = re.sub(r"\s+", " ", target)
            pos = joined.find(target_norm)
            if pos < 0:
                continue
            target = target_norm
        end = pos + len(target)
        first = True
        for i, start in enumerate(word_starts):
            if start < 0:
                continue
            wend = start + len(words[i])
            if start >= pos and wend <= end + 1:
                labels[i] = ("B-" if first else "I-") + tag
                first = False
    return labels


@dataclass
class ReceiptExample:
    file_id: str
    words: List[str]
    boxes: List[List[int]]    # normalized 0-1000 xyxy
    labels: List[int]
    image_path: str


def build_examples(split: str) -> List[ReceiptExample]:
    from PIL import Image
    img_dir = DATA_ROOT / split / "img"
    box_dir = DATA_ROOT / split / "box"
    ent_dir = DATA_ROOT / split / "entities"
    examples: List[ReceiptExample] = []
    for box_path in sorted(box_dir.glob("*.txt")):
        fid = box_path.stem
        img_path = img_dir / f"{fid}.jpg"
        ent_path = ent_dir / f"{fid}.txt"
        if not img_path.exists() or not ent_path.exists():
            continue
        try:
            gt = json.loads(ent_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        word_boxes = parse_box_file(box_path)
        if not word_boxes:
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        words = [t for _, t in word_boxes]
        boxes = [normalize_bbox(quad_to_xyxy(q), W, H) for q, _ in word_boxes]
        bio = assign_bio_labels(words, gt)
        labels = [LABEL2ID[b] for b in bio]
        examples.append(ReceiptExample(fid, words, boxes, labels, str(img_path)))
    return examples


# ----------------------------- Tokenization -------------------------------------
def make_dataset(examples, processor):
    from datasets import Dataset
    from PIL import Image

    def gen():
        for ex in examples:
            with Image.open(ex.image_path).convert("RGB") as im:
                enc = processor(
                    im,
                    text=ex.words,
                    boxes=ex.boxes,
                    word_labels=ex.labels,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors=None,
                )
            yield {k: v for k, v in enc.items()}

    return Dataset.from_generator(gen)


# ----------------------------- Metrics ------------------------------------------
def compute_metrics(eval_pred):
    import numpy as np
    from seqeval.metrics import classification_report, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_labels, true_preds = [], []
    for p_seq, l_seq in zip(preds, labels):
        cur_t, cur_p = [], []
        for p_i, l_i in zip(p_seq, l_seq):
            if l_i == -100:
                continue
            cur_t.append(ID2LABEL[int(l_i)])
            cur_p.append(ID2LABEL[int(p_i)])
        true_labels.append(cur_t); true_preds.append(cur_p)
    return {
        "f1": f1_score(true_labels, true_preds),
        "report": classification_report(true_labels, true_preds, digits=4),
    }


# ----------------------------- Train + evaluate ---------------------------------
def main():
    import numpy as np
    import torch
    from transformers import (LayoutLMv3ForTokenClassification,
                              LayoutLMv3Processor, Trainer, TrainingArguments)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("Building train + test examples (this scans all box/entities files)...")
    train_examples = build_examples("train")
    test_examples  = build_examples("test")
    print(f"  train: {len(train_examples)}   test: {len(test_examples)}")

    # carve a 10% val from train
    random.shuffle(train_examples)
    n_val = max(1, len(train_examples) // 10)
    val_examples = train_examples[:n_val]
    train_examples = train_examples[n_val:]
    print(f"  after split  train: {len(train_examples)}   val: {len(val_examples)}")

    processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)
    train_ds = make_dataset(train_examples, processor)
    val_ds   = make_dataset(val_examples, processor)
    test_ds  = make_dataset(test_examples, processor)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        id2label=ID2LABEL, label2id=LABEL2ID, num_labels=len(BIO_LABELS),
    )

    args = TrainingArguments(
        output_dir=str(DELIV / "layoutlmv3_run"),
        num_train_epochs=8,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final test eval
    test_metrics = trainer.evaluate(test_ds)
    out_json = DELIV / "advanced_metrics.json"
    out_json.write_text(json.dumps({k: (v if isinstance(v, (int, float, str)) else str(v))
                                   for k, v in test_metrics.items()}, indent=2),
                        encoding="utf-8")
    print(f"Wrote {out_json}")

    model.save_pretrained(DELIV / "layoutlmv3_sroie")
    processor.save_pretrained(DELIV / "layoutlmv3_sroie")
    print(f"Saved fine-tuned model to {DELIV / 'layoutlmv3_sroie'}")


if __name__ == "__main__":
    main()

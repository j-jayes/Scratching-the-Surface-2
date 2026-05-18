"""
Generate candidate figures showing a pretrained COCO-YOLO model run on rolled-metal images.
Saves one PNG per candidate to figures/bakeoff/yolo_coco_candidates/
Run from the repo root with: uv run python scripts/make_yolo_coco_demo.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
QUIZ_DIR  = REPO_ROOT / "figures" / "datasets" / "severstal_quiz"
ASSETS    = REPO_ROOT / "website" / "assets" / "rolled-metal"
OUT_DIR   = REPO_ROOT / "figures" / "bakeoff" / "yolo_coco_candidates"
FINAL_OUT = REPO_ROOT / "figures" / "bakeoff" / "yolo_coco_on_steel.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helper: print image dimensions ───────────────────────────────────────────
def img_dims(path: Path) -> tuple[int, int]:
    img = Image.open(path)
    return img.size  # (width, height)

for p in [QUIZ_DIR / "quiz_a_guess.png", QUIZ_DIR / "quiz_b_guess.png"]:
    w, h = img_dims(p)
    print(f"{p.name}: {w}×{h} px")

# ── candidate source images ───────────────────────────────────────────────────
def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

qa = load_rgb(QUIZ_DIR / "quiz_a_guess.png")
qb = load_rgb(QUIZ_DIR / "quiz_b_guess.png")
ha, wa = qa.shape[:2]
hb, wb = qb.shape[:2]

print(f"quiz_a numpy: {wa}×{ha}")
print(f"quiz_b numpy: {wb}×{hb}")

# Crop fractions tuned to isolate the two strips in each quiz composite,
# skipping the title block (top ~13 %) and the "?" labels (~3 % rows each).
# quiz_a strip 1: rows  16%–54%,  quiz_a strip 2: rows  57%–97%
# quiz_b strip 1: rows  16%–54%,  quiz_b strip 2: rows  57%–97%
candidates: list[tuple[str, np.ndarray]] = [
    ("qa_strip1", qa[int(ha * 0.16) : int(ha * 0.54), int(wa * 0.03) : int(wa * 0.97)]),
    ("qa_strip2", qa[int(ha * 0.57) : int(ha * 0.97), int(wa * 0.03) : int(wa * 0.97)]),
    ("qb_strip1", qb[int(hb * 0.16) : int(hb * 0.54), int(wb * 0.03) : int(wb * 0.97)]),
    ("qb_strip2", qb[int(hb * 0.57) : int(hb * 0.97), int(wb * 0.03) : int(wb * 0.97)]),
    ("factory",   load_rgb(ASSETS / "warm-walz-werk.jpg")),
]

# ── load model ────────────────────────────────────────────────────────────────
model = YOLO("yolo11s.pt")

# ── run inference and save each candidate ────────────────────────────────────
for name, img in candidates:
    results = model(img, conf=0.10, verbose=False)
    result  = results[0]
    annotated_rgb = cv2.cvtColor(result.plot(line_width=3), cv2.COLOR_BGR2RGB)

    n_det = len(result.boxes) if result.boxes is not None else 0
    classes = [model.names[int(b.cls[0])] for b in (result.boxes or [])]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(annotated_rgb)
    ax.axis("off")
    label = ", ".join(f"'{c}'" for c in classes) if classes else "— no detections —"
    ax.set_title(
        f"[{name}]  YOLO11s (COCO pretrained)  |  {n_det} det(s): {label}",
        fontsize=11, loc="left", pad=8, color="#0f172a", fontfamily="monospace",
    )
    plt.tight_layout()
    out = OUT_DIR / f"{name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  {name}: {n_det} det(s) {classes}  → {out}")

# ── produce clean final figure from the best candidate (qb_strip2) ───────────
best_img = qb[int(hb * 0.57) : int(hb * 0.97), int(wb * 0.03) : int(wb * 0.97)]
result   = model(best_img, conf=0.10, verbose=False)[0]
annotated_rgb = cv2.cvtColor(result.plot(line_width=3), cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(16, 5))
ax.imshow(annotated_rgb)
ax.axis("off")
ax.set_title(
    "YOLO11s (COCO pretrained, 80 classes: person, car, bicycle, dog, bird…)  —  no 'steel defect' class exists",
    fontsize=13, loc="left", pad=10, color="#0f172a",
)
plt.tight_layout()
plt.savefig(FINAL_OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nFinal figure saved → {FINAL_OUT}")

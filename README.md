# Scratching the Surface — Rolled-Metal Defect Detection Bake-Off

**Four-way bake-off: ResNet50 vs YOLO11s vs zero-shot VLMs vs Hybrid YOLO→VLM**  
**Cross-domain story:** train on Severstal + NEU-DET → test on KolektorSDD2 + GC10-DET  
**Headline experiment:** VLM auto-label bootstrap — zero human annotations on the unseen domain

---

## Reproduce in ≈30 minutes

```bash
# 1. Clone & install
git clone <repo>
cd Scratching-the-Surface-2
uv sync            # installs all dependencies incl. torch+MPS, ultralytics, openai

# 2. Copy and fill in credentials
cp .env.example .env
# edit .env → add NATIVE_OPENAI_API_KEY, AOAI_*, GCP_PROJECT_ID

# 3. Download datasets (requires Kaggle API key in .env)
uv run python scripts/download_datasets.py

# 4. Build manifests
uv run python -c "
from src.data.severstal import build_manifest as s; s()
from src.data.neu_det    import build_manifest as n; n()
from src.data.kolektor   import build_manifest as k; k()
from src.data.gc10       import build_manifest as g; g()
"

# 5. Train ResNet50 (Phase 2a, ~75 min on M2 Mac)
uv run python scripts/train_resnet.py

# 5b. Fine-tune ResNet50 on kolektor domain (Phase 2a+, ~75 min)
uv run python scripts/finetune_resnet_kolektor.py
# Then run the post-fine-tune pipeline (threshold sweep + figures + slides update):
bash scripts/run_post_finetune_pipeline.sh

# 6. Analyse + auto-launch YOLO training (Phase 2a → 2b)
uv run python scripts/post_training_2a.py

# 7. VLM zero-shot eval (Phase 3a, ~$0.10 with gpt-4o-mini)
uv run python scripts/eval_vlm_zeroshot.py --dry-run   # cost estimate first
uv run python scripts/eval_vlm_zeroshot.py

# 8. Hybrid eval (Phase 3b)
uv run python scripts/eval_hybrid.py

# 9. Bootstrap experiment (Phase 4, ~$5 with gpt-4o)
uv run python scripts/bootstrap_labels.py --dry-run    # cost estimate first
uv run python scripts/bootstrap_labels.py
uv run python scripts/train_yolo_bootstrap.py

# 10. Bake-off comparison figures (Phase 5)
uv run python scripts/make_comparison_figures.py
uv run python scripts/make_qual_grid.py --dataset gc10_test

# 11. Render slides
quarto render website/
```

---

## Project structure

```
configs/          Hyperparameter YAMLs (resnet50.yaml, yolo11s.yaml)
data/
  raw/            Downloaded datasets (gitignored)
  processed/      Manifest parquets — 4 datasets × metadata + split labels
  yolo/           YOLO-format symlink tree (train/val)
  yolo_bootstrap/ VLM-pseudo-labelled YOLO tree (Phase 4)
figures/          Committed PNGs consumed by Quarto slides
models/           Trained checkpoints (gitignored)
notebooks/        Data exploration (01_data_exploration.ipynb)
results/          JSON metrics + VLM JSONL outputs (gitignored)
scripts/          All experiment entry points (see below)
src/
  data/           Dataset parsers + PyTorch Dataset
  models/         ResNet50 model + training loop
  prompts/        SteelDefectX-T3 + CoT prompt template
  vlm_clients.py  Unified wrapper: OpenAI / Azure / Gemini
  cost.py         Budget tracking ($50 cap)
  config.py       Central paths, model IDs, pricing
website/          Quarto source → docs/ (published via GitHub Pages)
app/              Gradio live demo
```

### Script index

| Script | Phase | Purpose |
|--------|-------|---------|
| `train_resnet.py` | 2a | Train ResNet50 binary classifier |
| `analyse_resnet.py` | 2a | Metrics table + training curves + GO/NO-GO |
| `optimise_threshold.py` | 2a | Threshold sweep + ROC curves (also works on fine-tuned model via `--checkpoint`) |
| `post_training_2a.py` | 2a | Master post-training runner (chains above + YOLO gate) |
| `finetune_resnet_kolektor.py` | 2a+ | Fine-tune ResNet50 on kolektor domain data (supervised adaptation ablation) |
| `plot_finetune_history.py` | 2a+ | Plot fine-tune training curve |
| `build_yolo_dataset.py` | 2b | Build YOLO symlink tree from manifests |
| `train_yolo.py` | 2b | Fine-tune YOLO11s (gated on 2a results) |
| `eval_yolo.py` | 2b | Cross-domain YOLO eval + PR curves (also works on bootstrap YOLO via `--weights`, `--output-dir`) |
| `eval_vlm_zeroshot.py` | 3a | VLM zero-shot batch eval (all providers) |
| `eval_hybrid.py` | 3b | YOLO→VLM hybrid pipeline eval |
| `bootstrap_labels.py` | 4 | VLM pseudo-label GC10 images |
| `train_yolo_bootstrap.py` | 4 | Train fresh YOLO on pseudo-labels |
| `make_qual_grid.py` | 5 | Qualitative TP/FP/FN/TN prediction grids |
| `make_comparison_figures.py` | 5 | Bake-off F1/AUC bar charts |
| `update_slides_results.py` | 5 | Auto-update slides.qmd + index.qmd with actual experiment numbers |
| `run_post_finetune_pipeline.sh` | 2a+ | Shell pipeline: threshold sweep → figures → slides update |

---

## Datasets

| Dataset | Role | Images | Source |
|---------|------|--------|--------|
| Severstal | Train + val | 12,568 (53% defect) | [Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection) |
| NEU-DET | Train supplement | 1,800 (100% defect + XML bboxes) | [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |
| KolektorSDD2 | **Held-out test 1** | 3,336 (train+test, binary GT mask) | [Kolektor](https://www.vicos.si/Downloads/KolektorSDD2) |
| GC10-DET | **Held-out test 2** | 2,294 (all defect, Supervisely JSON) | [Kaggle](https://www.kaggle.com/datasets/ztl9/gc10-det) |

The cross-domain split (train on Severstal+NEU → test on Kolektor+GC10) simulates arriving at a new client factory cold, with no labelled images available.

---

## Key results

| Approach | KolektorSDD2 F1 | GC10-DET F1 | Notes |
|----------|-----------------|-------------|-------|
| ResNet50 | 0.162 | **0.782** | Phase 2a, τ=0.475, AUC 0.50 (kolektor) |
| ResNet50+FT | — | — | Phase 2a+, fine-tuning on kolektor train in progress |
| **VLM zero-shot** (gpt-4.1-mini) | **0.793** | 0.658 | Phase 3a complete, AUC 0.8156 (kolektor) |
| Hybrid YOLO→VLM | — | — | Phase 3b (pending YOLO weights) |
| YOLO-bootstrap | — | — | Phase 4 (pending training on 484 pseudo-labels) |

**Headline result:** VLM zero-shot achieves 5× better kolektor F1 (0.793 vs 0.162) with zero domain-specific training. ResNet50+FT tests whether 2,332 labeled kolektor images can match VLM zero-shot.

---

## Cost tracking

All VLM API calls are logged to `results/cost_ledger.csv` with a hard $50 cap.  
Run `uv run python -c "from src.cost import print_ledger; print_ledger()"` to see current spend.

---

## Live demo

```bash
uv run python app/app.py   # → http://localhost:7860
```

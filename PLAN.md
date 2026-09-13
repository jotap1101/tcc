# PLAN.md — Implementation Roadmap

Definitive execution roadmap for the comparative study **U-Net (CNN) vs SegFormer (ViT)** on **semantic segmentation of coffee crops** (Guaxupé – MG, Brazil) using Sentinel-2 Level-2A imagery.

> `AGENTS.md` is the single source of truth for coding, formatting, language and infrastructure rules. Every artifact described here must comply with it.

## 1. Purpose

Guide the sequential, modular implementation of the full pipeline — from Google Earth Engine acquisition to the explainability analysis — while preserving methodological transparency and computational reproducibility.

## 2. Architecture decisions (MLOps)

- **Hybrid layout**: thin Jupyter notebooks (`.ipynb`) act as orchestration/visualization layers; all reusable logic lives in the shared **`src/`** package (pure Python), which is the only target of `ruff`, `mypy` and `pytest`.
- **Single source of truth**: `src/config.yaml` (loaded by `config.py`) centralizes paths, bands, patch size, hyperparameters and seeds. No hardcoded paths in notebooks.
- **Manifest-driven data**: `manifest.parquet` registers every patch (`patch_id`, `tile_id`, `fold`, bbox, `coffee_ratio`, `mask_source`, paths). It is the backbone of reproducibility.
- **Identical training protocol** across models: same split, loss, optimizer, scheduler, metrics and seed — only the architecture differs. This guarantees a scientifically fair comparison.
- **CI (GitHub Actions)** runs `ruff` + `mypy` + `pytest` on push; heavy GPU training runs on Kaggle.
- **Release**: dataset (CC BY 4.0) and model weights published on the Hugging Face Hub.

## 3. Directory layout

```bash
tcc/
├── AGENTS.md                      # source of truth (rules)
├── PLAN.md                        # this roadmap
├── pyproject.toml / uv.lock       # uv + ruff + mypy + pytest
├── src/                            # shared package (reusable logic)
│   ├── config.py + config.yaml
│   ├── data/{dataset,augmentations,manifest,mask_utils,mask_compare,raster_io,gee_client,aoi}.py
│   ├── losses.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── models/{unet,segformer}.py
│   ├── xai/{gradcam,attention_rollout}.py
│   └── io.py  utils.py
├── tests/                         # pytest over src/ only
├── notebooks/                     # 17 sequential .ipynb files
├── data/{raw,interim,processed,external}/   # git-ignored
├── models/                        # git-ignored weights
├── artifacts/                     # git-ignored metrics/figures
└── .github/workflows/ci.yml       # ruff + mypy + pytest
```

## 4. Data flow

```bash
IBGE vector mesh (Região Geográfica Imediata 310044) → AOI polygon
GEE Sentinel-2 L2A (B2/B3/B4/B8) filtered by AOI
  → cloud-free mosaic (Cloud Score+ / QA60)        [01]
  → reference masks (MapBiomas / AlphaEarth)       [02]
  → mask comparison + finalization                 [03, 04]
  → normalized aligned composites                  [05]
  → 512x512 patches + manifest                     [06]
  → spatial k-fold assignment                      [07]
  → EDA / normalization stats                      [08]
  → train U-Net / SegFormer (k=5)                  [09, 10]
  → pixel metrics (IoU/F1/Precision/Recall)        [11]
  → statistical comparison                         [12]
  → Grad-CAM + Attention Rollout                   [13, 14]
  → figures for monografia                         [15]
  → package + publish                              [16]
```

## 5. Phases & notebooks

Each notebook is an isolated stage with a single responsibility and declared inputs/outputs. Execution order is numeric.

| #    | Notebook                                | Phase            | Input → Output                                                    |
| ---- | --------------------------------------- | ---------------- | ----------------------------------------------------------------- |
| `00` | `setup_environment.ipynb`               | 0. Setup         | — → env ready, config loaded, GEE authenticated                   |
| `01` | `gee_sentinel2_acquisition.ipynb`       | 1. Acquisition   | IBGE mesh (310044) → AOI polygon → Sentinel-2 L2A GeoTIFF mosaics |
| `02` | `gee_reference_masks.ipynb`             | 1. Acquisition   | MapBiomas/AlphaEarth → rasterized 10 m binary masks               |
| `03` | `mask_sources_comparison.ipynb`         | 2. Ground Truth  | candidate masks → comparative diagnostic                          |
| `04` | `mask_finalization.ipynb`               | 2. Ground Truth  | chosen protocol → final binary masks                              |
| `05` | `preprocessing.ipynb`                   | 3. Preprocessing | mosaics → cloud-free normalized composites                        |
| `06` | `patch_generation.ipynb`                | 3. Dataset       | composites+masks → 512x512 patches + manifest                     |
| `07` | `spatial_kfold_split.ipynb`             | 3. Dataset       | manifest → manifest with `fold`                                   |
| `08` | `dataset_eda.ipynb`                     | 3. Dataset       | manifest → normalization stats + sanity checks                    |
| `09` | `train_unet.ipynb`                      | 4. Training      | dataset → U-Net weights + per-fold metrics                        |
| `10` | `train_segformer.ipynb`                 | 4. Training      | dataset → SegFormer weights + per-fold metrics                    |
| `11` | `evaluation.ipynb`                      | 5. Evaluation    | predictions → pixel-level IoU/F1/P/R                              |
| `12` | `comparative_analysis.ipynb`            | 5. Evaluation    | metrics → statistical comparison + error maps                     |
| `13` | `xai_gradcam_unet.ipynb`                | 6. XAI           | U-Net → Grad-CAM heatmaps                                         |
| `14` | `xai_attention_rollout_segformer.ipynb` | 6. XAI           | SegFormer → Attention Rollout maps                                |
| `15` | `results_synthesis.ipynb`               | 7. Synthesis     | all → figures/tables for monografia                               |
| `16` | `export_release.ipynb`                  | 7. Dissemination | patches+weights → HF Hub dataset + models                         |

## 6. Config & artifacts schema

- **`config.yaml`**: `aoi` (IBGE region code `310044` + vector-mesh source), `dates`, `bands`, `patch_size`, `fold_count`, `seed`, `coffee_min_ratio`, `model_variants`, loss weights, `lr`, `epochs`, `batch_size`.
- **`manifest.parquet` columns**: `patch_id`, `tile_id`, `fold`, `row`, `col`, `bbox`, `coffee_ratio`, `mask_source`, `image_path`, `mask_path`.
- **Artifacts**: `artifacts/metrics/{model}/fold_{i}.json`, `artifacts/figures/`, `models/{model}/fold_{i}.pt`.

## 7. Progress tracker

### Phase 0 — Setup

- [x] `00_setup_environment.ipynb` — platform detection, dependency install, GEE auth, seeds, config load
- [x] `pyproject.toml` (uv, ruff, mypy, pytest)
- [ ] `uv.lock` gerado e commitado (`uv lock`)
- [x] `.github/workflows/ci.yml` (ruff + mypy + pytest)

### Phase 1 — Acquisition

- [x] `01_gee_sentinel2_acquisition.ipynb` — IBGE mesh import (AOI polygon) + Sentinel-2 acquisition
- [x] `02_gee_reference_masks.ipynb`

### Phase 2 — Ground Truth

- [x] `03_mask_sources_comparison.ipynb`
- [x] `04_mask_finalization.ipynb`

### Phase 3 — Preprocessing & Dataset

- [ ] `05_preprocessing.ipynb`
- [ ] `06_patch_generation.ipynb`
- [ ] `07_spatial_kfold_split.ipynb`
- [ ] `08_dataset_eda.ipynb`

### Phase 4 — Training

- [ ] `09_train_unet.ipynb`
- [ ] `10_train_segformer.ipynb`

### Phase 5 — Evaluation

- [ ] `11_evaluation.ipynb`
- [ ] `12_comparative_analysis.ipynb`

### Phase 6 — XAI

- [ ] `13_xai_gradcam_unet.ipynb`
- [ ] `14_xai_attention_rollout_segformer.ipynb`

### Phase 7 — Synthesis & Dissemination

- [ ] `15_results_synthesis.ipynb`
- [ ] `16_export_release.ipynb`

### Shared package (`src/`) — built alongside the phases

- [x] `config.py` + `config.yaml`
- [ ] `data/dataset.py`, `data/augmentations.py`, `data/manifest.py`
- [x] `data/aoi.py`, `data/mask_utils.py`, `data/mask_compare.py`, `data/raster_io.py`, `data/gee_client.py`
- [ ] `losses.py` (Dice + Focal + Boundary)
- [ ] `metrics.py` (IoU, F1, Precision, Recall)
- [ ] `trainer.py`
- [ ] `models/unet.py`, `models/segformer.py`
- [ ] `xai/gradcam.py`, `xai/attention_rollout.py`
- [ ] `io.py`, `utils.py`
- [ ] `tests/` (pytest for losses, metrics, augmentations, manifest)

## 8. Reproducibility checklist

- [ ] Seeds fixed (`python`/`numpy`/`torch`/`cuda`)
- [ ] `uv.lock` committed; environment dump logged per run
- [ ] Deterministic, non-overlapping patch generation
- [ ] Every notebook resolves input paths from `src/config.py`
- [ ] Identical split/loss/optimizer/metrics across both models

## 9. Risks & open decisions

- **Ground-truth source** is the main risk (MapBiomas vs AlphaEarth vs S2DR3/S2DR4) — handled in Phase 2 before any training.
- **CRS/georeferencing** of labels vs the Sentinel grid (UTM zone for MG).
- **GEE authentication** on Kaggle (service account) — validated in notebook `00`.
- **MiT variant** (b0–b2) balanced against T4/P100 VRAM.

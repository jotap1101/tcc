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
- **Platform-agnostic notebooks**: every notebook runs **correctly** on **Google Colab and Kaggle**, executing the same protocol (same code, seeds, config and processing). All environment detection, Drive mounting and path resolution is abstracted in `src/` — notebooks never use platform-specific paths or APIs.
- **Storage abstraction (`src/io.py`)**: the single contract for platform detection, Drive mounting and path resolution. Exposes at least `detect_platform()`, `mount_drive()` (Colab nativo; Kaggle via Drive API com cache local), `ensure_storage_root()` (idempotent — creates `MyDrive/tcc/` and every required subfolder recursively when absent, no-op when present), `resolve_storage_paths()` (built from `config.yaml`) e `sync_repo_to_workspace()` (cópia do espelho `MyDrive/tcc/repo/` para o workspace quando ele já existe). Notebooks consume only this module.
- **`src/` delivery to the runtime**: the first cell of every notebook downloads and executes `src/bootstrap.py` (stdlib-only) from the public repository — it extracts `src/`, `data/external/` and `requirements-runtime.txt` into the workspace, adds the workspace to `sys.path` and, on Colab, seeds the `MyDrive/tcc/repo/` mirror (idempotent). `src/io.py`'s `sync_repo_to_workspace()` copies from the mirror when it already exists. No manual upload or cloning setup.
- **Idempotent, re-runnable pipeline**: re-running a notebook is safe — it opens/creates the `tcc/` tree and never silently overwrites versioned artifacts; new runs write `run_id`-versioned outputs (metrics, checkpoints, logs) or confirm overwrites explicitly.
- **Platform-specific dependencies**: libraries tied to a host (e.g., `google.colab` Drive mount, Kaggle secrets) are installed inside notebook `00` at runtime — they are **not** added to `pyproject.toml`/`uv.lock`, keeping the repository environment-agnostic.
- **Canonical storage on Google Drive**: the `tcc/` folder at the **root of Google Drive** (`MyDrive/tcc/`) is the single storage root for every datum, model, metric, figure and log. Notebooks access it if it already exists, or create it (and any required subfolder) otherwise — the notebooks themselves generate the folder structure.
- **Ground truth by source**: each ground-truth source (MapBiomas, AlphaEarth, S2DR, and any future source) is integrated in its **own dedicated notebook cell** and stored in its **own subfolder** under `MyDrive/tcc/data/{raw,interim}/<source>/`. Adding a source means adding a cell — never modifying another source's logic.
- **CI (GitHub Actions)** runs `ruff` + `mypy` + `pytest` on push; heavy GPU training runs on Colab/Kaggle with artifacts persisted to the Drive `tcc/` root.
- **In-notebook authentication**: every authentication step (GEE, Hugging Face, Kaggle) is executed inside the notebook's own code cells using the required libraries — e.g., GEE via OAuth with the primary account (`ee.Authenticate()`, token persisted in the Drive `tcc/` root). Credentials are never committed; there is no external/manual authentication setup.
- **Code quality principles**: DRY, Single Responsibility, KISS, YAGNI and Separation of Concerns apply throughout — notebooks orchestrate/visualize and never re-implement or duplicate logic that lives in `src/`; notebook code meets the same quality bar as the shared package.
- **Release**: dataset (CC BY 4.0) and model weights published on the Hugging Face Hub.

## 3. Directory layout

```bash
tcc/
├── AGENTS.md                      # source of truth (rules)
├── PLAN.md                        # this roadmap
├── pyproject.toml / uv.lock       # uv + ruff + mypy + pytest
├── src/                           # shared package (reusable logic)
│   ├── config.py + config.yaml
│   ├── data/{dataset,augmentations,manifest,mask_utils,gee_client}.py
│   ├── losses.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── models/{unet,segformer}.py
│   ├── xai/{gradcam,attention_rollout}.py
│   └── io.py  utils.py
├── tests/                         # pytest over src/ only
├── notebooks/                     # 17 sequential .ipynb files
├── data/{raw,interim,processed}/            # git-ignored mirror of MyDrive/tcc/data
│   └── external/                            # dados de referência VERSIONADOS (ex.: malha IBGE)
├── models/                        # git-ignored mirror of MyDrive/tcc/models
├── artifacts/                     # git-ignored mirror of MyDrive/tcc/artifacts
└── .github/workflows/ci.yml       # ruff + mypy + pytest
```

**Storage layout (canonical — Google Drive `tcc/` root).** The notebooks create/access this tree on demand; no pre-created folder may be assumed:

```bash
MyDrive/tcc/                       # canonical root of all project storage (created/accessed by notebooks)
├── data/{raw,interim,processed,external}/   # spectral data, masks, patches
│   └── raw|interim/{mapbiomas,alphaearth,s2dr,...}/   # ground truth por fonte
├── models/{unet,segformer}/       # weights (mirrored locally, git-ignored)
├── artifacts/{metrics,figures,runs}/       # metrics, figures, run logs
├── repo/src/                      # espelho do código-fonte (entrega do src/ ao runtime)
├── secrets/                       # tokens OAuth persistidos (GEE, Drive) — nunca versionar
└── ...                            # any deeper subfolder required by the pipeline
```

## 4. Data flow

```bash
IBGE vector mesh (Região Geográfica Imediata 310044) → AOI polygon
GEE Sentinel-2 L2A (B2/B3/B4/B8) filtered by AOI
  → cloud-free mosaic (QA60)                       [01]
  → reference masks per source (MapBiomas / AlphaEarth / ...)  [02]
  → per-source mask comparison + finalization                  [03, 04]
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

| #    | Notebook                                | Phase            | Input → Output                                                                                |
| ---- | --------------------------------------- | ---------------- | --------------------------------------------------------------------------------------------- |
| `00` | `setup_environment.ipynb`               | 0. Setup         | — → env ready, `src/` delivered, Drive `tcc/` root resolved, config loaded, GEE authenticated |
| `01` | `gee_sentinel2_acquisition.ipynb`       | 1. Acquisition   | IBGE mesh (310044) → AOI polygon → Sentinel-2 L2A GeoTIFF mosaics                             |
| `02` | `gee_reference_masks.ipynb`             | 1. Acquisition   | each source (MapBiomas/AlphaEarth/...) → per-source 10 m binary masks                         |
| `03` | `mask_sources_comparison.ipynb`         | 2. Ground Truth  | per-source masks → comparative diagnostic                                                     |
| `04` | `mask_finalization.ipynb`               | 2. Ground Truth  | chosen source → final binary masks                                                            |
| `05` | `preprocessing.ipynb`                   | 3. Preprocessing | mosaics → cloud-free normalized composites                                                    |
| `06` | `patch_generation.ipynb`                | 3. Dataset       | composites+masks → 512x512 patches + manifest                                                 |
| `07` | `spatial_kfold_split.ipynb`             | 3. Dataset       | manifest → manifest with `fold`                                                               |
| `08` | `dataset_eda.ipynb`                     | 3. Dataset       | manifest → normalization stats + sanity checks                                                |
| `09` | `train_unet.ipynb`                      | 4. Training      | dataset → U-Net weights + per-fold metrics                                                    |
| `10` | `train_segformer.ipynb`                 | 4. Training      | dataset → SegFormer weights + per-fold metrics                                                |
| `11` | `evaluation.ipynb`                      | 5. Evaluation    | predictions → pixel-level IoU/F1/P/R                                                          |
| `12` | `comparative_analysis.ipynb`            | 5. Evaluation    | metrics → statistical comparison + error maps                                                 |
| `13` | `xai_gradcam_unet.ipynb`                | 6. XAI           | U-Net → Grad-CAM heatmaps                                                                     |
| `14` | `xai_attention_rollout_segformer.ipynb` | 6. XAI           | SegFormer → Attention Rollout maps                                                            |
| `15` | `results_synthesis.ipynb`               | 7. Synthesis     | all → figures/tables for monografia                                                           |
| `16` | `export_release.ipynb`                  | 7. Dissemination | patches+weights → HF Hub dataset + models                                                     |

## 6. Config & artifacts schema

- **`config.yaml`**: `aoi` (region code `310044`, `vector_source` `ibge_mesh`, `mesh_path` apontando para `data/external/ibge/mg_rg_immediatas_2025/`), `data` (`collection`, `dates`, `bands`, `patch_size`, `coffee_min_ratio`, `cloud_threshold`), `splits.fold_count`, `reproducibility.seed`, `model` (`unet_channels`, `segformer_variant`), `loss` weights, `training` (`lr`, `epochs`, `batch_size`). A `storage` block maps the Drive `tcc/` root and every subfolder (`data`, `models`, `artifacts`, `repo`, `secrets`, ...) — all paths are resolved from here, never hardcoded. A `ground_truth.sources` block lists every source with its output subfolder.
- **`manifest.parquet` columns**: `patch_id`, `tile_id`, `fold`, `row`, `col`, `bbox`, `coffee_ratio`, `mask_source`, `image_path`, `mask_path`.
- **Artifacts** (all under the Drive `tcc/` root, paths resolved from `config.yaml`): `MyDrive/tcc/artifacts/metrics/{model}/fold_{i}.json`, `MyDrive/tcc/artifacts/figures/`, `MyDrive/tcc/models/{model}/fold_{i}.pt`, `MyDrive/tcc/data/processed/manifest.parquet`, and `MyDrive/tcc/data/processed/normalization_stats.json` (estatísticas de normalização por banda — estágio 08).

## 7. Progress tracker

### Phase 0 — Setup

- [x] `00_setup_environment.ipynb` — platform detection, Drive mount + `tcc/` root resolution, `src/` delivery (primeira célula baixa e executa `src/bootstrap.py`: extrai `src/`, `data/external/` e `requirements-runtime.txt`, adiciona o workspace ao `sys.path` e cria o espelho `MyDrive/tcc/repo/` no Colab), dependency install (`requirements-runtime.txt`), GEE authentication (executed inside notebook cells), seeds + deterministic flags, config load, environment self-check
- [x] `src/io.py` (platform detection, Drive mount, `tcc/` root ensure/resolve, `src/` delivery) + unit tests
- [x] `pyproject.toml` (uv, ruff, mypy, pytest)
- [x] `requirements-runtime.txt` (versões pinadas instaladas pelo notebook 00)
- [x] `uv.lock` (gerado via `uv lock`)
- [x] `.github/workflows/ci.yml` (ruff + mypy + pytest + validação de notebooks)

### Phase 1 — Acquisition

- [x] `01_gee_sentinel2_acquisition.ipynb` — IBGE mesh import (AOI polygon) + Sentinel-2 acquisition
- [x] `02_gee_reference_masks.ipynb`

### Phase 2 — Ground Truth

- [x] `03_mask_sources_comparison.ipynb`
- [x] `04_mask_finalization.ipynb` — fonte escolhida (AlphaEarth) → máscara binária final

### Phase 3 — Preprocessing & Dataset

- [x] `05_preprocessing.ipynb`
- [x] `06_patch_generation.ipynb`
- [x] `07_spatial_kfold_split.ipynb` — divisão espacial k-fold (k-means determinístico em numpy puro sobre centroides) gravada na coluna `fold` do manifesto, com `split.meta.json` para idempotência
- [x] `08_dataset_eda.ipynb` — estatísticas de normalização por banda (média, desvio, fração de NaN) persistidas em `normalization_stats.json` com fingerprint, verificações de sanidade e figura de resumo

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
- [ ] `data/dataset.py`, `data/augmentations.py`
- [x] `data/patch_generation.py` (patches 512x512 + manifesto Parquet — estágio 06)
- [x] `data/mask_utils.py` (uma função por fonte de ground truth), `data/gee_client.py`
- [x] `data/mask_comparison.py` (diagnóstico comparativo das fontes — estágio 03)
- [x] `data/mask_finalization.py` (finalização da máscara — estágio 04)
- [x] `data/preprocessing.py` (alinhamento e normalização do composite — estágio 05)
- [x] `data/spatial_split.py` (divisão espacial k-fold do manifesto — estágio 07)
- [x] `data/eda.py` (estatísticas de normalização + sanidade do dataset — estágio 08)
- [ ] `losses.py` (Dice + Focal + Boundary)
- [ ] `metrics.py` (IoU, F1, Precision, Recall)
- [ ] `trainer.py` (protocolo de treino único, parametrizado pelo modelo — sem duplicação entre notebooks 09/10)
- [ ] `models/unet.py`, `models/segformer.py`
- [ ] `xai/gradcam.py`, `xai/attention_rollout.py`
- [x] `io.py` (platform detection + Drive mount + `tcc/` root ensure/resolve + `src/` delivery via espelho), `bootstrap.py` (entrega do `src/` ao runtime antes do import), `utils.py`
- [x] `tests/` (config, io, utils — pytest)

## 8. Reproducibility checklist

- [ ] Seeds fixed (`python`/`numpy`/`torch`/`cuda`)
- [ ] `uv.lock` committed; environment dump logged per run
- [ ] Runtime library versions pinned in notebook `00` (`requirements-runtime.txt`) and logged per run (Colab/Kaggle default envs differ)
- [ ] Torch deterministic flags set (`cudnn.deterministic`, `benchmark=False`, `use_deterministic_algorithms`) and residual nondeterminism documented
- [ ] Notebooks committed with cleared outputs (no saved cell outputs)
- [ ] Notebook structural conventions validated in CI (naming, markdown-above-code, empty outputs)
- [ ] Storage operations idempotent: re-runs never silently overwrite versioned artifacts (`run_id`-versioned outputs)
- [ ] Deterministic, non-overlapping patch generation
- [ ] Every notebook resolves input paths from `src/config.py`
- [ ] Notebooks are platform-agnostic: run correctly on Colab and Kaggle (same protocol), no hardcoded platform paths/APIs
- [ ] All artifacts stored under the Drive `tcc/` root; notebooks create the root and subfolders on demand
- [ ] Identical split/loss/optimizer/metrics across both models

## 9. Risks & open decisions

- **Ground-truth source** is the main risk (MapBiomas vs AlphaEarth vs S2DR3/S2DR4) — each source has its own integration cell and Drive subfolder; the choice is handled in Phase 2 before any training.
- **CRS/georeferencing** of labels vs the Sentinel grid (UTM zone for MG).
- **GEE OAuth on Kaggle** (headless `ee.Authenticate()` flow + token persistence across sessions) — validated in notebook `00`.
- **Drive mounting on Kaggle** (no native Drive mount — Drive API via OAuth token `GDRIVE_TOKEN`/`GDRIVE_TOKEN_FILE` + local cache) — must be validated in notebook `00` before any storage write; the `src/` abstraction isolates it from notebook code.
- **Credential provisioning per platform** (Colab `userdata`/Drive file vs Kaggle secrets vs environment variables) — `io.py` and the notebooks read credentials uniformly from environment variables; validated in notebook `00`.
- **Drive I/O latency and quota** on heavy write steps (patch generation, training checkpoints) — mitigated by a local working cache in the platform workspace, with the Drive `tcc/` root as the authoritative target.
- **MiT variant** (b0–b2) balanced against T4/P100 VRAM.

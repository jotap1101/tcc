# AGENTS.md

System instruction and operating manual for all AI-assisted development in this repository. Follow these directives for every code generation and interaction.

## Project Overview

Research project (deep learning / computer vision) on **semantic segmentation of coffee crops** for precision agriculture. The study performs a **comparative analysis between Convolutional Neural Networks (U-Net) and Vision Transformers (SegFormer)**.

- **Territorial focus:** Região Geográfica Imediata de Guaxupé – MG, Brazil.
- **Language:** Python (with PyTorch and Hugging Face Transformers).
- **Primary goal:** full methodological transparency and computational reproducibility through continuous versioning of the code.

## Scientific & Methodological Context

- **Data acquisition:** Google Earth Engine (GEE).
- **Spectral source:** Sentinel-2 Level-2A imagery, bands **B2, B3, B4, B8** at **10 m** spatial resolution.
- **Preprocessing:** split orbital spectral data into **512x512 px patches**.
- **Loss function:** multivariate loss combining **Dice Loss + Focal Loss + Boundary Loss**.
- **Validation:** spatial **k-fold (k=5)** with pixel-level metrics: **IoU, F1-Score, Precision, Recall**.
- **Explainability (XAI):**
  - U-Net (native CNN): **Grad-CAM** maps.
  - SegFormer (ViT): **Attention Rollout** for contextual inference.

## Licensing (dual)

- **MIT** — codebase.
- **CC BY 4.0** — documentation and datasets.

## Development Structure & Writing Rules (strict)

### Code Architecture

- Development uses a **hybrid layout**: each stage is an **isolated Jupyter Notebook (`.ipynb`)** acting as an orchestration/visualization layer that imports reusable logic from the shared **`src/`** package (pure Python).
- `ruff`, `mypy` and `pytest` target **`src/`** and **`tests/`** only — never notebooks. Notebooks are not linted, typed or unit-tested directly (their structural conventions are validated in CI).
- Notebooks run in numeric order and must resolve every input path from `src/config.py` (no hardcoded paths).
- Notebooks are committed with **cleared outputs** (no saved cell outputs/figures); strip outputs before committing — CI enforces it, keeping diffs clean and free of platform-specific paths.

### Modularity

- Development is structured in **sequential, integrated stages**. Each defined sub-stage must be an **isolated Jupyter Notebook (`.ipynb`)** containing **only** the cells required to execute that specific stage.

### Single responsibility

- Each cell (markdown or code) must have **one** clear, well-defined functional responsibility.

### Markdown cells

- **Mandatory:** immediately above **every** code cell there must be a markdown cell concisely describing what the following code does.
- Language: **exclusively pt-br**.
- Tone: neutral and **impersonal** — never reference any person ("eu", "nós", "meu" are forbidden).

### Code cells

- **Identifiers (variables, functions, classes, logic): written entirely in en-US** for global standardization.
- Every code cell must include short, objective technical comments (above or to the right of each critical instruction) explaining its technical role.
- Comment language: **pt-br**, impersonal and neutral.

### Code quality & engineering principles

- Apply standard software engineering principles to **every notebook cell and to `src/`**: **DRY** — never duplicate logic; notebooks import from `src/` instead of re-typing code the package already provides; **Single Responsibility** — one concern per cell/function; **KISS** — prefer the simplest correct solution; **YAGNI** — no speculative or unused code; **Separation of Concerns** — orchestration/visualization in notebooks, reusable logic in `src/`; and **readability** — clear en-US identifiers, pt-br comments, explicit and deterministic code.
- Notebook code follows the same quality bar as `src/`: idiomatic, consistent with repo conventions, and free of dead code or duplicated blocks.

## Tech Stack & Conventions

- **Package manager:** `uv`.
- **Linting:** `ruff`.
- **Type checking:** `mypy`.
- **Tests:** `pytest`.
- Follow community and academic best practices for AI / Deep Learning / Computer Vision.

### Reproducibility

- Single source of truth for configuration: `src/config.yaml` (loaded by `config.py`).
- Fix all seeds (`python`/`numpy`/`torch`/`cuda`); commit `uv.lock`; log the full environment per run.
- **Runtime versions pinned for cross-platform consistency**: Colab and Kaggle ship different default environments, so notebook `00` installs the pinned versions in `requirements-runtime.txt` and every run logs the full environment (versions, seeds, config hash, commit SHA, manifest hash).
- **Torch determinism enforced**: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False` and `torch.use_deterministic_algorithms(True)` where supported; any residual nondeterminism must be documented.
- Data is registered in a versioned `manifest.parquet`; heavy artifacts live outside git, in the Drive `tcc/` root (optionally mirrored in local git-ignored `data/`, `models/`, `artifacts/`).

### Directory Structure & Artifacts

- The canonical artifact tree lives inside the Drive `tcc/` root (see "Platform Agnosticism & Storage"): `tcc/data/{raw,interim,processed,external}`, `tcc/models/`, `tcc/artifacts/`, and any deeper subfolders the pipeline needs. Notebooks create this tree on demand.
- Local `data/`, `models/` and `artifacts/` are git-ignored and, at most, mirror the Drive tree — the Drive `tcc/` root is authoritative. Exception: `data/external/` holds small immutable reference inputs (e.g., the IBGE mesh) and **is versioned** in the repo.
- **Versioned reference data**: reference inputs (e.g., `data/external/ibge/mg_rg_immediatas_2025/`) are tracked in git and delivered to the runtime alongside `src/` by the workspace bootstrap (`src/bootstrap.py`, executed in the first cell of every notebook); `data/{raw,interim,processed}` remain git-ignored mirrors of the Drive.
- **Ground truth by source**: each ground-truth source (MapBiomas, AlphaEarth, S2DR, and any future source) is integrated in its **own dedicated code cell** and stored in its **own subfolder** under `MyDrive/tcc/data/{raw,interim}/<source>/`; adding a source means adding a cell, never modifying another source's logic.
- Notebook naming: `NN_verb_snake_case.ipynb` inside `notebooks/`, executed in numeric order.

### Secrets & Authentication

- Credentials (GEE OAuth token, Hugging Face token, Kaggle secrets) are read from environment variables only and are **never committed**, logged, or exposed.
- Service-account key files and other secret files (e.g., GEE `*.json` keys, `.env`) are **git-ignored** and never tracked — GEE is authenticated via OAuth with the primary account, so such keys are not required.
- **Authentication happens inside the notebooks themselves**: the code cells execute the authentication instructions with the required libraries/frameworks (e.g., GEE via OAuth with the primary account using `ee.Authenticate()`, Hugging Face `login`, Kaggle secrets), so every notebook run self-authenticates on the platform where it executes. There is no external or manual authentication step outside the notebooks.

### State of the art

- Prioritize the **official documentation** of libraries in their **latest versions**.
- Never restrict yourself to the proposed path if a superior approach exists: suggest and implement the most efficient, performant, or modern alternative for the context.

### MCP tools

Use the available MCP servers actively for repository inspection, architectural validation, and multi-step reasoning while designing the practical structure: `github`, `context7`, `gh_grep`, `sequential-thinking`, `hf-mcp-server` (see `opencode.json`).

## Execution Environment (critical)

- Notebook and training code is **generated in this local environment**, but **executed on Google Colab or Kaggle** (Kaggle is mandatory). Notebooks must therefore be **platform-agnostic**: the same `.ipynb` runs correctly on either platform and is worked the same way (same protocol, seeds, versions and processing) — not bit-identical outputs across different hardware.
- The repository `.venv` holds the project and development dependencies and is used to **validate the CI quality gates locally** (ruff, mypy, pytest and notebook-conventions). CI (GitHub Actions) runs the same gates on push.

## Platform Agnosticism & Storage (critical)

### Platform-agnostic notebooks

- **Rule:** every notebook must run **correctly** on **Google Colab** and **Kaggle**, executing the same protocol (same code, seeds, config and processing) — results are worked identically, not required to be bit-identical across platforms. No platform-specific APIs, magic commands, hardcoded paths, or environment-dependent logic in notebooks.
- All platform detection, Drive mounting, and path resolution is centralized in `src/` (e.g., `io.py`, `config.py`) — the **only** place environment specifics live. Notebooks consume that abstraction only.
- **`src/` + reference data delivery to the runtime**: the first cell of every notebook downloads and executes `src/bootstrap.py` (stdlib-only) from the public repository (`github.com/jotap1101/tcc`) — necessary because `src/` is not importable yet. The bootstrap downloads the repo tarball, extracts `src/`, `data/external/` and `requirements-runtime.txt` into the workspace, adds the workspace to `sys.path` and, on Colab, seeds the `MyDrive/tcc/repo/` mirror (idempotent). `src/io.py` still exposes `sync_repo_to_workspace()`, which copies the mirror to the workspace when it already exists. No manual upload or cloning setup is required.
- **Drive on Kaggle**: Kaggle has no native Drive mount; `src/io.py` accesses Drive via the Google Drive API (OAuth token of the primary account from `GDRIVE_TOKEN`/`GDRIVE_TOKEN_FILE`) with a local cache under `/kaggle/working/drive`. Notebook code stays uniform — a single `mount_drive()` / `ensure_storage_root()` call.
- Notebooks must be deterministic and platform-consistent: same seeds, same config, same protocol on both platforms; determinism refers to the protocol, not bit-identical outputs across different hardware.

### Google Drive `tcc/` root — canonical storage

- **Rule:** the `tcc/` folder at the **root of Google Drive** (`MyDrive/tcc/`) is the single, canonical storage root for the entire project.
- Every datum, file, model, metric, figure, log, or artifact produced by the pipeline is stored **inside** `tcc/`, organized in subfolders of any required depth. Nothing is persisted outside it.
- Notebooks must **access** `tcc/` when it already exists, or **create it** (and every needed subfolder, recursively) when it does not. The notebooks themselves generate the folder structure — no pre-created layout may be assumed.
- Storage operations are **idempotent and re-runnable**: `ensure_storage_root()` succeeds whether or not `tcc/` exists, and re-running a notebook never corrupts or silently overwrites versioned artifacts (manifest, data, weights) — new runs write versioned outputs (e.g., `run_id` subfolders) or require explicit confirmation to overwrite.
- Storage paths are resolved from `src/config.yaml` / `config.py`; notebooks never hardcode Drive paths.

## Repository Structure

- **`docs/`** is a **git submodule** (`tcc-docs`) containing only theoretical material: images, literature-review articles, defense slides, the research project, etc. Treat it as **read-only** — never modify it.
- **`PLAN.md`** is the authoritative roadmap: it defines phase ordering and each notebook's inputs/outputs. Notebooks are executed following it.

## Status

Scaffolding in place: `pyproject.toml`, `requirements-runtime.txt`, `src/` (config, io, utils), `tests/`, CI and this rulebook. Stage notebooks (`.ipynb`) not yet created.

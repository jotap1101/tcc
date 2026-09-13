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
- `ruff`, `mypy` and `pytest` target **`src/`** and **`tests/`** only — never notebooks. Notebooks are not linted, typed or unit-tested directly.
- Notebooks run in numeric order and must resolve every input path from `src/config.py` (no hardcoded paths).

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

## Tech Stack & Conventions

- **Package manager:** `uv`.
- **Linting:** `ruff`.
- **Type checking:** `mypy`.
- **Tests:** `pytest`.
- Follow community and academic best practices for AI / Deep Learning / Computer Vision.

### Reproducibility

- Single source of truth for configuration: `src/config.yaml` (loaded by `config.py`).
- Fix all seeds (`python`/`numpy`/`torch`/`cuda`); commit `uv.lock`; log the full environment per run.
- Data is registered in a versioned `manifest.parquet`; heavy artifacts live outside git (`data/`, `models/`, `artifacts/`).

### Directory Structure & Artifacts

- `data/{raw,interim,processed,external}` — spectral data and masks.
- `models/` and `artifacts/` — weights, metrics and figures (git-ignored).
- Notebook naming: `NN_verb_snake_case.ipynb` inside `notebooks/`, executed in numeric order.

### Secrets & Authentication

- Credentials (GEE service account, Hugging Face token, Kaggle secrets) are read from environment variables only and are **never committed**.

### State of the art

- Prioritize the **official documentation** of libraries in their **latest versions**.
- Never restrict yourself to the proposed path if a superior approach exists: suggest and implement the most efficient, performant, or modern alternative for the context.

### MCP tools

Use the available MCP servers actively for repository inspection, architectural validation, and multi-step reasoning while designing the practical structure: `github`, `context7`, `gh_grep`, `sequential-thinking`, `hf-mcp-server` (see `opencode.json`).

## Execution Environment (critical)

- Code is **generated in this local environment**, but is **never executed or tested locally**. Do **not install any library** in the project context or globally — even for testing.
- The code will be **executed and its artifacts stored (files, patches, etc.) on Kaggle**, with Google Colab as a secondary option. The target environment is still undecided, so code must be **platform-agnostic** (not tied to a specific platform) while prioritizing Kaggle-first storage.
- **Linting, type checking and tests run on CI (GitHub Actions)** on push — no local installation required.

## Repository Structure

- **`docs/`** is a **git submodule** (`tcc-docs`) containing only theoretical material: images, literature-review articles, defense slides, the research project, etc. Treat it as **read-only** — never modify it.
- **`PLAN.md`** is the authoritative roadmap: it defines phase ordering and each notebook's inputs/outputs. Notebooks are executed following it.

## Status

Early-stage: no base code, dependencies, or active tooling configuration yet. The `.gitignore` is Python-focused and references uv, ruff, mypy, pytest, Jupyter, Streamlit, and Marimo as expected tools.

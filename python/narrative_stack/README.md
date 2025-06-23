# Narrative Stack Python Utils

> ⚠️ *This setup is provisional and will be refined.*

## 1. Installing Dependencies

This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

### Install `uv` (if not already installed):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create and activate a virtual environment:

```sh
uv venv
source .venv/bin/activate
```

### Install project dependencies:

```sh
uv pip install -e .
```

## 2. Prerequisites

After dependencies are installed, complete setup instructions in:

📄 [`STAGE_1_PREP.md`](./STAGE_1_PREP.md)

## 3. Starting Jupyter

Launch Jupyter Lab inside the environment:

```sh
jupyter lab
```

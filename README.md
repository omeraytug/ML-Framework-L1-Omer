# ML Framework – Environment Verification

This project uses **uv** for dependency management.

## Requirements
The environment includes:
- PyTorch (with correct accelerator support for the hardware)
- Scikit-learn
- Pandas
- Jupyter

All dependencies are locked in `uv.lock`.

## Setup
First, sync the dependencies:
```bash
uv sync
```

## Verification
Run the verification script:
```bash
uv run check_env.py
```
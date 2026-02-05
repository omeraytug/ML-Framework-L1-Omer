# ML Framework – Environment Verification

This project uses **uv** for dependency management.

## Requirements
The environment includes:
- PyTorch (with correct accelerator support for the hardware)
- Scikit-learn
- Pandas
- Jupyter

All dependencies are locked in `uv.lock`.

## Verification
Run the verification script:
```bash
uv run check_env.py
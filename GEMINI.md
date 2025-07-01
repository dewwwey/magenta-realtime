# Gemini Notes

## Project Setup

- This project uses a virtual environment (`venv`) for managing dependencies, created using `uv`.
- Dependencies are installed using `uv pip`.

## Current Progress & Issues

- Encountered `ImportError` with TensorFlow due to `ml-dtypes` version conflicts when using Python 3.12.
- Switched to Python 3.11, but a new `ml-dtypes` conflict arose between `flax` (requiring `>=0.5.0` via `flaxformer` and `jax`) and `tensorflow==2.16.1` (requiring `>=0.3.1,<0.4.dev0`). These versions are fundamentally incompatible.

## Next Steps

- **Fork `magenta-rt`:** Due to persistent and fundamental dependency conflicts between `flax` and `tensorflow`, the most viable path forward is to fork the `magenta-rt` repository. This will allow direct modification of `pyproject.toml` to specify compatible versions of these libraries.
- **Identify compatible versions:** Once forked, we will need to systematically identify a set of `flax` and `tensorflow` versions that are mutually compatible and also work with Python 3.11.
- **Reinstall dependencies:** After modifying `pyproject.toml` in the forked repository, reinstall all project dependencies.

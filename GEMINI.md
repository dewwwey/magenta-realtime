# Gemini Notes

## Project Setup

- This project uses a virtual environment (`venv`) for managing dependencies, created using `uv`.
- Dependencies are installed using `uv pip`.

## Current Progress & Issues

- Encountered `ImportError` with TensorFlow due to `ml-dtypes` version conflicts when using Python 3.12.
- Switched to Python 3.11, and resolved `ml-dtypes` conflict by ensuring `tensorflow==2.19.0` and `tf-keras==2.19.0` were installed.
- Relative import errors in tests were resolved by adding `pythonpath = ["."]` to `pyproject.toml`.
- `RuntimeError` with `tensorflow.config.experimental.set_memory_growth` was resolved by moving the `_globally_disable_gpu_memory_growth()` call to an earlier point in `magenta_rt/utils.py`.
- `DefaultCredentialsError` was encountered due to missing Google Cloud authentication. Attempted to install `gcloud` CLI, which required several manual steps due to `sudo` password prompts and GPG key issues.
- `gcloud auth application-default login` was successfully executed, but some tests still report `OSError: Project was not passed and could not be determined from the environment`.
- New error: `InvalidArgumentError: Graph execution error: Cannot deserialize computation` with `unregistered operation 'vhlo.sine_v2'` and `failed to deserialize portable artifact using StableHLO_v1.9.1`. This indicates an incompatibility between TensorFlow/XLA and `flaxformer`/`t5x`.

## Next Steps

- **Set Google Cloud Project ID:** Set the `GOOGLE_CLOUD_PROJECT` environment variable to `proven-space-337622` to resolve the `OSError`.
- **Investigate XLA/StableHLO incompatibility:** Research compatible versions of `jax`, `jaxlib`, `flaxformer`, and `t5x` with TensorFlow 2.19.0.
- **Add GEMINI.md to .gitignore:** Prevent `GEMINI.md` from being committed to the repository.
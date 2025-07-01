# Gemini Notes

## Project Setup

- This project uses a virtual environment (`venv`) for managing dependencies, created using `uv`.
- Dependencies are installed using `uv pip`.

## Current Progress & Issues

- Encountered `ImportError` with TensorFlow due to `ml-dtypes` version conflicts when using Python 3.12.
- Switched to Python 3.11.
- Relative import errors in tests were resolved by adding `pythonpath = ["."]` to `pyproject.toml`.
- `RuntimeError` with `tensorflow.config.experimental.set_memory_growth` was resolved by moving the `_globally_disable_gpu_memory_growth()` call to an earlier point in `magenta_rt/utils.py`.
- `DefaultCredentialsError` was encountered due to missing Google Cloud authentication. `gcloud auth application-default login` was successfully executed, and the `GOOGLE_CLOUD_PROJECT` environment variable was set to `proven-space-337622`.
- The original `magenta-realtime` repository uses `tf-nightly`, `tensorflow-text-nightly`, and `tf-hub-nightly`, suggesting it's built against bleeding-edge TensorFlow and is likely TPU-optimized.
- **Fasttext Build Failure:** `fasttext==0.9.2` consistently fails to build, even with `pybind11` pre-installed. This is a known issue with no prebuilt wheels available.
- **Solution for Fasttext:** The user has forked `t5x` (at `https://github.com/dewwwey/t5x.git`) and modified its dependencies to accept `fasttext==0.9.3`, which builds successfully.
- **Critical Blocker: Fundamental `ml-dtypes` and `tensorflow-text` Incompatibility:** There was a persistent and fundamental `ml-dtypes` version conflict between TensorFlow and the JAX ecosystem. Furthermore, `seqio` (a dependency of `t5x`) implicitly pulls in `tensorflow-text`, leading to `NotFoundError` and `undefined symbol` errors due to binary incompatibilities between `tensorflow-text` and the installed TensorFlow/JAX components. This made it impossible to install both sets of libraries simultaneously in the same environment.
- **Test Status:** 45 out of 49 tests passed before encountering the XLA errors, suggesting core functionality might be operational despite the XLA errors. However, the `ml-dtypes` and `tensorflow-text` conflicts prevented even basic import of necessary modules.
- **Attempted CPU-only execution:** Even with CPU-only setup, the `ml-dtypes` incompatibility persisted, preventing the `run_magenta.py` script from executing due to `AttributeError: cannot import name 'float8_e3m4' from 'ml_dtypes'`.
- **Solution for TensorFlow/JAX/ml-dtypes/tensorflow-text incompatibility:** Replicating the Colab notebook's installation strategy by explicitly uninstalling all TensorFlow packages and then installing specific nightly versions (`tf-nightly==2.20.0.dev20250619`, `tensorflow-text-nightly==2.20.0.dev20250316`, `tf-hub-nightly`) resolved the binary incompatibility issues. `tf2jax` was also explicitly installed.
- **Current Status:** The `run_magenta.py` script successfully executed, generating an audio file that sounds like actual music. This confirms that the core music generation functionality is now working on the GPU.

## Roadmap

**End Goal:** Use `magenta-rt` for live music generation, influenced by external sensors (e.g., light sensor controlling "Jazz" in the mix).

**Strategic Decision:** To achieve the end goal, we must prioritize the JAX/Flax ecosystem and remove TensorFlow dependencies, as `MagentaRT`'s core models are built on JAX.

**System Information:** Ubuntu 24.04, AMD 7800X3D, 64 GB RAM, RTX 4070 Ti.

### Next Steps:

1.  **Explore `MagentaRT` API:** Understand how to control the music generation (e.g., changing styles, parameters) from Python. This will involve examining `magenta_rt/system.py` and `magenta_rt/musiccoca.py` more deeply.
2.  **Integrate with Home Assistant Sensors:** This will involve reading sensor data and mapping it to `MagentaRT` parameters. This is a separate development task that can begin once the `MagentaRT` API is understood.

## Gemini CLI Usage Notes

- The `gh` (GitHub CLI) is available in this environment for repository interactions.
- Please regularly update this `GEMINI.md` file with efficiently worded notes on project progress, blockers, and a high-level roadmap. This helps maintain context and streamline future interactions.

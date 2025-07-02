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
- **Current Status:** The `run_magenta.py` script successfully executed, generating an audio file that sounds like actual music. The `run_dynamic_style.py` script also successfully executed, demonstrating dynamic style blending and real-time audio streaming. The core music generation functionality is now working on the GPU, and real-time performance has been achieved for audio generation (5 seconds of audio in ~4 seconds).

**VRAM Debugging Results:**
- Initial VRAM usage: ~2.0 GB (baseline)
- After style embedding: ~2.2 GB (+200 MB)
- During audio generation: Spikes up to 11.4 GB (near GPU limit)
- Final VRAM usage: ~11.4 GB (90% utilization)
- Observed XLA warnings about large memory allocations (8-130 GiB attempts)

## Roadmap

**End Goal:** Use `magenta-rt` for live music generation, influenced by external sensors (e.g., light sensor controlling "Jazz" in the mix).

**Strategic Decision:** To achieve the end goal, we must prioritize the JAX/Flax ecosystem and remove TensorFlow dependencies, as `MagentaRT`'s core models are built on JAX.

**System Information:** Ubuntu 24.04, AMD 7800X3D, 64 GB RAM, RTX 4070 Ti.

### Next Steps:

**Memory Optimization Tasks:**
1. Reduce batch size or other parameters in `generate_chunk` to lower VRAM usage.
2. Add detailed logging for TensorFlow/XLA memory allocations.
3. Test with smaller model variants if available.
4. Profile and optimize tensor retention in memory.
5. **Quantization:** Investigate reducing the precision of model weights (e.g., float32 to float16 or int8) to reduce VRAM usage and improve performance. This may require retraining or fine-tuning.
6. **Model Pruning/Distillation:** Explore techniques to reduce model size by removing redundant connections or training smaller models to mimic larger ones.
7. **Dynamic Batching:** Investigate implementing dynamic batching, where the batch size adapts based on available memory.
8. **JAX/XLA Memory Management:** Deep dive into JAX's memory allocation and deallocation mechanisms for explicit memory clearing or more aggressive garbage collection.
9. **Offloading Layers/Compute:** Explore offloading less performance-critical layers or computations to CPU RAM or the integrated GPU (iGPU) to free up VRAM on the dedicated GPU. This would require significant changes to the model loading and execution pipeline.
10. **Profiling:** Utilize advanced profiling tools (e.g., `nvprof`, `TensorFlow Profiler` for JAX) to pinpoint exact memory bottlenecks and inefficient operations.
11. **JAX/XLA Memory Profiling:** Use JAX's built-in memory profiling tools (e.g., `jax.profiler`) to get a detailed breakdown of memory allocation during different phases of generation. This will help pinpoint the exact source of high VRAM usage.
12. **Manual JAX/XLA Memory Management:** Investigate if there are any JAX/XLA APIs for explicit memory clearing or garbage collection that can be strategically called after certain operations (e.g., after `embed_style` or `generate_chunk`).
13. **Chunking Strategy Optimization:** Re-evaluate the `chunk_length` and `context_length` parameters in `MagentaRTConfiguration`. While my previous attempt to reduce `context_length` didn't yield significant results, a more nuanced approach might be needed, perhaps by analyzing the memory footprint of different chunk sizes.
14. **Explore `jax.experimental.compilation_cache`:** Investigate if JAX's compilation cache can be leveraged to reduce compilation overhead, which might indirectly affect perceived performance.


1.  **Integrate with Home Assistant Sensors:** This will involve reading sensor data and mapping it to `MagentaRT` parameters. This is a separate development task that can begin once the `MagentaRT` API is understood.

## Gemini CLI Usage Notes

- The `gh` (GitHub CLI) is available in this environment for repository interactions.
- Please regularly update this `GEMINI.md` file with efficiently worded notes on project progress, blockers, and a high-level roadmap. This helps maintain context and streamline future interactions.

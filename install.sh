#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting Magenta RT setup..."

# 1. Install uv (if not already installed)
if ! command -v uv &> /dev/null
then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the current session
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "uv is already installed."
fi

# 2. Create and activate a virtual environment
echo "Creating and activating virtual environment..."
uv venv
source .venv/bin/activate

# 3. Install core dependencies (including fasttext fix)
echo "Installing core dependencies..."
uv pip install .[test,gpu]

# 4. Force TensorFlow Nightly Builds
echo "Uninstalling conflicting TensorFlow packages..."
uv pip uninstall -y tensorflow tf-nightly tensorflow-cpu tf-nightly-cpu tensorflow-tpu tf-nightly-tpu tensorflow-hub tf-hub-nightly tensorflow-text tensorflow-text-nightly || true

echo "Installing specific TensorFlow nightly builds..."
uv pip install tf-nightly==2.20.0.dev20250619 tensorflow-text-nightly==2.20.0.dev20250316 tf-hub-nightly tf2jax

echo "Magenta RT setup complete!"
echo "To activate the virtual environment, run: source .venv/bin/activate"

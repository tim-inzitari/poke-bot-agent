#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  else
    PYTHON_BIN="python3"
  fi
fi
VENV_DIR="${VENV_DIR:-.venv}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_APT_PACKAGES="${INSTALL_APT_PACKAGES:-1}"

echo "Ubuntu poke-agent setup"
echo "  python:            $PYTHON_BIN"
echo "  venv:              $VENV_DIR"
echo "  pytorch index url: $PYTORCH_INDEX_URL"
echo

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "warning: this setup script is intended for Ubuntu/Linux; continuing anyway" >&2
fi

if [[ "$INSTALL_APT_PACKAGES" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
  echo "installing system packages"
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    curl \
    git \
    jq \
    python3-dev \
    python3-pip \
    python3-venv \
    tmux \
    unzip
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "missing $PYTHON_BIN; install Python 3.11 or set PYTHON_BIN=python3" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "$PYTORCH_INDEX_URL" torch torchvision torchaudio
python -m pip install -r requirements.txt

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  nvidia-smi
else
  echo "warning: nvidia-smi not found; install NVIDIA driver before long runs" >&2
fi

if [[ ! -f kaggle/input/cg-lib/cg/libcg.so ]]; then
  if [[ -f "$HOME/.kaggle/kaggle.json" || -n "${KAGGLE_USERNAME:-}" ]]; then
    echo "downloading Kaggle cg-lib and episodes index"
    scripts/download-kaggle-inputs.sh
  else
    echo "warning: missing kaggle/input/cg-lib/cg/libcg.so" >&2
    echo "         add ~/.kaggle/kaggle.json, then run scripts/download-kaggle-inputs.sh" >&2
  fi
fi

echo
python scripts/ubuntu_probe.py || true

echo
echo "Setup complete."
echo "Next:"
echo "  cp configs/ubuntu_two_week.env.example configs/ubuntu_two_week.env"
echo "  tmux new -s poke-agent"
echo "  bash scripts/ubuntu_long_run.sh configs/ubuntu_two_week.env"

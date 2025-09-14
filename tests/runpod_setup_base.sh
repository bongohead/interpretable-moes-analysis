#!/bin/bash
set -eu pipefail

KERNEL_NAME=moes-venv
PROJECT_DIR="/workspace/interpretable-moes-analysis"

# System prepare
apt update -y && apt upgrade -y
apt install -y nano python3.12 python3.12-venv python3.12-dev python3-pip

# Misc - needed for cairosvg export visualizations
apt update && apt-get install -y libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2

# Make/activate a Python 3.12 venv for this project
if [ ! -d "$PROJECT_DIR/.venv" ]; then
  python3.12 -m venv "$PROJECT_DIR/.venv"
else
  echo "Using existing venv at $PROJECT_DIR/.venv"
fi

"$PROJECT_DIR/.venv/bin/python" -m pip install -U pip setuptools wheel

# Upgrade/install Jupyter Lab and widgets within venv
KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}"
if [ ! -f "$KERNEL_DIR/kernel.json" ]; then
  "$PROJECT_DIR/.venv/bin/python" -m pip install -U ipykernel ipywidgets
  "$PROJECT_DIR/.venv/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_NAME"
fi
# Add project path via .pth (idempotent overwrite)
SITE_DIR="$("$PROJECT_DIR/.venv/bin/python"  -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$PROJECT_DIR" > "$SITE_DIR/add_path_analysis.pth"

# Install packages
sh unix_install_packages.sh

# Note: IN vscode, remember to go to command palette -> Jupyter: Select Interpreter to start Jupyter server -> choose the venv
# jupyter kernelspec list
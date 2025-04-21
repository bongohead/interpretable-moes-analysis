#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

VENV_PATH="/workspace/interpretable-moes-analysis/.venv"
REQUIREMENTS_FILE="/workspace/interpretable-moes-analysis/unix_packages.txt"
PROJECT_DIR="/workspace/interpretable-moes-analysis"

apt update -y && apt upgrade -y
apt install -y nano
apt install -y python3.12 python3.12-dev python3.12-venv

if [ -d "$VENV_PATH" ]; then
  echo "Virtual environment '$VENV_PATH' already exists. Skipping creation."
  echo "If you need to recreate it, please remove the directory first: rm -rf $VENV_PATH"
else
  echo "Creating Python 3.12 virtual environment in $VENV_PATH..."
  python3.12 -m venv "$VENV_PATH"
fi

. "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Install packages
. "$PROJECT_DIR/unix_install_packages.sh"

SITE_PACKAGES="$($VENV_PATH/bin/python -c "import site; print(site.getsitepackages()[0])")"
TARGET_PATH_FILE="${SITE_PACKAGES}/add_path_analysis.pth"
echo "Setting up path file: $TARGET_PATH_FILE"
echo "Target path file: $TARGET_PATH_FILE"
echo "$PROJECT_DIR" > "$TARGET_PATH_FILE" # Write the project dir path to the file

echo "--- One-Time Setup Complete ---"
echo "To use the environment in the future, run: source $VENV_PATH/bin/activate"
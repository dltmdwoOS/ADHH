#!/usr/bin/env bash
set -euo pipefail

# Run this from the VILA repository root after activating a fresh VILA-only env.
# Recommended:
#   python3.10 -m venv /venv/vila
#   source /venv/vila/bin/activate
#   bash bash_scripts/setup_chair_base_env.sh

python -m pip install --upgrade 'pip<26' 'setuptools<81' wheel
python -m pip uninstall -y deepspeed flash-attn ring-flash-attn ring_flash_attn ps3-torch ps3_torch || true
python -m pip install -r requirements_chair_base.txt
python -m pip install -e . --no-deps

python - <<'PY'
import pkg_resources
import llava
print('VILA CHAIR base environment import OK')
print('llava module:', llava.__file__)
PY

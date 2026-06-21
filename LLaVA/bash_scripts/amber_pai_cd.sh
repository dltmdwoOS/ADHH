#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PAI_USE_CFG=${PAI_USE_CFG:-true}
exec bash "${script_dir}/amber_pai.sh"

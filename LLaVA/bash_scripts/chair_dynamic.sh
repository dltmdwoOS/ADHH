#!/usr/bin/env bash
set -euo pipefail

exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_layer_list_dynamic_pipeline.sh" "$@"

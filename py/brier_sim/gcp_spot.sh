#!/usr/bin/env bash
# Run from the repository checkout on a CPU spot VM with attached GCS credentials.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo 'Usage: bash py/brier_sim/gcp_spot.sh ENV SHARD SHARDS [--baseline]' >&2
  exit 2
fi
[[ $# -eq 3 || "$4" == --baseline ]] || { echo 'Fourth argument must be --baseline' >&2; exit 2; }
case "$1" in
  pusht|aloha_transfer_cube) ;;
  *) echo 'ENV must be pusht or aloha_transfer_cube' >&2; exit 2 ;;
esac
[[ "$2" =~ ^[0-9]+$ && "$3" =~ ^[1-9][0-9]*$ ]] || {
  echo 'SHARD and SHARDS must be nonnegative and positive integers' >&2; exit 2;
}
(( 10#$2 < 10#$3 )) || { echo 'SHARD must be less than SHARDS' >&2; exit 2; }
[[ "${BRIER_DATA:-}" == gs://* ]] || {
  echo 'BRIER_DATA must point to the GCS data prefix' >&2; exit 2;
}
[[ "$(uname -s)" == Linux ]] || { echo 'The spot runner requires Linux' >&2; exit 2; }
command -v gcloud >/dev/null
command -v python3.11 >/dev/null
[[ -f pyproject.toml && -f py/brier_sim/farm.py ]] || {
  echo 'Run this script from the Brier repository root' >&2; exit 2;
}
python3.11 - <<'PY'
import ctypes
import ctypes.util

for library in ("EGL", "GL", "GLESv2", "OpenGL"):
    path = ctypes.util.find_library(library)
    if path is None:
        raise SystemExit(f"Missing native {library} library; provision libegl1, libgl1, libgles2 and libopengl0 before running the spot worker")
    try:
        ctypes.CDLL(path)
    except OSError as error:
        raise SystemExit(f"Cannot load native {library} library: {error}") from error
PY

python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e .
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_DISABLE_XET=1
exec .venv/bin/python -m brier_sim.farm --env "$1" --shard "$2" --shards "$3" "${@:4}"

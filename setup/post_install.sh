#!/usr/bin/env bash
set -euo pipefail

: "${CONDA_PREFIX:?Activate the conda environment first}"

export TMPDIR="$HOME/tmp_r_exec"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

export RGL_USE_NULL=TRUE
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${CONDA_PREFIX}/share/pkgconfig:${PKG_CONFIG_PATH:-}"

python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

Rscript post_install.R
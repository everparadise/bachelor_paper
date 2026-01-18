#!/usr/bin/env bash
set -e

# 永远以仓库根目录为基准
export TEXMFHOME="$(cd "$(dirname "$0")" && pwd)/texmf"

exec xelatex "$@"

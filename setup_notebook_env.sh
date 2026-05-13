#!/usr/bin/env bash
# Crea .venv en la raíz del repo e instala lo necesario para correr ode_rnn_dramatic.ipynb.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python3 -m venv .venv
.venv/bin/pip install -U pip setuptools wheel
.venv/bin/pip install -r requirements-notebook.txt
echo
echo "Listo. Ejecutar el notebook con el kernel de este venv, o en terminal:"
echo "  source $ROOT/.venv/bin/activate"

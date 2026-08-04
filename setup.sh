#!/usr/bin/env bash
# setup.sh
# --------
# Automatic environment setup for Linux / macOS.
# - Finds a compatible Python interpreter (3.9, 3.10, or 3.11).
# - Creates/updates a virtual environment in ./venv
# - Upgrades pip automatically
# - Installs the pinned requirements.txt
# - Runs env_check.py to confirm everything (including mediapipe) works
#
# Usage:
#     chmod +x setup.sh   # first time only
#     ./setup.sh

set -e

echo "=============================================================="
echo " Hand Mesh Hologram - Automatic Setup (Linux/macOS)"
echo "=============================================================="

PYTHON_BIN=""
for v in 3.11 3.10 3.9; do
    if command -v "python${v}" >/dev/null 2>&1; then
        PYTHON_BIN="python${v}"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[FAIL] Tidak menemukan Python 3.9/3.10/3.11 terinstall."
    echo "Install salah satunya, contoh (Ubuntu/Debian):"
    echo "    sudo apt install python3.11 python3.11-venv"
    echo "atau (macOS + Homebrew):"
    echo "    brew install python@3.11"
    exit 1
fi

echo "[ OK ] Menggunakan $PYTHON_BIN"

if [ ! -d "venv" ]; then
    echo "[..] Membuat virtual environment di ./venv ..."
    "$PYTHON_BIN" -m venv venv
else
    echo "[ OK ] Virtual environment ./venv sudah ada, akan dipakai ulang."
fi

VENV_PY="venv/bin/python"

echo "[..] Meng-upgrade pip ..."
"$VENV_PY" -m pip install --upgrade pip

echo "[..] Menginstall dependencies dari requirements.txt ..."
"$VENV_PY" -m pip install -r requirements.txt

echo "[..] Menjalankan diagnosa environment ..."
"$VENV_PY" env_check.py

echo ""
echo "=============================================================="
echo " Setup selesai. Untuk menjalankan aplikasi:"
echo "     source venv/bin/activate"
echo "     python main.py"
echo "=============================================================="

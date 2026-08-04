"""
env_check.py
------------
Startup diagnostics that verify the Python interpreter AND every third-party
dependency are actually usable BEFORE the app tries to open a webcam / GL
window. Without this, a broken/incompatible install (e.g. MediaPipe on an
unsupported Python version) surfaces as a deep, confusing traceback deep
inside hand_tracker.py. With this, the user gets one clear message and a
concrete fix.

Usage:
    - Imported automatically by main.py (fails fast with a clean message).
    - Can also be run standalone for a full diagnostic + guided auto-fix:

        python env_check.py
"""

import importlib
import platform
import subprocess
import sys

# mp.solutions.hands (the legacy "solutions" API this project uses) is only
# reliably published for these Python versions at the time of writing.
MIN_PY = (3, 9)
MAX_PY = (3, 11)


class EnvironmentIssue(Exception):
    """Raised with a human-readable, actionable message."""
    pass


def _py_version_ok() -> bool:
    v = sys.version_info
    return MIN_PY <= (v.major, v.minor) <= MAX_PY


def check_python_version():
    if not _py_version_ok():
        v = sys.version_info
        raise EnvironmentIssue(
            f"Python {v.major}.{v.minor}.{v.micro} terdeteksi, tapi MediaPipe "
            f"'solutions' API (dipakai untuk deteksi tangan) hanya stabil di "
            f"Python {MIN_PY[0]}.{MIN_PY[1]} - {MAX_PY[0]}.{MAX_PY[1]}.\n\n"
            "Perbaikan tercepat (Windows, pakai Python launcher 'py'):\n"
            "    py -3.11 -m venv venv\n"
            "    venv\\Scripts\\activate\n"
            "    pip install --upgrade pip\n"
            "    pip install -r requirements.txt\n\n"
            "Atau cukup jalankan skrip otomatis:\n"
            "    setup.ps1      (Windows PowerShell)\n"
            "    ./setup.sh     (Linux / macOS)\n"
            "yang akan mencari Python 3.11, membuat virtual environment, dan "
            "menginstall semua dependency dengan versi yang benar secara otomatis."
        )


def check_module(name: str, pip_name: str = None):
    pip_name = pip_name or name
    try:
        importlib.import_module(name)
    except ImportError as e:
        raise EnvironmentIssue(
            f"Package '{pip_name}' belum terinstall atau gagal di-import "
            f"({e.__class__.__name__}: {e}).\n"
            f"Perbaikan:\n    pip install -r requirements.txt"
        ) from e


def check_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as e:
        raise EnvironmentIssue(
            "Package 'mediapipe' belum terinstall. Jalankan:\n"
            "    pip install -r requirements.txt"
        ) from e

    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        raise EnvironmentIssue(
            "mediapipe ter-install tapi 'mediapipe.solutions.hands' tidak "
            "ditemukan (module 'mediapipe' has no attribute 'solutions').\n\n"
            f"  Versi mediapipe : {getattr(mp, '__version__', 'unknown')}\n"
            f"  Lokasi file     : {getattr(mp, '__file__', 'unknown')}\n"
            f"  Python          : {sys.version.split()[0]}\n\n"
            "Penyebab paling umum:\n"
            "  1) Versi Python tidak didukung mediapipe legacy 'solutions' API\n"
            f"     (butuh Python {MIN_PY[0]}.{MIN_PY[1]}-{MAX_PY[0]}.{MAX_PY[1]}).\n"
            "  2) Instalasi mediapipe korup / tidak lengkap.\n\n"
            "Perbaikan cepat:\n"
            "    pip uninstall mediapipe -y\n"
            "    pip cache purge\n"
            "    pip install mediapipe==0.10.9\n\n"
            "Jika Python Anda 3.12 ke atas, buat ulang virtual environment "
            "dengan Python 3.11 (lihat 'py -3.11 -m venv venv' di README), "
            "atau jalankan setup.ps1 / setup.sh untuk perbaikan otomatis.\n\n"
            "Untuk diagnosa lengkap + tawaran perbaikan otomatis, jalankan:\n"
            "    python env_check.py"
        )


def run_all():
    """Raise EnvironmentIssue with a clear message on the first problem
    found. Returns True if every check passes."""
    check_python_version()
    check_module("numpy")
    check_module("cv2", pip_name="opencv-python")
    check_module("scipy")
    check_module("glfw")
    check_module("moderngl")
    check_mediapipe()
    return True


# --------------------------------------------------------------------------- #
# Standalone diagnostic / auto-fix mode: `python env_check.py`
# --------------------------------------------------------------------------- #
def _run(cmd):
    print(f">> {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def _pip_upgrade_self():
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])


def _autofix_mediapipe():
    _run([sys.executable, "-m", "pip", "uninstall", "-y", "mediapipe"])
    _run([sys.executable, "-m", "pip", "install", "mediapipe==0.10.9"])


def main():
    print("=" * 62)
    print("Hand Mesh Hologram - Environment Diagnostic")
    print("=" * 62)
    print(f"Python     : {sys.version}")
    print(f"Platform   : {platform.platform()}")
    print(f"Executable : {sys.executable}")
    print("-" * 62)

    if not _py_version_ok():
        print("[FAIL] Versi Python tidak didukung untuk mediapipe 'solutions' API.")
        try:
            check_python_version()
        except EnvironmentIssue as e:
            print(str(e))
        sys.exit(1)
    print("[ OK ] Versi Python didukung.")

    _pip_upgrade_self()

    modules = [
        ("numpy", "numpy"), ("cv2", "opencv-python"), ("scipy", "scipy"),
        ("glfw", "glfw"), ("moderngl", "moderngl"),
    ]
    all_ok = True
    for mod, pip_name in modules:
        try:
            check_module(mod, pip_name)
            print(f"[ OK ] {pip_name}")
        except EnvironmentIssue as e:
            print(f"[FAIL] {pip_name}\n{e}")
            all_ok = False

    try:
        check_mediapipe()
        print("[ OK ] mediapipe (mp.solutions.hands tersedia)")
    except EnvironmentIssue as e:
        print(f"[FAIL] mediapipe\n{e}")
        print()
        try:
            answer = input("Coba perbaikan otomatis (reinstall mediapipe==0.10.9)? [y/N] ").strip().lower()
        except EOFError:
            answer = "n"
        if answer == "y":
            _autofix_mediapipe()
            importlib.invalidate_caches()
            try:
                check_mediapipe()
                print("[ OK ] mediapipe sekarang berfungsi.")
            except EnvironmentIssue as e2:
                print(f"[FAIL] Masih gagal setelah reinstall:\n{e2}")
                all_ok = False
        else:
            all_ok = False

    print("-" * 62)
    if all_ok:
        print("Semua pengecekan LULUS - siap menjalankan: python main.py")
    else:
        print("Ada masalah yang perlu diperbaiki sebelum menjalankan main.py.")
        sys.exit(1)


if __name__ == "__main__":
    main()

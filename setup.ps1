# setup.ps1
# ----------
# Automatic environment setup for Windows.
# - Finds a compatible Python interpreter (3.9, 3.10, or 3.11) via the
#   'py' launcher, preferring the newest supported version.
# - Creates/updates a virtual environment in .\venv
# - Upgrades pip automatically
# - Installs the pinned requirements.txt
# - Runs env_check.py to confirm everything (including mediapipe) works
#
# Usage (PowerShell):
#     .\setup.ps1
# If script execution is blocked, run once as admin:
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"

function Find-CompatiblePython {
    $candidates = @("3.11", "3.10", "3.9")
    foreach ($v in $candidates) {
        $test = & py "-$v" -c "print('ok')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $test -eq "ok") {
            return $v
        }
    }
    return $null
}

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " Hand Mesh Hologram - Automatic Setup (Windows)" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $pyLauncher) {
    Write-Host "[FAIL] Python launcher 'py' tidak ditemukan." -ForegroundColor Red
    Write-Host "Install Python 3.11 dari https://www.python.org/downloads/ dan centang 'Add python.exe to PATH', lalu jalankan skrip ini lagi."
    exit 1
}

$ver = Find-CompatiblePython
if (-not $ver) {
    Write-Host "[FAIL] Tidak menemukan Python 3.9/3.10/3.11 terinstall." -ForegroundColor Red
    Write-Host "Download Python 3.11 di: https://www.python.org/downloads/release/python-3119/"
    Write-Host "Saat instalasi, pastikan opsi 'Add python.exe to PATH' dicentang."
    exit 1
}
Write-Host "[ OK ] Menggunakan Python $ver (via 'py -$ver')" -ForegroundColor Green

if (-not (Test-Path ".\venv")) {
    Write-Host "[..] Membuat virtual environment di .\venv ..."
    & py "-$ver" -m venv venv
} else {
    Write-Host "[ OK ] Virtual environment .\venv sudah ada, akan dipakai ulang."
}

$venvPython = ".\venv\Scripts\python.exe"

Write-Host "[..] Meng-upgrade pip ..."
& $venvPython -m pip install --upgrade pip

Write-Host "[..] Menginstall dependencies dari requirements.txt ..."
& $venvPython -m pip install -r requirements.txt

Write-Host "[..] Menjalankan diagnosa environment ..."
& $venvPython env_check.py

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " Setup selesai. Untuk menjalankan aplikasi:" -ForegroundColor Cyan
Write-Host "     venv\Scripts\activate"
Write-Host "     python main.py"
Write-Host "==============================================================" -ForegroundColor Cyan

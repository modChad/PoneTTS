@echo off
REM ============================================================
REM  MOSS-TTS Voice Server - Windows setup (run once)
REM  Builds a Python environment, installs everything, and
REM  downloads the voice model (~12 GB). Takes a while.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   MOSS-TTS Voice Server - first-time setup (Windows)
echo ============================================================
echo.

REM --- find Python ---
set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo ERROR: Python was not found.
  echo Install Python 3.10, 3.11 or 3.12 from https://www.python.org/downloads/
  echo and TICK the box "Add Python to PATH" during install, then run this again.
  echo.
  pause
  exit /b 1
)

REM --- venv ---
if not exist venv (
  echo [1/5] Creating Python environment...
  %PY% -m venv venv || ( echo Failed to create venv. & pause & exit /b 1 )
)
set "VPY=venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip wheel

REM --- PyTorch (CUDA 12.8 build) ---
echo [2/5] Installing PyTorch (CUDA 12.8)... this is a big download
"%VPY%" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchaudio==2.11.0 ^
  || ( echo PyTorch install failed. & pause & exit /b 1 )

REM --- the rest of the deps ---
echo [3/5] Installing server dependencies...
"%VPY%" -m pip install -r requirements.txt || ( echo Dependency install failed. & pause & exit /b 1 )

REM --- model weights ---
echo [4/5] Downloading the voice model from HuggingFace (~12 GB, one time)...
"%VPY%" download_model.py || ( echo Model download failed. & pause & exit /b 1 )

REM --- sanity check the model bundle downloaded (runtime is used as-is) ---
if not exist "release\moss_tts_torchopt_runner_bundle\portable_tts_runtime.py" ( echo Model bundle missing - download may have failed. & pause & exit /b 1 )

REM --- v2.5 voice patch (pronunciation fine-tune, merged into the model once) ---
echo [5/5] Applying the v2.5 voice patch (a few minutes)...
"%VPY%" apply_voice_patch.py || ( echo Voice patch failed. & pause & exit /b 1 )

REM Mark setup COMPLETE - only now is the model fully downloaded.
REM The launcher checks this marker so it never starts with no model.
echo done> ".setup_done"

echo.
echo ============================================================
echo   Setup complete! You can now start the server.
echo ============================================================
endlocal

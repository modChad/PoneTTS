@echo off
REM Start the MOSS-TTS voice server on Windows (native, no WSL needed).
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Server is not set up yet. Run setup.bat first.
  pause
  exit /b 1
)
REM Let PyTorch hand VRAM back to the driver (keeps long lines from running out of memory).
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
venv\Scripts\python.exe server.py --host 0.0.0.0 --port 8000 %*

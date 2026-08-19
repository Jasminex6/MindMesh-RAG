"""Unified Dev Launcher for Pediatric Asthma CDS Application.

Launches both FastAPI REST Backend (port 8000) and Vite React Frontend (port 5173).
Usage:
    uv run python dev.py
    or
    python dev.py
"""

from __future__ import annotations

import os
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"


def main():
    print("======================================================================")
    print("  Pediatric Asthma Clinical Decision Support System — Dev Launcher")
    print("======================================================================")

    # 1. Spawn FastAPI Backend Subprocess
    print("[1/2] Launching FastAPI REST Backend Server on http://127.0.0.1:8000 ...")
    python_exe = sys.executable
    backend_cmd = [python_exe, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=str(ROOT),
    )

    time.sleep(3)

    # 2. Spawn Vite React Frontend Subprocess
    print("[2/2] Launching Vite React Frontend Dev Server on http://localhost:5173 ...")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_process = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(FRONTEND_DIR),
    )

    print("\n----------------------------------------------------------------------")
    print("  ✅ All services launched successfully!")
    print("  🌐 React Frontend Web UI:  http://localhost:5173")
    print("  ⚡ FastAPI Backend API:    http://localhost:8000")
    print("  (Press Ctrl+C to stop all dev servers)")
    print("----------------------------------------------------------------------\n")

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nStopping dev servers...")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)


if __name__ == "__main__":
    main()

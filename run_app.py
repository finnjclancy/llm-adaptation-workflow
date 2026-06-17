"""
Cross-platform bootstrap for the LLM Adaptation frontend.

This is the brain behind the double-click launchers (start_windows.bat /
start_mac.command). It:

  1. Creates an isolated virtual environment (.venv) on first run
  2. Installs the app's dependencies into it (once)
  3. Launches the Streamlit app in the default browser

Run directly if you prefer:  python run_app.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "app" / "requirements.txt"
APP = ROOT / "app" / "app.py"
INSTALLED_MARKER = VENV / ".installed"


def venv_python() -> Path:
    if os.name == "nt":  # Windows
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    subprocess.check_call(cmd, **kwargs)


def main():
    print("=" * 64)
    print("  LLM Adaptation Workflow — local app")
    print("=" * 64)

    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required. Please install it from python.org.")

    # 1. Create the virtual environment if needed
    if not venv_python().exists():
        print("\nFirst-time setup: creating a virtual environment (.venv)…")
        run([sys.executable, "-m", "venv", str(VENV)])

    py = str(venv_python())

    # 2. Install dependencies once (marker file avoids re-installing every launch)
    if not INSTALLED_MARKER.exists():
        print("\nInstalling dependencies — this can take a few minutes the first time…")
        run([py, "-m", "pip", "install", "--upgrade", "pip"])
        run([py, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        INSTALLED_MARKER.write_text("ok")
    else:
        print("\nDependencies already installed — skipping setup.")

    # 3. Launch Streamlit
    print("\nStarting the app… your browser should open shortly.")
    print("Leave this window open while you use the app. Close it to stop.\n")
    env = dict(os.environ, STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    run(
        [py, "-m", "streamlit", "run", str(APP), "--server.headless=false"],
        env=env,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nSomething went wrong (exit code {e.returncode}).")
        print("Please share this window's output for help.")
        if os.name == "nt":
            input("\nPress Enter to close…")
        sys.exit(e.returncode)

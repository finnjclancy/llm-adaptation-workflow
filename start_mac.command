#!/bin/bash
# ====================================================================
#  LLM Adaptation Workflow - macOS launcher
#  Double-click this file to start the app. No terminal commands needed.
#  (If macOS blocks it the first time: right-click > Open > Open.)
# ====================================================================

cd "$(dirname "$0")" || exit 1

if command -v python3 >/dev/null 2>&1; then
    python3 run_app.py
else
    echo ""
    echo "  Python 3 was not found on this Mac."
    echo ""
    echo "  Install it from https://www.python.org/downloads/macos/"
    echo "  (or run 'xcode-select --install'), then double-click this file again."
    echo ""
    read -r -p "Press Enter to close…"
fi

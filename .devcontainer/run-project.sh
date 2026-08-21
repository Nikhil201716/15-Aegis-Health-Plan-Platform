#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Runs this project end to end inside a GitHub Codespace.
# Invoked automatically by .devcontainer/devcontainer.json on attach.
# ---------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

B=$'\033[1m'; D=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; C=$'\033[36m'; R=$'\033[0m'

echo ""
echo "${B}${C}================================================================${R}"
echo "${B}  Aegis Health Plan Intelligence${R}"
echo "${D}  Nikhil Sinha  ·  running the full pipeline${R}"
echo "${B}${C}================================================================${R}"
echo ""
if ! command -v ollama >/dev/null 2>&1; then
  echo "${Y}Note:${R} this project has optional local-LLM stages. Ollama is not"
  echo "installed, so those stages will be skipped or fall back to their"
  echo "deterministic control arm. Everything else runs normally."
  echo "${D}To enable them:  curl -fsSL https://ollama.com/install.sh | sh${R}"
  echo "${D}                 ollama serve & && ollama pull qwen2.5:0.5b${R}"
  echo ""
fi

echo "${D}Every number this prints is written to reports/ and is the same"
echo "data the portfolio site quotes.${R}"
echo ""

START=$SECONDS
echo "${B}>${R} python run_all.py"
python run_all.py
STATUS=$?

echo ""
echo "${B}${C}----------------------------------------------------------------${R}"
if [ $STATUS -eq 0 ]; then
  echo "${B}${G}  Pipeline finished in $((SECONDS-START))s${R}"
else
  echo "${B}${Y}  Pipeline exited with status $STATUS${R}"
  echo "${D}  Some stages are SUPPOSED to fail - see the project README.${R}"
fi
echo ""
echo "${B}  Results${R}"
if [ -d reports ]; then
  ls -1 reports/ 2>/dev/null | sed 's/^/    reports\//'
fi
echo ""
echo "${B}  Explore it${R}"
echo "    ${C}python app/api/main.py${R}"
echo "${D}    Codespaces will offer to open the forwarded port in a browser.${R}"
echo ""
echo "${B}  The 60+ page technical notebook for this project is in docs/${R}"
echo "${B}${C}----------------------------------------------------------------${R}"
echo ""

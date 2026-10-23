#!/bin/zsh
# validate → build → commit → push。どれかが失敗したら commit しない。
# 使い方: scripts/ship.sh "コミットメッセージ"
set -e
set -o pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/validate.py | tail -2
.venv/bin/python scripts/build.py | tail -3
git add -A
git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main 2>&1 | tail -1 || { sleep 3; git push origin main 2>&1 | tail -1; }

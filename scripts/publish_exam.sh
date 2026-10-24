#!/bin/sh
# 見直しの終わったセットを公開する：検算をやり直し → 見直し済みを記録 → 配分表 → validate・build → commit・push
# 使い方: sh scripts/publish_exam.sh 4 "見直しで3問を修正"
set -e
N=$1; NN=$(printf %02d "$N")
.venv/bin/python verify/exam_set$NN.py | tail -1
.venv/bin/python scripts/exam_review.py --set "$N"
.venv/bin/python scripts/exam_blueprint.py > /dev/null
.venv/bin/python scripts/build.py | grep "模擬試験\|validate"
git add -A
git commit -qm "set$NN: 模擬試験 第${N}回（25問）を公開（検算・見直し済み。$2）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q origin main || git push -q origin main
git log --oneline -1

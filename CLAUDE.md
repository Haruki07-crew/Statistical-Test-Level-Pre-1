# 統計検定準1級 用語辞典プロジェクト

このフォルダは、統計検定準1級の学習用「用語解説Webページ」を作るプロジェクトです。
公式参考書『統計学実践ワークブック』（全32章）の章立てに沿って用語を整理し、
PC・iPad・スマホで見られる1枚のHTML（`docs/index.html`）として出力します。

## 最初に必ず読むもの（セッション開始時・再開時・会話の要約後は毎回）

1. `PROGRESS.md` … どこまで終わったか、次に何をするか（存在しなければ初回）
2. `KICKOFF.md` … 作業手順と完了条件
3. 必要に応じて `spec/DESIGN.md`（設計）、`spec/CONTENT_GUIDE.md`（書き方）、`spec/TERMS.md`（用語リスト）

## 自律動作のルール（最重要）

- 利用者は寝ていて返事ができない。**質問・確認で止まらない。** 迷ったら最も妥当な案を選んで進む。
- 判断したことは `DECISIONS.md` に1行ずつ追記する（日時・判断・理由）。
- エラーが出たら原因を調べて直す。3回直せなければ、その用語・機能を `PROGRESS.md` の「保留」に書いて次へ進む。
- 1つの章が終わるたびに `PROGRESS.md` を更新し、git commit する。途中で止まっても再開できるようにするため。
- 作業を「完了」と言う前に、必ず検証コマンドを実行して結果を会話に表示する。

## コマンド

Pythonは必ずプロジェクトの仮想環境を使う（macOSのシステムPythonに直接 pip install しない）。

```bash
python3 -m venv .venv                                   # 初回のみ
.venv/bin/pip install -r requirements.txt               # 初回・依存追加時
.venv/bin/python scripts/validate.py                    # 用語ファイルの検査
.venv/bin/python scripts/build.py                       # docs/index.html を生成
.venv/bin/python verify/ch05.py                         # 章ごとの数値検算（例）
node --check src/app.js                                 # nodeがあればJSの構文確認
```

## 編集してよい範囲

- 自由に編集: `content/`, `src/`, `scripts/`, `verify/`, `docs/`, `PROGRESS.md`, `DECISIONS.md`, `REPORT.md`, `spec/research/`
- 原則編集しない: `spec/DESIGN.md`, `spec/CONTENT_GUIDE.md`, `spec/TERMS.md`, `CLAUDE.md`, `KICKOFF.md`
  （設計を変える必要があれば変えてよいが、理由を `DECISIONS.md` に書く）
- `content/plan.yaml` と `PROGRESS.md` を書き換えるのはメインの作業者だけ。サブエージェントは自分の担当章の `content/chNN/` だけを編集する。

## 著作権のルール

- 『統計学実践ワークブック』の文章・問題・図をそのまま写さない。解説はすべてオリジナルで書く。
- Webで調べるのは「どの用語を扱うか」「定義に誤りがないか」を確かめるためだけ。他サイトの文章をコピーしない。
- 参考にしたURLは `spec/research/chNN.md` に記録する。

## 品質のルール

- 解説は日本語。初学者にもわかる言葉で、準1級レベルの厳密さは保つ。
- 期待値・分散・母関数・検定統計量・具体例の計算結果は、`verify/` のPythonスクリプトで数値検算してから「verified」にする。
- パラメータの取り方が流儀で分かれるもの（幾何分布の台、ガンマ分布の尺度/レート など）は、採用した定義を本文に明記し、もう一方の流儀も一言触れる。

## git

- リモート `origin` は `https://github.com/Haruki07-crew/Statistical-Test-Level-Pre-1.git`、ブランチは `main`。
  `git remote -v` で設定されていなければ `git remote add origin https://github.com/Haruki07-crew/Statistical-Test-Level-Pre-1.git` を実行する。
- 公開ページは GitHub Pages（`main` ブランチの `/docs`）。URL は https://haruki07-crew.github.io/Statistical-Test-Level-Pre-1/
  → push する前に必ず `build.py` を実行し、`docs/index.html` を最新にしておく。
- 1章終わるごとに `git add -A && git commit -m "chNN: 用語を追加"` → `git push origin main` する。
  フェーズの区切り（P1の土台完成、P2の計画完成など）でも commit・push する。
- push が失敗したら1回だけやり直す。それでも失敗したら `DECISIONS.md` に記録し、以後も commit は続け、push は次の章でまた試す（push の失敗で作業を止めない）。
- force push・履歴の書き換え・`git reset --hard`・`rm -rf` はしない。
- `.gitignore` に `.venv/`、`__pycache__/`、`.DS_Store` を入れる。

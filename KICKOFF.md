# 作業手順書（KICKOFF）

あなたはこのプロジェクトのメインの作業者（司令塔）です。
統計学の講師としての知識と、フロントエンドエンジニアとしての技術を使って、
`spec/DESIGN.md` の設計どおりに用語辞典を完成させてください。

**利用者は寝ています。質問せず、止まらず、最後まで進めてください。**
ルールは `CLAUDE.md` にあります。再開したときは、まず `PROGRESS.md` を読んで続きから始めてください。

---

## フェーズ P0：準備（最初の1回だけ）

1. `spec/` の4つのファイル（DESIGN、CONTENT_GUIDE、TERMS、sample/bayes-theorem.md）を読む。
2. git の状態を確認する（`git status`、`git remote -v`）。リポジトリがなければ `git init -b main`、ブランチ名が main でなければ `git branch -M main`、origin がなければ CLAUDE.md の URL で追加する。`.gitignore` に `.venv/`、`__pycache__/`、`.DS_Store` を書く。
3. `requirements.txt`（pyyaml, markdown, numpy, scipy, sympy）を作り、`python3 -m venv .venv` → `.venv/bin/pip install -r requirements.txt`。
4. `node --version` で node があるか確認し、結果を `DECISIONS.md` に書く（ない場合は JS のテストを省略する）。
5. `PROGRESS.md` を設計書9章の形式で作る。
6. commit し、`git push origin main` する（push できるかをここで確認しておく）。

## フェーズ P1：アプリの土台

1. `scripts/validate.py`、`scripts/build.py` を設計書どおりに作る。
2. `src/template.html`、`app.css`、`app.js`、`dist.js`、`charts.js` を作る。
   - 画面（3列／2列／1列の切り替え）、章ナビ、一覧、解説、検索、テーマ切り替え、ハッシュでの遷移、「覚えた」チェック。
   - dist.js は設計書4.5の全分布。charts.js は `pmf` `pdf` `cdf` `pdf-cdf` `survival-hazard` `scatter` と、カスタム図 `bayes-area` `joint-marginal` まで。他のカスタム図は、その図を使う章を書くときに足す。
3. `scripts/make_dist_reference.py` と `verify/test_dist.mjs` を作り、node があれば dist.js をテストする。
4. 見本 `spec/sample/bayes-theorem.md` を `content/ch01/` にコピーし、仮の `plan.yaml` を作ってビルドが通ることを確認する。
5. commit する。

## フェーズ P2：全32章の用語計画

1. `spec/TERMS.md` の第1〜6章の指定用語を `plan.yaml` に入れる（id は英語名をもとに付ける）。
2. 第7〜32章は、TERMS.md の手順でWeb検索して用語を選び、重要度 A/B/C を付けて `plan.yaml` に入れる。
   - `term-researcher` サブエージェントに、数章ずつまとめて調べさせてよい（最大3つ同時）。
   - 調べた内容と参考URLは `spec/research/chNN.md` に保存させる。
3. 全章で重複や id の衝突がないか確認する。見本の `related` にある id（`prior-distribution`、`posterior-distribution` など）が計画に含まれているか確認する。
4. commit する。

この時点で一度ビルドし、「全用語が準備中として並んだページ」ができていることを確認する。

## フェーズ P3：第1〜6章を書く

章ごとに次を繰り返す（第1章から順に）。

1. `term-writer` サブエージェントにその章の用語を書かせる（担当は `content/chNN/` だけ）。
   独立した章なら最大3章まで同時に進めてよい。
2. 書き終わったら `validate.py` を実行し、エラーを直す。
3. `math-checker` サブエージェントに `verify/chNN.py` を作らせ、数値検算させる。間違いがあれば直させる。
4. 必要なカスタム図があれば `charts.js` に追加する（charts.js を触るのはメインの作業者だけ）。
5. `plan.yaml` の status を更新し、`build.py` を実行する。
6. `PROGRESS.md` を更新し、commit（リモートがあれば push）する。

第1〜6章が終わったら、ページ全体を見直す：
- 狭い幅（375px）・iPad幅（768px）・PC幅（1280px）でレイアウトが崩れないか、CSSを読み直して確認する。
- 検索で「ガウス」「がんま」「MGF」などがヒットするか、build 後のデータで確認する。

## フェーズ P4：第7〜32章の重要度 A と B

第7章から順に、P3 と同じ手順で、重要度 A と B の用語だけを書く。
**先に全章の A・B を終わらせる**ことを優先する（朝の時点で、どの章にも主要な用語がある状態にするため）。

## フェーズ P5：第7〜32章の重要度 C

第7章から順に、残りの C の用語を書く。

## フェーズ P6：仕上げ

1. `related` で、まだ書かれていない用語を指しているものを確認し、可能なら書くか、リンク先を変える。
2. `validate.py`（エラー0）、`build.py`、node があれば `node --check` と dist.js のテストを実行し、結果を表示する。
3. `REPORT.md` を書く（下記）。
4. commit（リモートがあれば push）する。

---

## 完了条件

次のすべてを満たしたら完了です。

1. `content/plan.yaml` の全用語の status が `verified`（直せなかったものは `on_hold` とし、理由を PROGRESS.md に記載）。
2. `.venv/bin/python scripts/validate.py` がエラー0で終わり、その出力を会話に表示した。
3. `.venv/bin/python scripts/build.py` が成功し、`docs/index.html` ができ、その出力（章別の用語数・サイズ）を会話に表示した。
4. `REPORT.md` を書いた。

## REPORT.md に書くこと（日本語、短く）

- 完成した章と章ごとの用語数（A/B/C 別）
- 終わらなかった用語・保留した用語と理由
- 章の割り当てを変えた用語と理由
- 内容に自信がない箇所、利用者が確認したほうがいい箇所
- 自分で判断して決めたこと（DECISIONS.md の要約）
- 参考にしたサイトの一覧（spec/research/ のまとめ）
- ページの開き方（PCでは `docs/index.html` をブラウザで開く。iPad・スマホは GitHub Pages か、Claudeアプリでアーティファクト化）

## 時間と使用量について

全章を一晩で終えられない可能性が高いです。それでかまいません。
大事なのは、**止まった時点までの成果が必ず `docs/index.html` に反映され、commit されていること**と、
**PROGRESS.md を見れば次に何をすればよいかわかること**です。

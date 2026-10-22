# はじめに（あなたが読むファイル）

このフォルダを Claude Code に渡すと、寝ている間に「統計検定準1級 用語辞典」のページを作り、
GitHub リポジトリ <https://github.com/Haruki07-crew/Statistical-Test-Level-Pre-1> に1章ずつ commit・push してくれます。
ページは GitHub Pages で、PC・iPad・スマホから見られます。

- リポジトリ：https://github.com/Haruki07-crew/Statistical-Test-Level-Pre-1
- 公開ページ：https://haruki07-crew.github.io/Statistical-Test-Level-Pre-1/

## フォルダの中身

| ファイル | 役割 |
|---|---|
| `README_はじめに.md` | このファイル（あなた用） |
| `CLAUDE.md` | Claude Code が毎回自動で読むルール（質問せずに進める、著作権、検算、git） |
| `KICKOFF.md` | 作業手順と完了条件 |
| `spec/DESIGN.md` | 設計書（フォルダ構成、データ形式、画面、検索、グラフ、検証、公開方法） |
| `spec/CONTENT_GUIDE.md` | 解説の書き方（8つの節、分量、口調） |
| `spec/TERMS.md` | 用語リスト（第1〜6章はあなたの指定、第7〜32章は候補） |
| `spec/sample/bayes-theorem.md` | 解説の完成見本（品質の基準） |
| `docs/index.html` | 公開ページ（今は「準備中」。Claude Code が作業するたびに置き換わる） |
| `.claude/settings.json` | 確認なしで実行してよいコマンドの一覧 |
| `.claude/agents/` | 手分けして作業する3人の助手（用語調べ・執筆・検算） |

`.claude` は Finder では見えない隠しフォルダですが、ちゃんと入っています。

---

## 寝る前にやること（5分）

ターミナルを開いて、上から順にコピーして実行してください。

### 1. リポジトリを手元に持ってくる

```bash
cd ~
git clone https://github.com/Haruki07-crew/Statistical-Test-Level-Pre-1.git
```

### 2. 指示書セットをリポジトリの中にコピーする

zip をダウンロードフォルダで解凍した場合（`~/Downloads/stats-glossary` ができている場合）:

```bash
cp -R ~/Downloads/stats-glossary/. ~/Statistical-Test-Level-Pre-1/
cd ~/Statistical-Test-Level-Pre-1
ls -a
```

`CLAUDE.md` `KICKOFF.md` `spec` `docs` `.claude` が並んでいればOKです。

### 3. 最初の push を自分でする（push できるかの確認）

```bash
git add -A
git commit -m "指示書セットを追加"
git branch -M main
git push -u origin main
```

- 成功すれば、寝ている間も Claude Code が同じように push できます。
- ユーザー名・パスワードを聞かれて失敗する場合は、`gh auth login` を実行して GitHub にログインしてから、もう一度 `git push -u origin main` を実行してください（`gh` がなければ `brew install gh`）。
- どうしても push できなくても作業は進みます。その場合 Claude Code は手元で commit だけ続けるので、朝に `git push origin main` すれば反映されます。

### 4. GitHub Pages を有効にする（ブラウザで1回だけ）

1. https://github.com/Haruki07-crew/Statistical-Test-Level-Pre-1/settings/pages を開く
2. 「Build and deployment」の Source を **Deploy from a branch** にする
3. Branch を **main**、フォルダを **/docs** にして **Save**
4. 数分後に https://haruki07-crew.github.io/Statistical-Test-Level-Pre-1/ を開くと「準備中」のページが出ます

リポジトリが Private の場合、無料プランでは Pages が使えません。Settings → General の一番下で Public にしてください（中身はオリジナルの解説と学習用のコードだけです）。

### 5. Mac を眠らせない設定

- 電源アダプタをつなぐ
- ノートパソコンのふたは**開けたまま**にする（閉じるとスリープします。画面は暗くなっても大丈夫です）
- 念のため、システム設定 → バッテリー → オプション で「電源アダプタ接続時にディスプレイがオフのときに自動でスリープさせない」をオンにする（項目名は macOS のバージョンで少し違います）

### 6. Claude Code を起動する

```bash
cd ~/Statistical-Test-Level-Pre-1
caffeinate -dims claude --permission-mode auto
```

- `caffeinate` は、Claude Code が動いている間 Mac をスリープさせない Mac 標準のコマンドです。Claude Code が終わると自動で解除されます。
- 「このフォルダを信頼しますか？」と聞かれたら、信頼する（Yes）を選びます。
- `--permission-mode auto` は、安全確認を別のAIが代わりに行い、あなたへの確認なしで作業を進めるモードです。

### 7. 作業を始める

起動した Claude Code に、次の1行をそのまま貼り付けて Enter を押します。

```
/goal KICKOFF.md と CLAUDE.md を読み、KICKOFF.md の手順どおりにフェーズP0から作業を進める（PROGRESS.md があれば続きから）。1章終わるごとに commit して origin の main に push する。完了条件：content/plan.yaml の全用語の status が verified または on_hold になり、.venv/bin/python scripts/validate.py がエラー0で終わり、.venv/bin/python scripts/build.py が成功し、両方の出力を会話に表示し、REPORT.md を書いて push し終えていること。利用者は寝ているので質問せず、判断は DECISIONS.md に記録して進める。
```

`/goal` は、完了条件を満たすまで Claude Code が自分で作業を続ける機能です。

### 8. 動き出したのを確認してから寝る

ファイルを読んだり `.venv` を作ったりし始めれば大丈夫です。**ターミナルのウィンドウは閉じずに**そのままにしてください（閉じると Claude Code も止まります）。

---

## 朝にやること

- iPad・スマホ・PC で https://haruki07-crew.github.io/Statistical-Test-Level-Pre-1/ を開く
  （push のたびに数分で更新されます。古い表示のままなら再読み込み）
- リポジトリの `PROGRESS.md` … どこまで終わったか
- `REPORT.md` … 全部終わっていれば最終報告（自信のない箇所、確認してほしい箇所）
- `DECISIONS.md` … Claude が自分で判断したこと

## 途中で止まっていたら

一晩で全32章（約400語）が終わらないことは十分ありえます。止まっていても、push 済みの分はページに反映されています。

- **ターミナルが開いたまま**：使用量の上限で止まった場合、上限がリセットされると自動で再開します。「press enter to continue」と出ていたら Enter を押します。
- **ターミナルを閉じてしまった・再起動した**：次を実行すると、前回の会話と目標を引き継いで再開します。

```bash
cd ~/Statistical-Test-Level-Pre-1
caffeinate -dims claude --continue --permission-mode auto
```

再開後に作業が始まらない場合は、手順7の `/goal` の行をもう一度貼り付けてください。PROGRESS.md を見て続きから進めます。

## 知っておいてほしいこと

- auto モードの安全確認で一部の操作が止められた場合、Claude は別の方法を探して進みます。それでも進めない作業は PROGRESS.md の「保留」に残ります。
- 解説はすべて Claude が書いたオリジナルで、数値は検算していますが、最終的には参考書と照らし合わせて使ってください。

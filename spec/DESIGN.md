# 設計書：統計検定準1級 用語辞典

## 1. 目的と前提

| 項目 | 内容 |
|---|---|
| 利用者 | 統計検定準1級の受験者1名（本人） |
| 使う端末 | PC（Mac）、iPad（縦・横）、スマホ |
| 使う場面 | 参考書を読みながら引く／電車でスマホで復習する／試験前に章ごとに見直す |
| 成果物 | 1枚で完結するHTML `docs/index.html`（外部はCDNのスクリプトとGoogle Fontsのみ） |
| 公開方法 | GitHub Pages（`main` ブランチの `/docs`）、または Claudeアプリにアップロードしてアーティファクト化 |
| 規模 | 全32章、約350〜450語を想定 |

**1枚のHTMLにする理由**：GitHub Pagesでもアーティファクトでも、そのまま置くだけで動く。
ファイルを分けると、アーティファクトでは別ファイルの読み込みが面倒になる。

## 2. フォルダ構成

```
Statistical-Test-Level-Pre-1/
├── CLAUDE.md                 作業者向けルール（Claude Codeが自動で読む）
├── KICKOFF.md                作業手順と完了条件
├── PROGRESS.md               進捗（作業者が作成・更新）
├── DECISIONS.md              自分で判断したことの記録（作業者が作成）
├── REPORT.md                 最終報告（作業者が作成）
├── requirements.txt          pyyaml, markdown, numpy, scipy, sympy
├── spec/
│   ├── DESIGN.md             この設計書
│   ├── CONTENT_GUIDE.md      解説の書き方
│   ├── TERMS.md              章一覧・指定用語・7章以降の候補
│   ├── sample/bayes-theorem.md   完成見本（品質の基準）
│   └── research/chNN.md      章ごとの調査メモと参考URL（作業者が作成）
├── content/
│   ├── plan.yaml             全用語の計画と状態（唯一の正）
│   └── ch01/ … ch32/         1用語＝1ファイル（Markdown＋YAMLフロントマター）
├── src/
│   ├── template.html         ページの骨組み（{{CSS}} {{JS}} {{DATA}} を差し込む）
│   ├── app.css
│   ├── app.js                画面・検索・ルーティング
│   ├── dist.js               分布の密度・確率・累積分布の計算ライブラリ
│   └── charts.js             グラフの種類ごとの描画関数と、カスタム図
├── scripts/
│   ├── validate.py           用語ファイルの検査
│   ├── build.py              docs/index.html の生成
│   └── make_dist_reference.py  dist.js テスト用の参照値をscipyで生成
├── verify/
│   ├── chNN.py               章ごとの数値検算
│   ├── dist_reference.json
│   ├── test_dist.mjs         nodeがあれば dist.js を参照値と比較
│   └── REPORT.md             検算結果の一覧
└── docs/
    └── index.html            成果物（GitHub Pagesの公開対象）
```

## 3. 技術選定

| 役割 | 採用 | 理由 |
|---|---|---|
| 画面 | 素のHTML/CSS/JavaScript（フレームワークなし） | 依存を増やさず、1枚のHTMLに収めやすい |
| 数式 | **MathJax 3 の SVG出力**（`https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js`） | フォントファイルやCSSを外部から読まないので、アーティファクトの制限下でも、GitHub Pagesでも同じように表示できる。KaTeXはCSSとフォントの外部読み込みが必要で、アーティファクトでは崩れる |
| グラフ | Chart.js 4（`https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js`）＋ Canvas 直書き | 折れ線・棒はChart.js、面積図・樹形図などはCanvasで描く |
| 本文 | Markdownで書き、ビルド時にPythonでHTMLへ変換 | 書き手（AI）が数式を書きやすい。JSONより `\` のエスケープ事故が起きない |
| ビルド・検査・検算 | Python 3（pyyaml, markdown, numpy, scipy, sympy） | 検算にscipyが必要なので言語を揃える |
| フォント | Google Fonts（本文：BIZ UDPGothic または Noto Sans JP、数字：等幅系） | 日本語の可読性。フォールバックも必ず指定 |

外部読み込みは上の2つのスクリプトとGoogle Fontsだけにする（それ以外のCSS・画像・データはすべてHTMLに埋め込む）。
バージョンは必ず固定する。

MathJaxの設定（スクリプト読み込みより前に置く）:

```html
<script>
window.MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$'], ['\\[', '\\]']] },
  svg: { fontCache: 'global' },
  startup: { typeset: false }   // 用語を開いたときに、その部分だけ typesetPromise([el]) する
};
</script>
```

## 4. データ設計

### 4.1 計画ファイル `content/plan.yaml`（全用語の一覧と状態）

```yaml
chapters:
  - no: 1
    title: 事象と確率
    terms:
      - id: bayes-theorem
        title: ベイズの定理
        importance: A        # A=必須・頻出 / B=重要 / C=補足
        status: verified     # planned → written → verified（保留は on_hold）
  - no: 7
    title: 極限定理、漸近理論
    terms:
      - id: central-limit-theorem
        title: 中心極限定理
        importance: A
        status: planned
```

- 画面はこのファイルも読み込み、**未作成の用語も「準備中」として一覧に薄く表示する**。朝に見たとき全体像と進み具合がわかるようにするため。
- `related` のリンク先は plan.yaml に載っていれば有効とみなす（まだ書かれていなくてもよい）。
- 任意の `see: <id>`（統合先）：内容がほかの用語と重なるため重要度Cに下げた用語に付ける。用語ページの上部に「この内容は「○○」でまとめて解説しています」と表示する（ページは削除しない）。

### 4.2 用語ファイル `content/chNN/<id>.md`

1用語＝1ファイル。ファイル名は `id` と同じ。

```markdown
---
id: gamma-distribution            # 英小文字・数字・ハイフン。全体で一意
title: ガンマ分布                  # 見出し
reading: がんまぶんぷ               # ひらがな読み（検索・並び順用）
en: Gamma distribution            # 英語名（検索用）
aliases: [Γ分布]                   # 別名（検索でヒットさせる）
chapter: 6
importance: A
tags: [連続型分布, 再生性]
related: [exponential-distribution, chi-squared-distribution, poisson-distribution, beta-distribution]
charts:
  - type: pdf
    dist: gamma
    params: { shape: 2, scale: 1 }
    sliders:
      - { param: shape, label: "形状 α", min: 0.5, max: 10, step: 0.5 }
      - { param: scale, label: "尺度 β", min: 0.5, max: 5, step: 0.5 }
    x: { min: 0, max: 20 }
    caption: α を大きくすると山が右へ移動し、左右対称に近づく（中心極限定理）
---

## 一言でいうと
…

## 定義・数式
…
```

フロントマターの項目:

| 項目 | 必須 | 内容 |
|---|---|---|
| id | ○ | 英小文字・数字・ハイフン。plan.yaml と一致 |
| title | ○ | 用語名 |
| reading | ○ | ひらがな読み |
| en | ○ | 英語名（なければ空文字） |
| aliases | | 別名・略語（例：MLE、ガウス分布） |
| chapter | ○ | 1〜32 |
| importance | ○ | A / B / C |
| tags | | 自由なタグ |
| related | ○ | 関連用語の id（2〜6個） |
| charts | A,Bは○ | グラフ指定（4.4参照）。Cは省略可 |

本文の見出しは次の8つを**この順番・この表記**で使う（validate.py が検査する）。
内容のルールは `CONTENT_GUIDE.md` を参照。

1. `## 一言でいうと`
2. `## 定義・数式`
3. `## 数式の読み解き`
4. `## 具体例`
5. `## 現場での使われ方`
6. `## グラフ・イメージ`（charts がない用語は省略可）
7. `## 試験のポイント`
8. `## よくある間違い`

「関連用語」は本文に書かず、フロントマターの `related` から画面が自動で並べる。

### 4.3 数式の書き方

- インライン `$ ... $`、ディスプレイ `$$ ... $$`（前後に空行を入れる）。
- 金額などで `$` を文字として使うときは `\$` と書く。
- 行列は `\begin{pmatrix} … \end{pmatrix}`、場合分けは `\begin{cases} … \end{cases}`、複数行は `\begin{aligned} … \end{aligned}` を `$$` の中で使う。
- MathJax 標準で使えないマクロ（`\bm` など）は使わない。太字ベクトルは `\boldsymbol{x}`。

### 4.4 グラフ指定 `charts`

1用語に0〜3個。グラフは「グラフ・イメージ」節の本文の下に描く。

| type | 用途 | 主な項目 |
|---|---|---|
| `pmf` | 離散分布の確率関数（棒） | `dist`, `params`, `sliders`, `x`, `overlay`（例：`norm-approx` で正規近似を重ねる） |
| `pdf` | 連続分布の密度（曲線） | `dist`, `params`, `sliders`, `x`, `compare`（固定の比較曲線のリスト） |
| `cdf` | 累積分布関数 | `dist`, `params`, `sliders`, `x`, `quantile`（分位点を示す p） |
| `pdf-cdf` | 密度と累積分布を上下に並べ、スライダーで x を動かすと「左側の面積＝CDFの値」が連動 | `dist`, `params`, `x` |
| `survival-hazard` | 生存関数とハザード関数を並べる | `dist`（expon / weibull / gamma / lognorm）, `params`, `sliders` |
| `scatter` | 乱数で作った散布図（相関・回帰・擬似相関など） | `model`（`bvnorm` / `spurious` / `regression`）, `params`, `sliders`, `n`, `seed` |
| `custom` | 上で表せない図 | `id`（charts.js に登録した関数名）, `params`, `caption` |

すべての type に `caption`（図の読み方を1〜2文）を付ける。

`dist` の名前とパラメータ名（dist.js と揃える）:

| dist | パラメータ | 注意 |
|---|---|---|
| bernoulli | p | |
| binom | n, p | |
| hypergeom | N, K, n | 母集団N、成功K、抽出n |
| poisson | lambda | |
| geom | p, start | start=0（失敗回数）または 1（試行回数）。本文と一致させる |
| nbinom | r, p | 成功r回までの失敗回数 |
| unif | a, b | |
| norm | mu, sigma | |
| lognorm | mu, sigma | log X ~ N(mu, sigma²) |
| expon | rate | 平均 1/rate |
| gamma | shape, scale | 本文でレート表記にも触れる |
| beta | a, b | |
| cauchy | loc, scale | |
| weibull | shape, scale | |
| logistic | loc, scale | |
| chi2 | df | |
| t | df | |
| f | df1, df2 | |
| ncx2 | df, nc | 非心度 λ |
| nct | df, nc | |
| ncf | df1, df2, nc | |
| normmix | w, mu1, sigma1, mu2, sigma2 | 2成分の混合正規 |

カスタム図（charts.js に実装し、`custom` の `id` で呼ぶ）の例。必要に応じて増やしてよい:

`bayes-area`（ベイズの面積図）, `joint-marginal`（同時分布表と周辺分布）, `clt-sim`（中心極限定理のシミュレーション）,
`lln-path`（大数の法則の標本平均の推移）, `ci-coverage`（信頼区間を100本描いて母数を含む割合）, `power-curve`（検出力曲線と2種類の過誤）,
`lr-test`（尤度比）, `markov-chain`（状態遷移図と定常分布への収束）, `poisson-process`（到着の様子）, `random-walk`,
`regression-residual`（回帰直線と残差、てこ比・クックの距離）, `logistic-curve`, `pca-axes`（主成分の軸）, `dendrogram`,
`kmeans-steps`, `acf-plot`（コレログラム）, `ar-sim`, `km-curve`（カプラン・マイヤー曲線）, `ridge-lasso-path`, `bias-variance`,
`prior-posterior`（事前・尤度・事後の重ね描き）, `mcmc-trace`, `inverse-transform`, `rejection-sampling`, `bootstrap-hist`, `box-cox`, `mosaic`（分割表）

乱数を使う図は、シード付きの乱数（例：mulberry32）で毎回同じ図にし、「もう一度」ボタンでシードを変える。

### 4.5 dist.js の実装方針

- 対数ガンマ関数（Lanczos近似）、誤差関数、正則化不完全ガンマ関数・不完全ベータ関数を自前で実装し、各分布の pdf/pmf/cdf を作る。
- 非心カイ二乗・非心F：ポアソン重みの級数和（重みが十分小さくなるまで）。
- 非心t：定義式の数値積分（台形則やシンプソン則で十分）。
- `scripts/make_dist_reference.py` で scipy の値を `verify/dist_reference.json` に書き出し、node があれば `verify/test_dist.mjs` で相対誤差1e-6程度（非心分布は1e-4程度）以内か確認する。

## 5. ビルド（scripts/build.py）

1. `content/plan.yaml` と `content/ch*/*.md` を読む。
2. 各用語の本文で、数式部分（`$$…$$`、次に `$…$`、`\$` は除外）を一時的な記号に置き換えてから Markdown→HTML 変換し、変換後に数式を元に戻す（`&` `<` `>` はHTMLエスケープ）。Markdownが `_` や `*` を数式の中で解釈して壊すのを防ぐため。
3. 本文から検索用のプレーンテキストを作る（HTMLタグと数式記号を落とす）。
4. `src/template.html` に `app.css`、`dist.js`＋`charts.js`＋`app.js`、データ（`<script type="application/json" id="data">`）を差し込み、`docs/index.html` を書き出す。
5. 最後に「用語数（章別）／未作成数／ファイルサイズ」を表示する。サイズは5MB以下を目安にする。

ビルドの前に必ず validate.py を実行し、エラーがあればビルドを止める。

## 6. 画面設計

### 6.1 レイアウト

| 幅 | 想定端末 | 構成 |
|---|---|---|
| 1024px以上 | PC、iPad横 | 2列：左＝一覧（部のタブ＋章の開閉、約340px）、右＝解説（残りすべて）。画面の切り替えはヘッダー |
| 600〜1023px | iPad縦 | 2列：左＝一覧（約300px）、右＝解説。画面の切り替えは画面下部 |
| 600px未満 | スマホ | 1列：一覧 → 解説へ切り替え。解説画面の上部に「← 一覧」ボタン。戻ったときはタブ・章の開閉・スクロール位置を保つ |

- ヘッダーは上部に固定し、検索欄を常に表示する（スマホでも1タップで検索できる）。
- 本文は1行あたり全角35〜40字程度に収める。
- タップできる要素は44px四方以上。スライダーは指で操作しやすい太さにする。
- iPhoneのノッチ等に配慮し `env(safe-area-inset-*)` で余白を取る。
- 印刷用の装飾は不要。

- PC・iPad（600px以上）では、ヘッダー左端のボタン（PCは `\` キーでも）で左の一覧をたたみ、解説を全幅で読める。状態は `localStorage` の `jd-list-collapsed` に保存する。検索したときと章を開く操作（`#chNN`）では自動でひらく。

### 6.2 一覧（部のタブ＋章の開閉）

章ナビと用語一覧は1列にまとめる。列の上部（スクロールしても固定）に次を置く:

1. 部のタブを横一列に並べる。入りきらない幅では横スクロール。部の分け方はこのページ独自の便宜的なもの:

| 部 | 章 |
|---|---|
| 確率と分布 | 1〜6 |
| 漸近理論と推測 | 7〜13 |
| 確率過程 | 14〜15 |
| 回帰分析 | 16〜19 |
| 実験計画と標本調査 | 20〜21 |
| 多変量解析 | 22〜26 |
| 時系列・分割表・不完全データ | 27〜29 |
| モデル選択・ベイズ・シミュレーション | 30〜32 |
| 統計的因果推論 | 33（増補版ワークブックの第33章。2027年からの出題範囲表に対応） |

2. タブの下に「すべて開く」「すべて閉じる」（選んだ部の章が対象）と「覚えた用語を隠す」の切り替え。

その下に、選んだ部の章を縦に並べる。

- 章の見出し（例：「02 確率分布と母関数 13/13」＝章番号・章名・作成済み数／予定数）を押すと、その章の用語一覧が開閉する。開いている章は ▼、閉じている章は ▶ の印を付ける。
- 今見ている用語の章は自動で開き（タブもその部に切り替える）、その用語の行を強調する。
- 選んだタブと開いている章は `localStorage` に保存し、次に開いたときも同じ状態にする。
- 検索中はタブを無視して、全章からの検索結果を1つの一覧で出す（タブと開閉ボタンは隠す）。検索を消すと元のタブ表示に戻る。

### 6.3 用語一覧の行

- 項目：用語名、英語名、重要度（A/B/C を小さな印で）、「覚えた」チェック。ひらがなの読みは画面に出さない（データには残し、検索には使う）。
- 並び順：章の中では plan.yaml の順（学習の流れ順）。検索中はスコア順。
- 未作成の用語は薄く表示し「準備中」と出す（タップしても解説は開かない）。

### 6.4 解説画面

上から順に:

1. 章番号・章名、用語名、英語名、別名、重要度（読みは表示しない）
2. 節へのジャンプ用の小さなボタン列（一言／定義／具体例／現場／グラフ／試験／間違い）
3. 本文8節（見出しは CONTENT_GUIDE の8つ）
4. グラフは「グラフ・イメージ」節の中、説明文の下。スライダーと現在値の表示付き
5. 関連用語（ボタン。未作成は薄く）
6. 同じ章の「← 前の用語／次の用語 →」

数式が横にはみ出す場合は、その数式ブロックだけ横スクロールさせる（ページ全体は横スクロールさせない）。

### 6.5 状態の保存と URL

- 用語ごとに URL のハッシュ（`#gamma-distribution`）で開けるようにする。章は `#ch06`（その部のタブを選び、章を開いて見出しまでスクロール）。読み込み時にハッシュを見て開く。
- 「覚えた」チェック、最後に見た用語、テーマ、選んだ部のタブ、開いている章は `localStorage` に保存する。読み書きは必ず try/catch で囲み、使えなくても動くようにする。

### 6.6 見た目

- 方向性：学習用の道具として、落ち着いて長時間読める紙面。装飾は控えめにし、数式とグラフを主役にする。
- 色はCSS変数で定義し、ライト／ダークの両方を作る（`prefers-color-scheme` に従い、ヘッダーの切り替えボタンでも変更可）。グラフの線・文字の色もCSS変数から取り、テーマ切り替え時に再描画する。
- 重要度の印や「準備中」は色だけに頼らず、文字や形でも区別する。
- 絵文字は使わない。

### 6.7 画面の切り替え（辞書／問題／復習／記録）

- PC（1024px以上）はヘッダーの右側、iPad・スマホは画面下部の固定ナビに「ホーム・辞書・問題・復習・記録」を置く。復習には期限の来た問題数のバッジを出す。
- URL：`#quiz`（問題演習）、`#review`（今日の復習）、`#stats`（学習記録と設定）、`#formulas`（公式まとめ。辞書の一部として一覧の横に出す）。辞書以外の画面は一覧を隠して1列で使う。
- ホーム（`#home`。開いたときと、ヘッダーの題名・切り替えの「ホーム」から）：
  - 今日の状況（復習の問題数・今日の回答数・連続学習日数・試験までの日数）
  - 今日のメニュー（①復習 ②用語を読む：続きから＋学習の順番で次に読む重要度Aの3語 ③問題・用語カード）
  - はじめての人へ：学習の順番（6ステップ：確率と分布の土台 → 推定と検定 → いろいろな検定 → 回帰と実験計画・標本調査 → 多変量解析 → 発展分野。各ステップに説明・まず読む用語・重要度Aを覚えた数と正答率・用語カード・重要度Aの問題5問・一覧で見る）
  - 勉強のコツ、使い方（折りたたみ：辞書／問題／用語カード・公式まとめ／記録・データの移し方／ホーム画面に追加／キーボード）
  - 今日の1問、最近見た用語、ブックマーク
- 辞書で用語を選んでいないときの解説欄は、「用語を選んでください」と、続きから・ホーム・公式まとめ・ランダムへのボタンだけにする。

### 6.8 問題

- 形式・書き方・検算は `spec/QUESTIONS.md`。用語ページの最後（メモの次）に「この用語の問題」を出す。ジャンプ欄にも「問題」「メモ」を足す。
- 問題カード：4択（選択肢は毎回シャッフル）、数値入力（全角・分数 a/b・% も読む。answer ± tolerance で正誤）、正誤、導出（途中式を1つずつ表示→最終結果→自己採点）。どれもヒントと途中式付きの解説を持つ。
- 問題演習：出題範囲（全体・部・章・苦手・未回答・ブックマーク）と出題数（5／10／20）を選ぶ。最後に正答率、間違えた問題、関係する用語へのボタン、「間違えた問題だけもう一度」を出す。
- 今日の復習：間違えたままの問題と、期限が来た問題（正解が続いた回数 s に応じて 1・3・7・14・30 日後）を最大20問。
- 苦手な用語：回答のある用語のうち、正答率が60%未満か、間違えたままの問題がある用語。
- 模擬試験モード：問題演習の設定で「時間を計る」を選ぶと、1問3分の制限時間で残り時間を表示し、時間切れになったらそこまでの回答で結果を出す。結果にはかかった時間も出す。
- 問題にも★を付けられる。出題範囲「ブックマーク」は、★を付けた問題と、★を付けた用語の問題。
- 用語カード（`#cards`）：問題演習と同じ範囲・枚数で用語名を出し、「めくる」で「一言でいうと」を表示、「覚えた」で覚えたチェックを付ける。最後に「まだ」の用語だけもう一度。
- キーボード（問題演習中）：`1`〜`4`（`a`〜`d`）で選択、`Enter` で次へ、`h` でヒント。

### 6.9 学習データ（localStorage `jd-data`）

```json
{ "v": 1,
  "memo":  { "用語id": 覚えた時刻 },  "bm": { "用語id": 付けた時刻 },
  "notes": { "用語id": { "x": "メモ本文", "t": 更新時刻 } },
  "recent": ["最近見た用語id", ...],            // 最大30
  "ans":   { "問題id": { "n": 回答数, "c": 正解数, "s": 連続正解, "t": 最後の回答時刻, "d": 次に出す時刻, "r": 直前の正誤 1/0 } },
  "days":  { "YYYY-MM-DD": { "a": 回答数, "c": 正解数, "v": 見た用語数 } },
  "searches": ["最近の検索語", ...],            // 最大8
  "font": "s" | "m" | "l",
  "exam": "YYYY-MM-DD",                         // 試験日（空文字なら未設定）
  "qbm":  { "問題id": ★を付けた時刻 } }
```

- 読み込み時に型を確かめ、合わない値は捨てて使える部分だけ残す（壊れたデータでも落ちない）。旧版の `jd-memo`・`jd-last` は初回だけ引き継ぐ。テーマ（`jd-theme`）、タブ（`jd-tab`）、開いている章（`jd-open`）、問題演習の設定（`jd-quizcfg`）は別のキー。
- 書き出し：`{"app": "jun1-dictionary", "v": 1, "exported": ISO時刻, "data": 上の形式}` の JSON ファイル。読み込み時に「統合」（覚えた・★は和集合、メモと回答は新しい方、日ごとの記録は大きい方）か「上書き」を選ぶ。

### 6.10 そのほか

- 文字サイズ（小・中・大）は `html[data-font]` で 14.5／16／18px を切り替える（本文は rem で指定）。
- キーボード：`/` 検索、`Esc` 検索解除、`j`／`k` 次・前の用語（章をまたぐ）、`b` ブックマーク。
- 初回表示を速くするため、数式（MathJax）とグラフは IntersectionObserver で画面に近づいた節だけ描く。
- ホーム画面に追加：`docs/manifest.json`・アイコン・`docs/sw.js`（build.py が `src/pwa/` からコピー）。service worker は https のときだけ登録し、ページはネット優先・つながらなければ保存した版、CDN とフォントは保存した版を優先して返す（2回目以降はオフラインでも開ける）。
- 指で操作する端末（`pointer: coarse`）では、押せる部品を 44px 以上にする。
- 公式まとめは `content/formulas.md`（`##` ごとに節。検算は `verify/formulas.py`）。

## 7. 検索設計

- 対象：用語名、読み、英語名、別名、タグ、本文のプレーンテキスト。
- 正規化：NFKC 正規化 → 小文字化 → カタカナをひらがなに → 空白・中黒・ハイフン類を除去。クエリも同じ処理をする。
  例：「ｶﾞﾝﾏ」「ガンマ」「がんま」「Gamma」がすべて同じ用語に当たる。
- スコア：名前・別名の完全一致 100 ＞ 前方一致 80 ＞ 部分一致 60 ＞ 読み・英語名の部分一致 50 ＞ タグ 30 ＞ 本文 10。
- 入力中に絞り込む（100ms程度の遅延で十分）。章の絞り込みと同時に使える。
- 本文で当たった場合は、一覧に該当箇所の前後20字程度を表示し、一致部分を強調する。
- 0件のときは「見つかりませんでした」と、近い候補（部分一致の緩い検索）を表示する。
- キーボード：`/` で検索欄へ、`Esc` で検索解除（PC向け）。

## 8. 検証設計

| 検査 | 方法 | タイミング |
|---|---|---|
| 用語ファイルの形式 | `scripts/validate.py`：必須項目、id の一意性と plan.yaml との一致、別名の重複、`related` のリンク先が plan.yaml にあるか、見出し8つの順番、`$` の対応、charts の type・dist・パラメータ名・スライダーの param が有効か | 用語を書くたび・ビルド前 |
| 数式・数値 | `verify/chNN.py`：期待値・分散・母関数の微分・具体例の計算を numpy/scipy/sympy で確認し、結果を `verify/REPORT.md` に追記 | 章ごと |
| 分布ライブラリ | `verify/test_dist.mjs` で dist.js と scipy の参照値を比較 | dist.js を変更したとき（node がある場合） |
| JavaScript | `node --check` で構文確認（node がある場合） | ビルド後 |

validate.py は、エラー（ビルドを止める）と警告（報告だけ）を分けて表示し、エラーがあれば終了コード1を返す。
`--only <id>` で1用語だけ、`--chapter N` で1章だけ検査できるようにする（サブエージェントが使う）。

## 9. 進捗管理と再開

`PROGRESS.md` の形式:

```markdown
# 進捗
最終更新: 2026-10-23 03:12

## 次にやること
- 第9章：用語の執筆（plan.yaml の planned 12件）

## フェーズ
- [x] P0 準備
- [x] P1 アプリの土台
- [x] P2 全章の用語計画
- [x] P3 第1〜6章
- [ ] P4 第7〜32章（A・B）… 第8章まで完了
- [ ] P5 第7〜32章（C）
- [ ] P6 仕上げ

## 章ごとの状況
| 章 | 予定 | 作成 | 検算済 | 備考 |
|---|---|---|---|---|
| 1 | 3 | 3 | 3 | |

## 保留
- （直せなかったもの、理由）
```

再開時は PROGRESS.md と plan.yaml の status を見て、続きから始める。

## 10. 公開

- **GitHub Pages**（メイン）：リポジトリ https://github.com/Haruki07-crew/Statistical-Test-Level-Pre-1 の Settings → Pages で「Deploy from a branch」→ `main` / `/docs`（利用者が設定済み）。公開URLは https://haruki07-crew.github.io/Statistical-Test-Level-Pre-1/ 。push するたびに数分で更新される。
  - GitHub Pages で Jekyll の処理を避けるため、`docs/.nojekyll`（空ファイル）を置く。
  - ページ内のリンクはすべて相対パスかハッシュにする（サイトがサブパス `/Statistical-Test-Level-Pre-1/` の下で配信されるため）。
- **アーティファクト**：`docs/index.html` をClaudeアプリのチャットに添付し、「このHTMLをそのままアーティファクトとして公開して」と頼む。外部読み込みを上記のCDNとGoogle Fontsだけにしているのは、この方法でも動かすため。

## 11. あとから用語を追加する方法

1. `content/plan.yaml` の該当章に1行追加する。
2. `content/chNN/<id>.md` を CONTENT_GUIDE に従って書く。
3. `validate.py` → `build.py` を実行する。

## 12. 数式表示の仕組み（2026-10-24）
- build.py は数式を `<m>$…$</m>`（インライン）と `<div class="math-display">$$…$$</div>`（別行）にして埋め込む。
- app.js の `showMath(el)` が唯一の入口。画面に HTML を入れたら、その部分に対して呼ぶ（呼び忘れは MutationObserver が拾う）。
  - MathJax の準備完了（`mathjax-ready` または `MathJax.startup.promise`）を待ってから組む。待つ間は列に積む。
  - `typesetPromise` は 1 本の Promise の列で順番に実行する（同時に走らせない）。画面に付く前の要素は付くまで待つ。
  - 隠れている部分（ヒント・解説・練習モードの答え）も表示時に組む。数式には遅延描画（IntersectionObserver）を使わない。
  - 読み込み失敗（script の onerror・startup の失敗・25 秒の時間切れ）では、数式を「［数式］」にし、「数式を読み込めませんでした。再読み込みしてください」と出す。あとで読み込めたら元に戻して組む。
- `\boldsymbol` の拡張は MathJax の設定で最初に読み込む。sw.js は MathJax 本体・boldsymbol・Chart.js をキャッシュし、ページは no-cache で取り直す。キャッシュ名は index.html のハッシュで、新しい版の有効化時に古いキャッシュを消す。
- 検査：`scripts/check_math.py`（validate.py から。$ の対応・数式外の LaTeX・括弧の対応。MJ_DIR があれば mathjax-full で全数式を変換）、`verify/math_webkit.mjs`（WebKit・820×1180 で主要画面を開き、LaTeX の文字が残っていないか・読み込み失敗時の表示）。

## 13. 模擬試験（#exam）
- データ：`content/exam_sets/setNN.yaml`（形式は spec/EXAM.md）。数値は «名前» で書き、`generator/setNN.py` の `values()` の値を差し込む。
- 品質チェック（3 段階）：自動検査 `scripts/check_exam.py` → 検算 `verify/exam_setNN.py`（`verify/examcheck.py`、結果は verify/exam_results/）→ 見直し（math-checker、記録は verify/exam_review/setNN.md と `scripts/exam_review.py`）。検算と見直しは小問の内容のハッシュで記録し、build.py は 3 段階すべてに通った 25 問がそろったセットだけを `DATA.exams` に入れる。
- 画面：`#exam`（一覧・得点の推移・苦手な章・受験履歴）、`#exam/N`（モード選択）、`#exam/N/run`（解答）、`#exam/N/result/時刻`（結果）。
  - 本番モード：残り時間・問題番号の一覧から移動・「あとで見直す」・最後にまとめて採点・時間切れで自動採点。
  - 練習モード：1 問ごとに「答え合わせ」で正誤と解説。
  - 結果：得点と合格ライン、章ごとの正答率、間違えた問題（状況設定・選択肢・正解・解説・関係する用語）、ロードマップへの案内。
- 保存：学習データ jd-data の `mock = { hist: [{s 回, sc 得点, c 正解数, n, t 時刻, d 所要, m 'r'|'p', ch {章: [正解, 出題]}, a 解答, tu 時間切れ, v 版}], cur: {s, m, i, left 残り, el 経過, a, f 印, k 答え合わせ済み, st, v 版} }`。読み込み時に型を確かめ、書き出し・読み込み（統合）にも含める。
- 版：build.py が各セットに `v`（小問 id・選択肢・正解のハッシュ 10 桁）を付ける。解答は小問ごとに「選んだ番号」で保存するので、選択肢を並べ替えると番号がずれる。起動時に途中経過の `v` が今のセットと違えば途中経過を捨て、結果の画面では `v` の違う過去の結果に「問題が更新された」と注意書きを出す（点数は当時のまま）。
- 確認：`verify/exam_ui_webkit.mjs`（375/768/820/1180/1280px で本番・再開・採点・練習・ロードマップ）。

## 14. ロードマップ（#roadmap）
- データ：`content/roadmap.yaml`（`scripts/make_roadmap.py` が生成、`scripts/check_roadmap.py` が検査）。各用語に段階（1〜4）・優先度（0〜100）・前提・目安時間・理由。根拠は spec/research/roadmap.md。
- 画面：段階ごとの達成率（「覚えた」の割合）、用語ごとの覚えた状況と問題の正答率、「次にやる5語」（未習得で前提を満たすものを段階→優先度の順）。

## 15. 2級のページ（2026-10-24）

利用者の依頼で、統計検定2級の用語辞典・問題演習・模擬試験・ロードマップ・公式まとめを、同じサイトの別ページ `docs/g2/index.html`（URL は `…/g2/`）として追加した（理由は DECISIONS.md の 2026-10-24 19:54〜）。

- **級の切り替え**：スクリプトは環境変数 `JD_LEVEL`（`pre1`＝準1級・既定、`g2`＝2級）で置き場所を切り替える（`scripts/level.py`）。2級は `content/g2/`・`verify/g2/`・`docs/g2/`。`scripts/build.py` は両方の級を作り、`docs/sw.js` は両方のページを保存する（版は両方のページのハッシュ）。
- **級ごとの設定**：`content/g2/site.yaml`（名前・見出し・説明文・部の分け方・学習の順番・模擬試験の説明・記録のキー）を `DATA.site` として渡す。準1級は省略された項目に app.js の既定値を使うので、表示は今までと同じ。両方のホームに、もう一方の級へのリンクを出す。2級は見出しの色（`--accent`）を青系にして見分けやすくした（`:root[data-level="g2"]`）。
- **学習記録**：localStorage のキーを級ごとに分ける（準1級 `jd-`、2級 `jd2-`）。テーマ（`jd-theme`）だけは共通。
- **用語**：出題範囲表の大項目に沿った 15 章・136 語（`content/g2/plan.yaml`）。最初は 32 語だけを2級向けに書き、ほかの 104 語は準1級の解説を共通で使っていた（plan の `from: pre1`。build.py が準1級のファイルを読み、章と重要度を2級のものに置き換え、関連用語を2級にあるものだけに絞り、「準1級と共通」の注意書きと `content/g2/notes.yaml` の「2級で押さえること」を出す）。2026-10-25 に 104 語すべてを2級向けに書き直したので、いまは 136 語とも `content/g2/chNN/` にあり、`from: pre1` の用語はない（仕組みは残してある）。書き直した用語の本文の数値・図の値は `verify/g2/rewrite_terms.py` で検算する。2級の用語の本文の字数の目安は準1級の 8 割（validate.py）。
- **問題演習**：2級の問題（`content/g2/questions/`、検算は `verify/g2/questions_ch*.py`）。2026-10-25 時点で 352 問。重要度 A の用語は 3 問以上、B は 2 問、C は 1 問。共通の用語があるときは、準1級の問題のうち level が「基本」で検算済みのものも出す（いまは共通の用語がないので出ない）。
- **模擬試験**：35 問・90 分・すべて5肢選択（`content/g2/exam_sets/`）。分野の配分・難易度・定義問題の上限は `scripts/exam_lib.py` の `LEVEL == "g2"` の設定。数値は問題ファイルに直接書き、`verify/g2/exam_setNN.py` で別の方法で検算する。見直しの記録は `verify/g2/exam_review/setNN.md`。正解をいったん①に置いて書き、`scripts/rebalance_choices.py` で①〜⑤に 7 問ずつならす。画面のセットの枠は `site.yaml` の `exam.sets`（15）。新しいセットを作るときは、全セットの論点（topic）と照合して、数値替えや同じ要点の繰り返しを作り直す（見直し記録に書く）。
- **ロードマップ**：`scripts/make_roadmap_g2.py` が `content/g2/roadmap.yaml` を作る（136 語・4 段階）。
- **公式まとめ**：`content/g2/formulas.md`（検算は `verify/g2/formulas.py`）。
- **検査**：`JD_LEVEL=g2 .venv/bin/python scripts/validate.py`（build.py が自動で実行）、画面は `verify/g2_smoke.mjs`（2級のホーム・用語・模擬試験の採点・記録のキー・準1級との行き来。共通の用語が 1 つもないときは、注意書きの確認を飛ばす）。
- **図の追加・修正（2026-10-25）**：`power-curve` に左片側（`alt: less`）を追加した。同時確率の表の図（`joint-marginal`）で選んだ行の class を `jm-sel` に変えた（汎用の `.sel { display: flex }` とぶつかって行が崩れていたため。両方の級）。表の数字がスマホで途中で折り返さないよう、`.table-wrap` のセルは `overflow-wrap: normal` にした。

## 16. 統計学の教科書（学部1年〜修士2年）のページ（2026-10-26）

利用者の依頼で、2級・準1級に加えて 3 つ目のページ「統計学の教科書」を `docs/tb/index.html`（URL は `…/tb/`）として追加した（理由は DECISIONS.md の 2026-10-26 19:27〜）。

- **級の切り替え**：`JD_LEVEL=tb`（`scripts/level.py`）。置き場所は `content/tb/`・`verify/tb/`・`docs/tb/`。`scripts/build.py` は 3 つのページを作り、`docs/sw.js` は 3 つのページを保存する。
- **構成**：学年ごとの 6 部・57 章（`content/tb/plan.yaml`。`scripts/make_plan_tb.py` が作る。直接書き換えない）。学部1・2年（第1〜15章）は2級の解説、学部2〜4年（第16〜44章）は準1級の解説を、学ぶ順に並べ直して共通で使う（plan の `from: g2 / pre1`。同じ用語は先に学ぶ学年の側に 1 回だけ置き、準1級の `see`（補足のページ）は入れない）。修士1・2年（第45〜57章・52 語）は教科書のために書いた解説（`content/tb/chNN/`）。章ごとの「ねらいと前提」（plan の `lead`）を、一覧で章を開いたときに出す。
- **共通の用語**：章と重要度は教科書のものに置き換え、関連用語は教科書にあるものだけを残す。元の辞典を示す注記（`site.yaml` の `shared_note`）を出す。問題は元の級の問題をすべての難易度で出す（`shared_q_levels`）。学部の用語 58 語には、修士の用語への「大学院での発展」リンクを出す（`content/tb/advanced.yaml`）。
- **教科書で書いた用語**：見出しは「試験のポイント」の代わりに「学習のポイント」（validate.py の HEADINGS を級で切り替え、build.py は同じ節として扱う。節ボタンは「要点」）。字数の目安は準1級と同じ。数値の検算は `verify/tb/chNN.py`（共通部品 `verify/tb/tbcheck.py`。本文の数値と図の値をページから読み、正確な分布・数値積分・総当たり・シミュレーションで計算し直す。シミュレーションの図は同じ乱数で計算し直して一致を確かめる）。問題は `content/tb/questions/ch45〜57.yaml`（134 問、検算は `verify/tb/questions_m1.py`・`questions_m2.py`）。公式まとめは `content/tb/formulas.md`（修士の範囲、検算 `verify/tb/formulas.py`）。
- **画面**：`site.yaml` の `textbook: true` で、ホームの「本番に向けて」（模擬試験・ロードマップ）と下のメニューの模試・ロードマップを出さない。勉強のコツは `tips`、見出しは `h1`、ほかの級へのリンクは `others`（3 つの級のホームで互いにリンク）。色は茶系（`:root[data-level="tb"]`）、学習記録のキーは `jdt-`。
- **検査**：validate.py は、共通で使うほかの級の用語の題名・別名とも照合する（別名の重なりはエラー、題名がほかの用語の別名と同じときは警告）。画面は `verify/tb_smoke.mjs`（ホーム・学年のタブ・章のねらい・共通の用語の注記と章・教科書の用語の数式とはみ出し・問題・公式まとめ・発展のリンク・記録のキー・3 つの級の行き来）。

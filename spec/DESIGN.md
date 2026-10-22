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
| 1024px以上 | PC、iPad横 | 3列：左＝章ナビ（固定）、中＝用語一覧、右＝解説 |
| 600〜1023px | iPad縦 | 2列：左＝用語一覧、右＝解説。章ナビはヘッダーのボタンで開く引き出し |
| 600px未満 | スマホ | 1列：一覧 → 解説へ切り替え。解説画面の上部に「← 一覧」ボタン。章ナビは引き出し |

- ヘッダーは上部に固定し、検索欄を常に表示する（スマホでも1タップで検索できる）。
- 本文は1行あたり全角35〜40字程度に収める。
- タップできる要素は44px四方以上。スライダーは指で操作しやすい太さにする。
- iPhoneのノッチ等に配慮し `env(safe-area-inset-*)` で余白を取る。
- 印刷用の装飾は不要。

### 6.2 章ナビ

32章を「部」でまとめて折りたたみ表示する（部の分け方はこのページ独自の便宜的なもの）:

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

各章に「作成済み数／予定数」を小さく表示する。「すべて」「覚えた用語を隠す」の切り替えも置く。

### 6.3 用語一覧

- 項目：用語名、読み、重要度（A/B/C を小さな印で）、「覚えた」チェック。
- 並び順：章の中では plan.yaml の順（学習の流れ順）。検索中はスコア順。
- 未作成の用語は薄く表示し「準備中」と出す（タップしても解説は開かない）。

### 6.4 解説画面

上から順に:

1. 章番号・章名、用語名、読み、英語名、別名、重要度
2. 節へのジャンプ用の小さなボタン列（一言／定義／具体例／現場／グラフ／試験／間違い）
3. 本文8節（見出しは CONTENT_GUIDE の8つ）
4. グラフは「グラフ・イメージ」節の中、説明文の下。スライダーと現在値の表示付き
5. 関連用語（ボタン。未作成は薄く）
6. 同じ章の「← 前の用語／次の用語 →」

数式が横にはみ出す場合は、その数式ブロックだけ横スクロールさせる（ページ全体は横スクロールさせない）。

### 6.5 状態の保存と URL

- 用語ごとに URL のハッシュ（`#gamma-distribution`）で開けるようにする。章は `#ch06`。読み込み時にハッシュを見て開く。
- 「覚えた」チェック、最後に見た用語、テーマは `localStorage` に保存する。読み書きは必ず try/catch で囲み、使えなくても動くようにする。

### 6.6 見た目

- 方向性：学習用の道具として、落ち着いて長時間読める紙面。装飾は控えめにし、数式とグラフを主役にする。
- 色はCSS変数で定義し、ライト／ダークの両方を作る（`prefers-color-scheme` に従い、ヘッダーの切り替えボタンでも変更可）。グラフの線・文字の色もCSS変数から取り、テーマ切り替え時に再描画する。
- 重要度の印や「準備中」は色だけに頼らず、文字や形でも区別する。
- 絵文字は使わない。

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

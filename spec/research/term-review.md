# 用語選定の見直し（term-review）

調査日: 2026-10-23。content/plan.yaml（494語）と各用語の aliases を、公式の出題範囲表と学習まとめ記事に照らし合わせて確認した。
まとめ記事の文章は写さず、どの用語を扱うかを判断するためだけに使った。

## 重要な前提（メインの作業者が判断すること）

- **出題範囲表が 2027-01-01 から改訂される。** 新しい表では「概収束・確率収束・分布収束・連続写像定理・スルツキーの補題」「Box-Cox変換」「バイアス・バリアンス分解」「母分散の比」「イエーツの補正」「決定係数・自由度調整済み決定係数・L2正則化・Mallows' Cp」「Cookの距離」「ポアソン回帰」「カプラン・マイヤー推定量」「BIC」などが明記された。回帰診断法にあった「系列相関・DW比」は時系列解析に移っている。また、大項目「統計的因果推論」が新しく加わった。
- **『統計学実践ワークブック』の増補版（2026-08-20 発売）に第33章「統計的因果推論」が加わった。** 第1〜32章の章名は変わっていない。この辞典は全32章なので、第33章の候補（下の表の ch33）を入れるなら、章を1つ増やす設計変更が必要になる（判断の理由は `DECISIONS.md` に書く）。入れない場合は、ch33 の行は ch25（グラフィカルモデル）に置くか見送る。

## 参考にしたまとめ

| URL | 説明 |
|---|---|
| https://www.toukei-kentei.jp/exam/grade1semi/ | 統計検定公式サイトの準1級ページ。新旧2つの出題範囲表 PDF へのリンクがある |
| https://www.toukei-kentei.jp/hubfs/files/grade_range/grade1semi_hani_190628_2.pdf | 公式の出題範囲表（2018-06-28 版、2026-12-31 まで有効）。用語選定で最優先にした |
| https://www.toukei-kentei.jp/hubfs/files/20.grade/grade_range/grade1semi_hani_20270101.pdf | 公式の出題範囲表（2027-01-01 から有効）。統計的因果推論が加わり、用語の例が増えた |
| https://www.maruzenjunkudo.co.jp/products/9784780614756 | ワークブック増補版の書誌情報と目次。第33章「統計的因果推論」が追加されたことの確認に使った |
| https://zenn.dev/ryoooou/articles/718fd8051b40a6 | ワークブックの補足資料を章ごとに共同編集する記事。各章の重要語が並んでいる（順序統計量、状態空間モデル、研究デザインなど） |
| https://www.agaroot.jp/datascience/column/grade1semi/ | 準1級の解説。重回帰・時系列・主成分分析・ベイズを頻出としている |
| https://qiita.com/pascalize/items/f20714eef80bed50e8b9 | 合格体験記。ワークブックで後回しにした例題（23-3 など）と、覚えた手法が書かれている |
| https://zenn.dev/h4tnuy/articles/4eb39f9bcc42ef | 受験記。分野ごとの優先度と、L1・L2 正則化などの重要語が挙がっている |
| https://zenn.dev/hanata/articles/bb27da3a296ac3 | 受験記。ベイズ（共役事前分布）とサンプルサイズの計算を重視している |
| https://zenn.dev/gogohme/articles/statistics_grade1 | 1級の受験記。準1級から続く重要語として順序統計量、経験分布、バイアス・バリアンス分解を挙げている |
| https://zenn.dev/youken/articles/97c97a9f6bbeee | 準1級の前提知識（母集団と標本、標準誤差、不偏分散、推定量の性質）のまとめ |
| http://www.kotora.jp/c/138843-2/ | ワークブック中心の学習ガイド。分野ごとの重要事項が挙がっている |
| https://skillupai.com/blog/certification/about-statistics | 統計検定の概要。準1級の出題分野の一覧が載っている |

略号：旧表＝2018年版の出題範囲表、新表＝2027年版の出題範囲表、ry＝zenn ryoooou、gg＝zenn gogohme、yk＝zenn youken、ag＝agaroot。

## 追加候補

aliases と本文を grep で確認し、別名や近い用語として既にあるものは外した。重要なものから並べた。

| # | 用語名 | 提案する id | 章 | 重要度 | 根拠 |
|---|---|---|---|---|---|
| 1 | 確率関数と確率密度関数 | probability-mass-and-density-function | 2 | A | 旧表・新表の「確率分布と母関数」の最初の語。同時・周辺・条件付きの各関数はあるが、土台になる単独の項目がない |
| 2 | 順序統計量 | order-statistics | 8 | A | 旧表・新表（統計量）、ry（第8章）、gg。最大値・最小値の分布は計算問題でよく出る |
| 3 | 確率変数の和・線形結合の分布（再生性） | sum-of-random-variables | 4 | A | 旧表・新表（変数変換）。本文で「再生性」に触れている用語は多いが、単独の項目がない |
| 4 | モーメント（原点モーメント・中心モーメント） | moments | 3 | A | 旧表・新表（分布の特性値の最初の語）、ry（第8章）。母関数や歪度・尖度の前提になる |
| 5 | 二項分布・ポアソン分布の正規近似 | normal-approximation-binomial-poisson | 7 | A | 旧表・新表（極限定理）が明記している。今は continuity-correction と central-limit-theorem の本文で扱うだけ |
| 6 | 状態空間モデル | state-space-model | 27 | B | 旧表・新表（時系列解析）、ry（第27章） |
| 7 | 極値分布 | extreme-value-distribution | 7 | B | 旧表・新表（極限定理）。順序統計量（#2）と組で扱える |
| 8 | ファイ係数と連関係数（クラメールの V） | phi-coefficient-and-association | 28 | B | 旧表・新表（分割表の解析） |
| 9 | 一部実施要因計画 | fractional-factorial-design | 20 | B | 旧表・新表（実験計画法）。今は factorial-design と orthogonal-array の本文で触れるだけ |
| 10 | 片側信頼限界 | one-sided-confidence-limit | 9 | B | 旧表・新表（区間推定） |
| 11 | 包除原理 | inclusion-exclusion-principle | 1 | B | 旧表・新表（事象と確率） |
| 12 | 研究デザイン（コホート研究・ケースコントロール研究） | cohort-and-case-control-study | 28 | B | ry（第28章）。オッズ比と相対リスクの使い分けの前提になる。sampling-schemes では標本抽出の観点しか扱っていない |
| 13 | 標準誤差 | standard-error | 8 | B | yk。区間推定・検定・回帰のどの章でも使う語なのに、単独の項目がない（robust-standard-error しかない） |
| 14 | ブロック計画（釣合い型不完備ブロック計画） | incomplete-block-design | 20 | C | 旧表・新表（ブロック計画）。乱塊法（完備型）とは別の語 |
| 15 | ニューラルネットワークモデル | neural-network-model | 19 | C | 旧表・新表（回帰分析その他）。今の辞典には該当する語がない |
| 16 | 因果効果と潜在結果（反実仮想） | causal-effect | 33 | A | 新表（統計的因果推論）、ワークブック増補版の第33章、ag |
| 17 | バックドア基準 | backdoor-criterion | 33 | A | 新表（統計的因果推論） |
| 18 | 有向分離（d分離） | d-separation | 33 | A | 新表（統計的因果推論）。directed-acyclic-graph の続きにあたる |
| 19 | フロントドア基準 | frontdoor-criterion | 33 | B | 新表（統計的因果推論） |

補足：
- 新表の「直接効果・間接効果」は path-analysis の aliases にあり、「因果ダイアグラム・非巡回的有向グラフ」は directed-acyclic-graph が扱っているので、追加候補にしていない。第33章を作る場合は、directed-acyclic-graph を ch33 に移すか、ch33 から related で参照する。

### 新しい項目を立てずに aliases を足せば足りるもの

| 公式表の語 | 足す先の id |
|---|---|
| 分散分析表 | one-way-anova |
| ブロック化 | fisher-three-principles（局所管理）、randomized-block-design |
| 距離行列 | distance-measures |
| 潜在変数・潜在因子 | common-factor-unique-factor、structural-equation-modeling |
| 因果図・因果ダイアグラム | path-analysis、directed-acyclic-graph |
| 検定統計量 | null-and-alternative-hypothesis または rejection-region |
| 線形模型 | multiple-regression-model |
| 削除法・補完法 | complete-case-analysis、single-imputation（今の aliases は「リストワイズ削除」「単一補完」なので、表の言い方そのものを足す） |
| 一部実施要因計画（#9 を見送る場合） | factorial-design |

## 重要度の変更候補

| id | 今 → 提案 | 理由 |
|---|---|---|
| box-cox-transformation | B → A | 新表に明記された |
| almost-sure-convergence | B → A | 新表に明記された（概収束） |
| continuous-mapping-theorem | B → A | 新表に明記された |
| ci-variance-ratio | B → A | 新表に明記された（母分散の比） |
| yates-correction | B → A | 新表に明記された（イエーツの補正） |
| quadratic-discriminant-analysis | B → A | 旧表・新表とも「2次判別」を明記 |
| support-vector-machine | B → A | 旧表・新表とも「SVM」を明記 |
| path-analysis | B → A | 旧表・新表とも「パス解析／パス図」を明記。新表では因果推論の土台にもなる |
| directed-acyclic-graph | B → A | 新表の統計的因果推論に「DAG」が明記された |
| hierarchical-loglinear-model | B → A | 旧表・新表とも「階層モデル」を明記 |
| residual-analysis-contingency | C → B | 旧表・新表とも分割表の「残差分析」を明記 |
| complete-case-analysis | B → A | 新表に明記された（削除法） |
| single-imputation | B → A | 新表に明記された（補完法・代入法） |
| martingale | B → C | 出題範囲表になく、まとめ記事にも出てこない |
| independent-component-analysis | B → C | 出題範囲表になく、まとめ記事にも出てこない |
| hotelling-t2 | B → C | 出題範囲表になく、準1級では主題にならない |
| kolmogorov-smirnov-test | B → C | 出題範囲表になく、まとめ記事にも出てこない |

## 統合・格下げ候補

第1〜6章の「利用者が指定した用語」は対象外にした。

| id | 提案 | 理由 |
|---|---|---|
| lindeberg-levy-clt | central-limit-theorem に統合 | 準1級で扱う中心極限定理はリンデバーグ・レヴィ型そのものなので、内容が重なる |
| mann-whitney-u-test | wilcoxon-rank-sum-test に統合（U は aliases にする） | 2つの検定は同じもので、統計量の書き方が違うだけ。aliases も「ウィルコクソン・マン・ホイットニー検定」で重なっている |
| kolmogorov-smirnov-two-sample | kolmogorov-smirnov-test に統合 | 1標本と2標本で項目を分けるほどの範囲ではない |
| permutation-simulation | permutation-test に統合 | モンテカルロ検定は並べ替え検定を近似するもので、内容が重なる |
| test-ci-relationship-mean-difference | confidence-interval-test-duality に統合 | 第9章の「信頼区間と検定の対応」を母平均の差に当てはめた例にすぎない |
| sample-size-determination-test | required-sample-size に統合（信頼区間の幅による決め方と、検出力による決め方を1項目にする） | 公式表では「サンプルサイズの決定」の1語だけ。power-of-t-test と内容が重なる |
| information-criteria-comparison | model-selection に統合 | AIC と BIC の使い分けは model-selection・aic・bic と重なる |
| training-test-split | cross-validation に統合 | C の基礎語で、交差検証と汎化誤差の項目で説明できる |
| principal-component | principal-component-analysis に統合 | 主成分分析と主成分の定義で、内容がほぼ同じ |
| markov-property | markov-chain に統合 | マルコフ連鎖の定義そのものなので、項目を分けると重なる |
| queueing-theory | mm1-queue に統合 | 準1級で扱う待ち行列は M/M/1 が中心で、概論だけの項目になってしまう |
| counting-process | poisson-process に統合 | ポアソン過程の定義の一部として説明できる |
| pure-birth-process | birth-death-process に統合 | 出生死滅過程で死亡率を0にした特別な場合 |
| positive-recurrence | recurrence-and-transience に統合 | 再帰の細分類で、範囲表の「再帰性」に含まれる |
| studentized-range-distribution | tukey-multiple-comparison に統合 | テューキー法の中でしか使わない分布 |
| bias-in-penalized-estimation | ridge-regression に統合 | リッジ推定量のバイアスの話で、ridge と regularization に重なる |
| observed-fisher-information | fisher-information に統合 | フィッシャー情報量の推定の話として1段落で書ける |
| dfbetas | dffits に統合（影響の指標として1項目にする） | 出題範囲表は Cook の距離とてこ比までで、細かすぎる |
| multiple-correspondence-analysis | quantification-type-3 に統合 | 数量化 III 類と多重対応分析はほぼ同じ手法 |
| classical-mds | multidimensional-scaling に統合 | 準1級で扱う MDS は古典的 MDS が中心で、重なる |
| non-hierarchical-clustering | k-means-method に統合 | 準1級で扱う非階層的手法は k-means だけ |
| standardization-before-clustering | distance-measures に統合 | 距離を計算する前処理として書ける |
| goodness-of-fit-indices | structural-equation-modeling に統合 | 出題範囲表になく、SEM の項目の中で触れれば足りる |
| sampling-weight | horvitz-thompson-estimator に統合 | 包含確率の逆数としてそこで説明できる |
| bartlett-test, one-way-test-variance-equality-levene | 1項目「等分散性の検定（バートレット・ルビーン）」に統合 | どちらも出題範囲表になく C。2項目に分ける必要がない |
| tic-ebic | 重要度Cのまま、aic の「発展」に統合してもよい | 出題範囲表になく、範囲を超えている |

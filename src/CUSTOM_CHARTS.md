# カスタム図の一覧（charts の `type: custom` で使う）

`id` と `params` の名前はこの表のとおりに書く。`sliders` の `param` には params の名前（数値のもの）を指定できる。
乱数を使う図は「もう一度」ボタンはないが、シードは固定で毎回同じ図になる。表にない図が必要なら、報告に「id・params・何を描くか」を書く。

| id | 用途 | params（既定値） |
|---|---|---|
| bayes-area | ベイズの面積図 | prior(0.01), sensitivity=P(B\|A)(0.99), specificity=P(Bᶜ\|Aᶜ)(0.95), labelA, labelNotA, labelB（文字） |
| joint-marginal | 同時確率の表と周辺・条件付き分布 | table: [[...]], rowLabels, colLabels, rowName, colName |
| mean-inequality | AM ≥ GM ≥ HM を数直線に | a(2), b(8) |
| mgf-curve | M(t) と t=0 での接線 | family: poisson\|binom\|norm\|expon, lambda, n, p, mu, sigma, rate |
| bar-data | データを直接渡す棒グラフ（スクリープロット等） | labels: [...], values: [...], xlabel, ylabel, cumulative: true で累積割合の線, ymin・ymax（負の値があるとき ymin） |
| line-data | データを直接渡す折れ線 | x: [...], series: [{label, y: [...]（null の点は描かない）, dash: 1 で点線}], xlabel, ylabel, ymin |
| scatter-data | データを直接渡す散布図（相関係数を表示） | series: [{label, x: [...], y: [...]}]（1 系列なら x, y を直接）, xlabel, ylabel, fit: 1 回帰直線, fitEach: 1 系列ごとの回帰直線, means: 1 平均の点線, hline: 0 横線, noR: 1 相関係数を出さない, xmin, xmax, ymin, ymax |
| logistic-curve | ロジスティック曲線 | beta0(0), beta1(1), xmin(-8), xmax(8), probit: 1 でプロビットを重ねる |
| box-cox | Box-Cox 変換の曲線 | lambda(0.5), xmax(4) |
| clt-sim | 中心極限定理のシミュレーション | base: expon\|unif\|bernoulli\|poisson\|chi2, n(5), reps(3000), p(0.2), lambda(1) |
| lln-path | 大数の法則（標本平均の推移） | p(0.5), n(1000), paths(5) |
| ci-coverage | 信頼区間を何本も描いて被覆率を見る | mu(0), sigma(1), n(10), level(0.95), count(100), known: 1 で z区間／0 で t区間 |
| power-curve | z検定の α・β と検出力関数 | mu0(0), mu1(0.5), sigma(1), n(20), alpha(0.05), alt: greater\|less\|two-sided（less は左片側） |
| markov-chain | 状態遷移図と分布の推移・定常分布 | P: [[...]], labels: [...], start(0), steps(20) |
| poisson-process | 到着時刻と計数過程 | lambda(1), T(20) |
| random-walk | ランダムウォーク・破産問題 | p(0.5), steps(200), paths(5), start(0), lower, upper（吸収壁、省略可） |
| regression-residual | 回帰直線・残差・てこ比・クックの距離 | n(20), beta0(1), beta1(0.5), sigma(1), outlier（右端の点のyのずれ, 0） |
| qq-plot | 正規Q-Qプロット | dist: norm\|t\|expon\|unif, n(50), df(3) |
| ridge-lasso-path | リッジ・Lasso の係数の推移 | method: ridge\|lasso |
| bias-variance | 多項式の次数と訓練誤差・テスト誤差 | degree(3), noise(0.3), n(20) |
| pca-axes | 散布図と主成分の軸 | rho(0.8), sx(1), sy(1), n(200) |
| dendrogram | 階層的クラスタリング | n(10), method: single\|complete\|average\|ward |
| kmeans-steps | k-means法の反復 | k(3), n(90), step(0) |
| roc-curve | 2群の分布とROC曲線・AUC | mu0(0), mu1(1.5), sigma(1), threshold(0.75) |
| acf-plot | 時系列・標本自己相関（コレログラム） | model: ar1\|ar2\|ma1\|white, phi(0.7), phi2(0), theta(0.6), n(200), lags(20), pacf: 1 で偏自己相関も |
| ar-sim | AR(1)/AR(2) の系列 | phi(0.7), phi2(0), n(200) |
| mosaic | 分割表のモザイク図・オッズ比・カイ二乗 | table: [[...]], rowLabels, colLabels |
| km-curve | カプラン・マイヤー曲線（2群ならログランク） | n(30), rate(0.1), censor(0.04), groups: 1\|2, hr(2) |
| prior-posterior | 事前・尤度・事後と信用区間 | model: beta-binomial {a,b,n,x} \| gamma-poisson {shape,rate,n,sum} \| normal-normal {mu0,tau0,sigma,n,xbar} |
| mcmc-trace | メトロポリス法のトレースとヒストグラム | target: norm\|normmix, step(1), iters(3000), burnin(300), init(-6) |
| inverse-transform | 逆関数法 | dist（連続分布名）, n(1000), その分布のパラメータ（例 rate: 1） |
| rejection-sampling | 棄却法（ベータ分布） | a(2), b(5), n(600) |
| bootstrap-hist | ブートストラップ分布 | n(20), B(1000), stat: mean\|median |
| obs-exp-bar | 観測度数と期待度数の棒、カイ二乗適合度 | labels, observed: [...], expected: [...], ddof（推定した母数の数, 0） |
| ks-ecdf | 経験分布関数と理論分布、KS統計量 | data: [...], dist（連続分布名）, その分布のパラメータ |
| residual-pattern | 残差 対 予測値のプロット | pattern: none\|funnel\|curve\|autocorr, strength(1), n(80) |
| dw-sim | AR(1)誤差の回帰残差とダービン・ワトソン比 | phi(0.5), n(30) |
| studentized-range | スチューデント化された範囲の分布と上側α点 | k(3), df(9), alpha(0.05) |
| iv-scatter | 内生性のある回帰で最小二乗法と操作変数法の直線を比較 | rho(0.6), strength(1), n(200), beta(1) |
| points-2d | ラベル付きの点（負荷量プロット・得点の散布図） | points: [{label, x, y}], arrows: 1 で原点から矢印, circle: 1 で単位円, xlabel, ylabel |
| graph-diagram | 無向グラフ・有向グラフ（グラフィカルモデル、パス図） | nodes: [...], edges: [[i, j], ...]（0始まり）, directed: 1 で矢印, weights: [...]（辺のラベル）, layout: circle\|line |
| discriminant-boundary | 2つの正規分布のベイズ判別境界（線形／二次） | delta(2), rho(0.5), ratio(1：共分散の倍率、1以外で二次), prior(0.5：群2の事前確率), n(60) |
| fisher-projection | フィッシャーの判別：射影の向きと射影値の分布 | angle(0, 度), rho(0.6), delta(2), n(60) |
| mahalanobis-contour | マハラノビス距離の楕円とユークリッド距離の円 | rho(0.7), sx(1), sy(1) |
| knn-boundary | k近傍法の判別領域 | k(5), n(60) |
| svm-margin | マージン最大の分離直線とサポートベクトル | gap(1：負で重なる), n(30) |
| minkowski-ball | ミンコフスキー距離の単位円 | p(2) |
| graph-network | graph-diagram と同じ（edgeLabels で辺のラベル、layout: line\|circle\|triangle） | nodes, edges, edgeLabels, directed, layout |
| rotation-angle | 因子負荷量を固定して座標軸を回す（バリマックス基準を表示） | loadings: [[a,b],...], labels, angle(0, 度) |

補足：points-2d は points の各要素に group を付けるか、ラベルの先頭を「[行]」「[列]」のように書くと、種類ごとに色が変わる。
| missing-scatter | 欠測メカニズム（MCAR/MAR/MNAR）と観測値だけの平均の偏り | mechanism: mcar\|mar\|mnar, rho(0.6), n(200) |
| credible-hpd | 等裾の信用区間と HPD 区間の比較 | dist: beta\|gamma, a, b（beta）または shape, rate（gamma）, level(0.95) |
| box-muller | 一様乱数と、ボックス・ミュラー変換後の正規乱数の散布図 | n(1000) |
| gibbs-path | 2変量正規分布でのギブスサンプリングの軌跡 | rho(0.8), iters(30), init(2) |

（2級で追加）

| id | 用途 | params（既定値） |
|---|---|---|
| box-plot | 箱ひげ図（四分位数は中央値で分けた下半分・上半分の中央値。ひげは 1.5×IQR の内側の最も外の値まで、外は外れ値の点） | groups: [{label, data: [...]}], xlabel, mean: 1 で平均を × で表示 |
| lorenz-curve | ローレンツ曲線とジニ係数 | values: [...]（1人あたりの量）, weights: [...]（人数、省略時は1） |
| moving-average | 元の系列と中心化移動平均（k が偶数なら 2×k の中心化） | y: [...], x: [...], k(3), xlabel, ylabel |
| stratified-scatter | 層別の散布図（層の中と全体で相関が違う例） | gap(3：層の平均のずれ), within(-0.5：層の中の傾き), n(40：各層の点の数), labels: [層の名前2つ], xlabel, ylabel |

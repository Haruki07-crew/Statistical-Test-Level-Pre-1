# カスタム図の一覧（charts の `type: custom` で使う）

`id` と `params` の名前はこの表のとおりに書く。`sliders` の `param` には params の名前（数値のもの）を指定できる。
乱数を使う図は「もう一度」ボタンはないが、シードは固定で毎回同じ図になる。表にない図が必要なら、報告に「id・params・何を描くか」を書く。

| id | 用途 | params（既定値） |
|---|---|---|
| bayes-area | ベイズの面積図 | prior(0.01), sensitivity=P(B\|A)(0.99), specificity=P(Bᶜ\|Aᶜ)(0.95), labelA, labelNotA, labelB（文字） |
| joint-marginal | 同時確率の表と周辺・条件付き分布 | table: [[...]], rowLabels, colLabels, rowName, colName |
| mean-inequality | AM ≥ GM ≥ HM を数直線に | a(2), b(8) |
| mgf-curve | M(t) と t=0 での接線 | family: poisson\|binom\|norm\|expon, lambda, n, p, mu, sigma, rate |
| bar-data | データを直接渡す棒グラフ（スクリープロット等） | labels: [...], values: [...], xlabel, ylabel, cumulative: true で累積割合の線 |
| line-data | データを直接渡す折れ線 | x: [...], series: [{label, y: [...]}], xlabel, ylabel |
| logistic-curve | ロジスティック曲線 | beta0(0), beta1(1), xmin(-8), xmax(8), probit: 1 でプロビットを重ねる |
| box-cox | Box-Cox 変換の曲線 | lambda(0.5), xmax(4) |
| clt-sim | 中心極限定理のシミュレーション | base: expon\|unif\|bernoulli\|poisson\|chi2, n(5), reps(3000), p(0.2), lambda(1) |
| lln-path | 大数の法則（標本平均の推移） | p(0.5), n(1000), paths(5) |
| ci-coverage | 信頼区間を何本も描いて被覆率を見る | mu(0), sigma(1), n(10), level(0.95), count(100), known: 1 で z区間／0 で t区間 |
| power-curve | z検定の α・β と検出力関数 | mu0(0), mu1(0.5), sigma(1), n(20), alpha(0.05), alt: greater\|two-sided |
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

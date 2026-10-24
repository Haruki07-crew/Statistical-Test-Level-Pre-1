"""2級の学習ロードマップ（content/g2/roadmap.yaml）を作る。

段階・優先度・前提・目安時間・理由は、公式の出題範囲表（spec/research/g2/format.md）と、
2級の問題で毎回のように問われる分野（記述統計・確率分布・推定と検定・回帰の出力）をもとに決めた。
準1級の make_roadmap.py と同じ形式で、scripts/check_roadmap.py（JD_LEVEL=g2）で検査する。
使い方: .venv/bin/python scripts/make_roadmap_g2.py
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
STAGES = [
    {"no": 1, "title": "まず固める基礎", "desc": "データのまとめ方、確率と主要な分布、標本平均の分布、区間推定と検定の基本手順。2級の得点の土台になる。"},
    {"no": 2, "title": "得点源", "desc": "グラフと格差の指標、時系列の指数、標本調査と実験、2標本の検定とカイ二乗検定、回帰と分散分析の出力。毎回のように出る。"},
    {"no": 3, "title": "差がつく応用", "desc": "擬似相関と偏相関、物価指数、いろいろな分布の性質、多重共線性やダミー変数など、押さえると周りと差がつく項目。"},
    {"no": 4, "title": "余裕があれば", "desc": "出題がまれな発展事項と、準1級で詳しく扱う項目。直前に眺める程度でよい。"},
]
# (id, 段階, 優先度, 前提, 分, 理由)
T = [
    # ---- 段階1
    ("data-types", 1, 97, [], 20, "変数の種類で使えるグラフと要約が決まる。すべての出発点"),
    ("frequency-distribution-histogram", 1, 98, ["data-types"], 30, "度数分布とヒストグラムの読み取りは毎回出る"),
    ("cumulative-frequency", 1, 93, ["frequency-distribution-histogram"], 25, "割合の読み取りと中央値の見当付けに使う"),
    ("mean-median-mode", 1, 98, ["frequency-distribution-histogram"], 30, "代表値の計算と使い分けは記述統計の中心"),
    ("sample-variance-sd", 1, 98, ["mean-median-mode"], 35, "分散と標準偏差は推定・検定まで使い続ける"),
    ("quartiles-iqr", 1, 95, ["mean-median-mode"], 30, "四分位数は箱ひげ図の問題の前提"),
    ("box-plot", 1, 97, ["quartiles-iqr"], 30, "ヒストグラムとの対応を問う問題が定番"),
    ("standard-score", 1, 92, ["sample-variance-sd"], 25, "標準化は正規分布の確率計算につながる"),
    ("scatter-plot", 1, 95, ["data-types"], 20, "相関と回帰の前に必ず形を確かめる"),
    ("covariance-and-correlation", 1, 98, ["scatter-plot", "sample-variance-sd"], 35, "相関係数の計算と解釈は毎回出る"),
    ("cross-tabulation", 1, 94, ["data-types"], 25, "割合の比べ方と独立性の検定の前提"),
    ("simple-regression", 1, 97, ["covariance-and-correlation"], 35, "回帰直線の傾きと予測値の計算は頻出"),
    ("least-squares-method", 1, 90, ["simple-regression"], 25, "回帰の係数の決め方。重回帰の出力の理解に必要"),
    ("coefficient-of-determination", 1, 94, ["least-squares-method"], 25, "当てはまりの指標。出力の読み取りでも使う"),
    ("population-and-sample", 1, 96, [], 25, "推測統計の言葉の土台。母数と統計量の区別"),
    ("simple-random-sampling", 1, 91, ["population-and-sample"], 20, "無作為抽出は推測の前提。抽出法の基本"),
    ("observational-vs-experimental", 1, 93, ["population-and-sample"], 25, "因果を主張できるかの判断がよく問われる"),
    ("events-and-probability", 1, 96, [], 25, "加法定理と余事象は確率計算の出発点"),
    ("conditional-probability", 1, 96, ["events-and-probability"], 30, "条件付き確率は独立とベイズの前提"),
    ("independence-of-events", 1, 92, ["conditional-probability"], 20, "独立と排反の区別が頻出"),
    ("bayes-theorem", 1, 93, ["conditional-probability"], 30, "検査の陽性的中率などの定番問題"),
    ("probability-mass-and-density-function", 1, 92, ["events-and-probability"], 25, "確率変数と分布を扱う計算の前提"),
    ("expectation-variance-properties", 1, 97, ["probability-mass-and-density-function"], 35, "期待値と分散の公式は全分野で使う"),
    ("bernoulli-distribution", 1, 88, ["expectation-variance-properties"], 15, "二項分布と比率の推測の土台"),
    ("binomial-distribution", 1, 97, ["bernoulli-distribution"], 30, "二項確率と期待値・分散は頻出"),
    ("normal-distribution", 1, 99, ["expectation-variance-properties"], 35, "推定と検定の前提となる最重要の分布"),
    ("standard-normal-distribution", 1, 98, ["normal-distribution", "standard-score"], 30, "分布表で確率を求める計算は毎回出る"),
    ("sampling-distribution-of-mean", 1, 98, ["normal-distribution", "population-and-sample"], 30, "標本平均の分散 σ²/n はすべての推測の基礎"),
    ("central-limit-theorem", 1, 95, ["sampling-distribution-of-mean"], 25, "正規近似を使える理由。比率の推測につながる"),
    ("statistical-tables", 1, 97, ["standard-normal-distribution"], 25, "分布表の上側確率点を正しく選ぶ力が問われる"),
    ("t-distribution", 1, 96, ["statistical-tables", "sampling-distribution-of-mean"], 25, "母分散が未知のときの区間と検定に使う"),
    ("chi-squared-distribution", 1, 91, ["statistical-tables"], 25, "母分散の推測とカイ二乗検定に使う"),
    ("point-estimation", 1, 90, ["sampling-distribution-of-mean"], 20, "推定量と推定値の区別。推定の言葉の土台"),
    ("standard-error", 1, 95, ["sampling-distribution-of-mean"], 25, "区間推定と検定の分母になる量"),
    ("confidence-interval", 1, 97, ["standard-error"], 30, "信頼区間の意味の正しい解釈がよく問われる"),
    ("ci-mean-known-variance", 1, 94, ["confidence-interval"], 25, "区間推定の基本形。z の点を使う"),
    ("ci-mean-unknown-variance", 1, 97, ["ci-mean-known-variance", "t-distribution"], 30, "t の点を使う区間は最頻出の計算"),
    ("ci-proportion", 1, 96, ["confidence-interval", "binomial-distribution"], 25, "世論調査型の比率の区間は頻出"),
    ("null-and-alternative-hypothesis", 1, 96, ["sampling-distribution-of-mean"], 20, "仮説の立て方は検定の出発点"),
    ("significance-level", 1, 93, ["null-and-alternative-hypothesis"], 15, "棄却の基準。第1種の過誤の確率"),
    ("p-value", 1, 97, ["significance-level"], 25, "p値の意味と判断は毎回出る"),
    ("type-i-and-ii-errors", 1, 94, ["significance-level"], 25, "2種類の誤りと検出力の関係が問われる"),
    ("one-sided-two-sided-test", 1, 92, ["null-and-alternative-hypothesis"], 20, "片側と両側で比べる点が変わる"),
    ("z-test-known-variance", 1, 91, ["one-sided-two-sided-test", "standard-normal-distribution"], 20, "検定統計量の作り方の基本形"),
    ("one-sample-t-test", 1, 96, ["z-test-known-variance", "t-distribution"], 30, "1標本t検定は最頻出の検定"),
    ("proportion-test", 1, 94, ["z-test-known-variance", "ci-proportion"], 25, "比率の検定は正規近似で頻出"),
    # ---- 段階2
    ("distribution-shape", 2, 88, ["frequency-distribution-histogram", "mean-median-mode"], 20, "形と代表値の大小関係の問題が定番"),
    ("stem-and-leaf-plot", 2, 78, ["frequency-distribution-histogram"], 15, "幹葉図から中央値を読む問題が出る"),
    ("bar-pie-band-chart", 2, 80, ["data-types"], 20, "目的に合うグラフの選択と割合の読み取り"),
    ("lorenz-curve-gini", 2, 86, ["cumulative-frequency"], 30, "2級に特有の項目。台形の面積で計算する"),
    ("coefficient-of-variation", 2, 82, ["sample-variance-sd"], 15, "単位の違う散らばりの比較で出る"),
    ("residual", 2, 82, ["simple-regression"], 15, "残差の性質は回帰の基本"),
    ("growth-rate", 2, 87, ["mean-median-mode"], 25, "平均成長率と寄与度の計算が頻出"),
    ("index-number", 2, 85, ["growth-rate"], 20, "指数の基準とポイントの扱いが問われる"),
    ("moving-average", 2, 84, ["mean-median-mode"], 20, "移動平均の計算と項数の選び方"),
    ("time-series-components", 2, 83, ["moving-average"], 20, "時系列のグラフの読み取りの土台"),
    ("fisher-three-principles", 2, 86, ["observational-vs-experimental"], 20, "反復・無作為化・局所管理は定番の知識問題"),
    ("treatment-and-control", 2, 85, ["fisher-three-principles"], 20, "実験の設計の問題で必ず使う"),
    ("sampling-and-nonsampling-error", 2, 84, ["simple-random-sampling"], 20, "標本誤差と偏りの区別が問われる"),
    ("stratified-sampling", 2, 84, ["simple-random-sampling"], 20, "抽出法の識別問題で頻出"),
    ("cluster-sampling", 2, 80, ["simple-random-sampling"], 15, "層化抽出との区別が問われる"),
    ("systematic-sampling", 2, 76, ["simple-random-sampling"], 15, "抽出法の識別で選択肢に出る"),
    ("counting", 2, 80, ["events-and-probability"], 20, "組合せは二項分布と超幾何分布の前提"),
    ("law-of-total-probability", 2, 82, ["conditional-probability"], 20, "ベイズの定理の分母の計算に使う"),
    ("poisson-distribution", 2, 88, ["binomial-distribution"], 25, "まれな事象の分布として頻出"),
    ("exponential-distribution", 2, 80, ["poisson-distribution"], 20, "待ち時間の分布と期待値の計算"),
    ("uniform-distribution", 2, 76, ["probability-mass-and-density-function"], 15, "連続分布の基本形。期待値と分散"),
    ("normal-approximation-binomial-poisson", 2, 89, ["central-limit-theorem", "binomial-distribution"], 25, "二項分布の正規近似は頻出の計算"),
    ("continuity-correction", 2, 82, ["normal-approximation-binomial-poisson"], 15, "正規近似の精度を上げる修正"),
    ("weak-law-of-large-numbers", 2, 80, ["sampling-distribution-of-mean"], 15, "標本平均が母平均に近づく意味の理解"),
    ("f-distribution", 2, 85, ["chi-squared-distribution"], 25, "分散の比と分散分析で使う"),
    ("unbiased-estimator", 2, 87, ["point-estimation"], 20, "不偏分散が n−1 で割る理由"),
    ("confidence-coefficient", 2, 80, ["confidence-interval"], 15, "信頼係数と区間の幅の関係"),
    ("required-sample-size", 2, 84, ["ci-proportion"], 20, "必要な標本の大きさの計算が出る"),
    ("ci-two-sample-mean-difference", 2, 82, ["ci-mean-unknown-variance"], 25, "2群の平均の差の区間"),
    ("ci-two-sample-proportion", 2, 78, ["ci-proportion"], 20, "2群の比率の差の区間"),
    ("ci-variance", 2, 82, ["chi-squared-distribution", "confidence-interval"], 25, "カイ二乗分布の2つの点を使う区間"),
    ("rejection-region", 2, 82, ["significance-level"], 15, "棄却域による判断の形"),
    ("power-function", 2, 80, ["type-i-and-ii-errors"], 20, "検出力と標本の大きさの関係"),
    ("chi-square-test-variance", 2, 80, ["ci-variance"], 20, "母分散の検定"),
    ("two-sample-t-test", 2, 90, ["one-sample-t-test"], 30, "2群の平均の比較は頻出"),
    ("pooled-variance", 2, 82, ["two-sample-t-test"], 15, "等分散を仮定した2標本t検定の計算"),
    ("welch-test", 2, 84, ["two-sample-t-test"], 20, "分散が等しいとは限らないときの方法"),
    ("paired-t-test", 2, 89, ["one-sample-t-test"], 25, "対応のあるデータの検定。2標本との区別"),
    ("f-test-equal-variances", 2, 80, ["f-distribution"], 20, "分散の比の検定。自由度の順に注意"),
    ("two-sample-proportion-test", 2, 82, ["proportion-test"], 20, "2群の比率の差の検定"),
    ("goodness-of-fit-test", 2, 87, ["chi-squared-distribution"], 25, "適合度検定は頻出のカイ二乗検定"),
    ("pearson-chi-square-statistic", 2, 80, ["goodness-of-fit-test"], 15, "カイ二乗統計量の計算の形"),
    ("expected-frequency", 2, 82, ["cross-tabulation"], 15, "独立性の検定の期待度数の計算"),
    ("independence-test", 2, 90, ["expected-frequency", "pearson-chi-square-statistic"], 25, "クロス表の独立性の検定は頻出"),
    ("regression-output", 2, 90, ["coefficient-of-determination"], 30, "ソフトウェアの出力の読み取りは毎回出る"),
    ("multiple-regression-model", 2, 86, ["regression-output"], 25, "偏回帰係数の意味の理解"),
    ("regression-coefficient-t-test", 2, 87, ["regression-output", "one-sample-t-test"], 25, "係数のt値とp値の判断"),
    ("adjusted-r-squared", 2, 84, ["coefficient-of-determination"], 20, "自由度調整済み決定係数の計算と意味"),
    ("one-way-anova", 2, 87, ["f-distribution"], 30, "3群以上の平均の比較の基本"),
    ("sum-of-squares-decomposition", 2, 84, ["one-way-anova"], 20, "全体の変動を分ける考え方"),
    ("anova-table", 2, 88, ["sum-of-squares-decomposition"], 25, "空欄のある分散分析表を埋める問題が定番"),
    # ---- 段階3
    ("scales-of-measurement", 3, 66, ["data-types"], 15, "尺度に応じた要約の選び方を整理する"),
    ("price-index", 3, 72, ["index-number"], 25, "ラスパイレス指数の計算がときどき出る"),
    ("weighted-geometric-harmonic-mean", 3, 68, ["growth-rate"], 20, "幾何平均と調和平均の使いどころ"),
    ("skewness-kurtosis", 3, 64, ["distribution-shape"], 15, "歪みの向きと歪度の符号"),
    ("stratified-scatter", 3, 70, ["covariance-and-correlation"], 15, "全体と層の中で相関が違う例"),
    ("spurious-correlation", 3, 74, ["stratified-scatter"], 20, "相関と因果の区別がよく問われる"),
    ("partial-correlation-coefficient", 3, 66, ["spurious-correlation"], 20, "第3の変数の影響を除いた相関"),
    ("correlation-matrix", 3, 62, ["covariance-and-correlation"], 15, "多変数の相関をまとめて読む"),
    ("simpsons-paradox", 3, 68, ["cross-tabulation", "stratified-scatter"], 20, "層別で結論が逆転する例"),
    ("residual-plot", 3, 66, ["residual"], 15, "回帰の当てはまりの点検"),
    ("log-transformation", 3, 62, ["simple-regression"], 15, "右に裾の長いデータの扱い"),
    ("autocovariance-autocorrelation", 3, 60, ["time-series-components"], 20, "系列相関の考え方"),
    ("correlogram", 3, 62, ["autocovariance-autocorrelation"], 15, "コレログラムの読み取り"),
    ("two-stage-sampling", 3, 64, ["cluster-sampling"], 15, "多段抽出の仕組み"),
    ("cumulative-distribution-function", 3, 66, ["probability-mass-and-density-function"], 15, "分布関数と確率の関係"),
    ("joint-probability-function", 3, 64, ["probability-mass-and-density-function"], 20, "2変数の確率分布の扱い"),
    ("sum-of-random-variables", 3, 70, ["expectation-variance-properties"], 20, "和と差の期待値と分散"),
    ("geometric-distribution", 3, 66, ["bernoulli-distribution"], 15, "初めて成功するまでの回数の分布"),
    ("hypergeometric-distribution", 3, 66, ["counting", "binomial-distribution"], 20, "戻さずに取り出す抽出の分布"),
    ("chebyshev-inequality", 3, 64, ["sample-variance-sd", "weak-law-of-large-numbers"], 15, "分布の形によらない確率の評価"),
    ("consistent-estimator", 3, 60, ["unbiased-estimator"], 15, "推定量のよさの基準の1つ"),
    ("confidence-interval-test-duality", 3, 68, ["confidence-interval", "p-value"], 15, "区間と検定の結論の対応"),
    ("overall-f-test-regression", 3, 70, ["regression-coefficient-t-test", "f-distribution"], 15, "回帰全体の有意性の判断"),
    ("multicollinearity", 3, 72, ["multiple-regression-model"], 20, "説明変数どうしの相関の影響"),
    ("dummy-variable", 3, 70, ["multiple-regression-model"], 20, "質的な説明変数の扱い"),
    ("randomized-block-design", 3, 68, ["fisher-three-principles", "one-way-anova"], 20, "ブロック化による精度の向上"),
    ("main-effect-and-interaction", 3, 62, ["one-way-anova"], 20, "2つの要因の効果の分け方"),
    # ---- 段階4
    ("official-statistics", 4, 50, ["population-and-sample"], 15, "データの出所を確かめる姿勢を確認する"),
    ("seasonal-adjustment", 4, 46, ["time-series-components"], 15, "季節調整値の意味を知っておく"),
    ("nonresponse-bias", 4, 44, ["sampling-and-nonsampling-error"], 10, "無回答による偏りの例"),
    ("marginal-distribution", 4, 42, ["joint-probability-function"], 10, "同時分布から周辺分布を求める"),
    ("negative-binomial-distribution", 4, 38, ["geometric-distribution"], 10, "出題はまれ。幾何分布の一般化"),
    ("bivariate-normal-distribution", 4, 36, ["normal-distribution", "covariance-and-correlation"], 10, "出題はまれ。準1級で詳しく扱う"),
    ("ci-variance-ratio", 4, 44, ["f-test-equal-variances"], 10, "分散の比の区間。検定と同じ点を使う"),
    ("fisher-z-transformation", 4, 40, ["covariance-and-correlation", "ci-mean-known-variance"], 10, "相関係数の区間推定に使う"),
    ("finite-population-correction", 4, 42, ["sampling-distribution-of-mean"], 10, "戻さない抽出での分散の修正"),
    ("test-correlation-coefficient", 4, 40, ["covariance-and-correlation", "t-distribution"], 10, "母相関が0かどうかの検定"),
    ("confidence-and-prediction-interval-regression", 4, 40, ["regression-output"], 10, "予測の幅の考え方。出題はまれ"),
    ("two-way-anova-with-replication", 4, 38, ["main-effect-and-interaction"], 10, "出題はまれ。準1級で詳しく扱う"),
]


def main():
    out = ["# 2級の学習ロードマップ（scripts/make_roadmap_g2.py が生成。手で編集しない）",
           "# 根拠は spec/research/g2/format.md（公式の出題範囲表）", "stages:"]
    for s in STAGES:
        out.append(f"  - {{'no': {s['no']}, title: {s['title']}, desc: {s['desc']}}}")
    out.append("terms:")
    for tid, st, sc, pre, mn, why in T:
        out += [f"  - id: {tid}", f"    stage: {st}", f"    score: {sc}", f"    prereq: [{', '.join(pre)}]",
                f"    minutes: {mn}", f"    reason: {why}"]
    text = "\n".join(out) + "\n"
    yaml.safe_load(text)
    (ROOT / "content" / "g2" / "roadmap.yaml").write_text(text, encoding="utf-8")
    print(f"content/g2/roadmap.yaml を書きました（{len(T)} 語）")


if __name__ == "__main__":
    main()

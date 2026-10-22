"""一度だけ使う：第1〜6章の指定用語と全32章の章名で plan.yaml を作る。"""
import yaml
TITLES = {1:"事象と確率",2:"確率分布と母関数",3:"分布の特性値",4:"変数変換",5:"離散型分布",6:"連続型分布と標本分布",
7:"極限定理、漸近理論",8:"統計的推定の基礎",9:"区間推定",10:"検定の基礎と検定法の導出",11:"正規分布に関する検定",
12:"一般の分布に関する検定法",13:"ノンパラメトリック法",14:"マルコフ連鎖",15:"確率過程の基礎",16:"重回帰分析",
17:"回帰診断法",18:"質的回帰",19:"回帰分析その他",20:"分散分析と実験計画法",21:"標本調査法",22:"主成分分析",
23:"判別分析",24:"クラスター分析",25:"因子分析・グラフィカルモデル",26:"その他の多変量解析手法",27:"時系列解析",
28:"分割表",29:"不完全データの統計処理",30:"モデル選択",31:"ベイズ法",32:"シミュレーション"}
T = {
1: [("conditional-probability","条件付き確率と乗法定理","A"),("law-of-total-probability","全確率の公式","B"),
    ("independence-of-events","事象の独立","A"),("bayes-theorem","ベイズの定理","A"),("conditional-independence","条件付き独立","A")],
2: [("cumulative-distribution-function","累積分布関数","A"),("survival-function","生存関数","B"),("hazard-function","ハザード関数","A"),
    ("joint-probability-function","同時確率関数","A"),("marginal-distribution","周辺分布","A"),("marginal-probability-function","周辺確率関数","B"),
    ("conditional-probability-function","条件付き確率関数","A"),("joint-probability-density-function","同時確率密度関数","A"),
    ("marginal-probability-density-function","周辺確率密度関数","B"),("conditional-probability-density-function","条件付き確率密度関数","A"),
    ("moment-generating-function","モーメント母関数","A"),("probability-generating-function","確率母関数","A"),("characteristic-function","特性関数","B")],
3: [("expectation-variance-properties","期待値と分散の性質","A"),("quantile-function","分位点関数","B"),("coefficient-of-variation","変動係数","B"),
    ("covariance-and-correlation","共分散と相関係数","A"),("spurious-correlation","擬似相関","B"),("partial-correlation-coefficient","偏相関係数","A"),
    ("conditional-expectation","条件付き期待値","A"),("conditional-variance","条件付き分散","A"),("skewness-kurtosis","歪度と尖度","B"),
    ("weighted-geometric-harmonic-mean","加重平均・幾何平均・調和平均","B"),("mean-vector","期待値ベクトル","B"),
    ("variance-covariance-matrix","分散共分散行列","A"),("correlation-matrix","相関係数行列","B")],
4: [("change-of-variables","変数変換とヤコビアン","A"),("log-transformation","対数変換","B"),("power-transformation","べき乗変換","C"),
    ("box-cox-transformation","Box-Cox変換","B"),("logit-logistic-transformation","ロジット変換・ロジスティック変換","A"),("probit-transformation","プロビット変換","B")],
5: [("bernoulli-distribution","ベルヌーイ試行・ベルヌーイ分布","A"),("binomial-distribution","二項分布","A"),("hypergeometric-distribution","超幾何分布","A"),
    ("finite-population-correction","有限母集団修正","B"),("fisher-exact-test","フィッシャーの正確検定","B"),("poisson-distribution","ポアソン分布","A"),
    ("geometric-distribution","幾何分布と無記憶性","A"),("negative-binomial-distribution","負の二項分布","A"),("multinomial-distribution","多項分布","A")],
6: [("normal-distribution","正規分布","A"),("standard-normal-distribution","標準正規分布","A"),("uniform-distribution","一様分布","B"),
    ("exponential-distribution","指数分布","A"),("gamma-distribution","ガンマ分布","A"),("beta-distribution","ベータ分布","A"),
    ("cauchy-distribution","コーシー分布","B"),("lognormal-distribution","対数正規分布","B"),("bivariate-normal-distribution","2変量正規分布","A"),
    ("multivariate-normal-distribution","多変量正規分布","A"),("normal-mixture-distribution","混合正規分布","B"),("chi-squared-distribution","カイ二乗分布","A"),
    ("noncentral-chi-squared-distribution","非心カイ二乗分布","B"),("t-distribution","t分布","A"),("t-statistic","t統計量","A"),
    ("noncentral-t-distribution","非心t分布","B"),("f-distribution","F分布","A"),("noncentral-f-distribution","非心F分布","B")],
}
chs=[]
for n in range(1,33):
    chs.append({"no":n,"title":TITLES[n],"terms":[{"id":i,"title":t,"importance":imp,"status":"planned"} for i,t,imp in T.get(n,[])]})
yaml.safe_dump({"chapters":chs}, open("content/plan.yaml","w"), allow_unicode=True, sort_keys=False, width=200)

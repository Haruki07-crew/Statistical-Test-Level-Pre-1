"""scipy で dist.js テスト用の参照値を作り verify/dist_reference.json に書く。"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CASES = [
    ("bernoulli", {"p": 0.3}, stats.bernoulli(0.3), [0, 1]),
    ("binom", {"n": 20, "p": 0.35}, stats.binom(20, 0.35), [0, 3, 7, 12, 20]),
    ("hypergeom", {"N": 50, "K": 12, "n": 10}, stats.hypergeom(50, 12, 10), [0, 2, 4, 7]),
    ("poisson", {"lambda": 4.5}, stats.poisson(4.5), [0, 2, 5, 9, 15]),
    ("geom", {"p": 0.3, "start": 0}, stats.geom(0.3, loc=-1), [0, 1, 4, 10]),
    ("geom", {"p": 0.3, "start": 1}, stats.geom(0.3), [1, 2, 5, 11]),
    ("nbinom", {"r": 3, "p": 0.4}, stats.nbinom(3, 0.4), [0, 2, 5, 12]),
    ("unif", {"a": -1, "b": 3}, stats.uniform(-1, 4), [-0.5, 0.7, 2.9]),
    ("norm", {"mu": 1, "sigma": 2}, stats.norm(1, 2), [-6, -2, 0.3, 1, 4.5, 9]),
    ("lognorm", {"mu": 0.2, "sigma": 0.7}, stats.lognorm(0.7, scale=np.exp(0.2)), [0.1, 0.8, 1.5, 4]),
    ("expon", {"rate": 1.5}, stats.expon(scale=1 / 1.5), [0.01, 0.5, 2, 5]),
    ("gamma", {"shape": 2.5, "scale": 1.8}, stats.gamma(2.5, scale=1.8), [0.1, 1, 4, 10, 20]),
    ("gamma", {"shape": 0.6, "scale": 1}, stats.gamma(0.6), [0.05, 0.5, 3]),
    ("beta", {"a": 2.5, "b": 4}, stats.beta(2.5, 4), [0.05, 0.3, 0.6, 0.95]),
    ("beta", {"a": 0.5, "b": 0.5}, stats.beta(0.5, 0.5), [0.02, 0.5, 0.9]),
    ("cauchy", {"loc": 1, "scale": 2}, stats.cauchy(1, 2), [-10, 0, 1, 7]),
    ("weibull", {"shape": 1.7, "scale": 2}, stats.weibull_min(1.7, scale=2), [0.2, 1, 2.5, 5]),
    ("logistic", {"loc": 0.5, "scale": 1.3}, stats.logistic(0.5, 1.3), [-4, 0, 2, 6]),
    ("chi2", {"df": 1}, stats.chi2(1), [0.01, 0.5, 3.84, 10]),
    ("chi2", {"df": 7}, stats.chi2(7), [1, 5, 14.07, 25]),
    ("t", {"df": 3}, stats.t(3), [-4, -1, 0.5, 3.18]),
    ("t", {"df": 30}, stats.t(30), [-2.04, 0, 1.3]),
    ("f", {"df1": 4, "df2": 12}, stats.f(4, 12), [0.2, 1, 3.26, 6]),
    ("f", {"df1": 1, "df2": 5}, stats.f(1, 5), [0.1, 2, 6.61]),
    ("ncx2", {"df": 3, "nc": 4}, stats.ncx2(3, 4), [0.5, 3, 7, 15, 30]),
    ("ncx2", {"df": 1, "nc": 20}, stats.ncx2(1, 20), [5, 20, 40]),
    ("nct", {"df": 10, "nc": 1.5}, stats.nct(10, 1.5), [-1, 0.5, 1.5, 3, 6]),
    ("nct", {"df": 3, "nc": -1}, stats.nct(3, -1), [-5, -1, 0, 2]),
    ("ncf", {"df1": 3, "df2": 20, "nc": 5}, stats.ncf(3, 20, 5), [0.3, 1, 2.5, 6]),
    ("normmix", {"w": 0.3, "mu1": -1, "sigma1": 0.5, "mu2": 2, "sigma2": 1.5}, None, [-2, 0, 1, 3.5]),
]
out = []
for name, params, rv, xs in CASES:
    for x in xs:
        if rv is None:
            w, m1, s1, m2, s2 = (params[k] for k in ("w", "mu1", "sigma1", "mu2", "sigma2"))
            pdf = w * stats.norm.pdf(x, m1, s1) + (1 - w) * stats.norm.pdf(x, m2, s2)
            cdf = w * stats.norm.cdf(x, m1, s1) + (1 - w) * stats.norm.cdf(x, m2, s2)
        else:
            pdf = rv.pmf(x) if hasattr(rv.dist, "pmf") else rv.pdf(x)
            cdf = rv.cdf(x)
        out.append({"dist": name, "params": params, "x": x, "pdf": float(pdf), "cdf": float(cdf)})
for name, params, rv, p in [("norm", {"mu": 0, "sigma": 1}, stats.norm(), 0.975), ("t", {"df": 9}, stats.t(9), 0.975),
                           ("chi2", {"df": 4}, stats.chi2(4), 0.95), ("f", {"df1": 3, "df2": 10}, stats.f(3, 10), 0.95),
                           ("binom", {"n": 20, "p": 0.4}, stats.binom(20, 0.4), 0.5)]:
    out.append({"dist": name, "params": params, "p": p, "quantile": float(rv.ppf(p))})
(ROOT / "verify" / "dist_reference.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"{len(out)} 件の参照値を書きました")

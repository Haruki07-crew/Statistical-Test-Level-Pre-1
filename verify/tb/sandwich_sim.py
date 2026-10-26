"""サンドイッチ分散推定量のページの例（単回帰・不均一分散）のシミュレーション。verify/tb/ch47.py から使う。"""
import numpy as np
from scipy import stats
def run(g,n=100,R=20000,seed=47):
    rng=np.random.default_rng(seed)
    x=rng.random((R,n)); c=1.0 if g==0 else np.sqrt((np.exp(2*g)-1)/(2*g)); s=np.exp(g*x)/c
    y=1+2*x+rng.normal(size=(R,n))*s
    xc=x-x.mean(1,keepdims=True); Sxx=(xc**2).sum(1)
    b=(xc*(y-y.mean(1,keepdims=True))).sum(1)/Sxx; a=y.mean(1)-b*x.mean(1)
    r=y-a[:,None]-b[:,None]*x
    se_n=np.sqrt((r**2).sum(1)/(n-2)/Sxx)
    se_0=np.sqrt((xc**2*r**2).sum(1))/Sxx
    h=1/n+xc**2/Sxx[:,None]
    se_3=np.sqrt((xc**2*r**2/(1-h)**2).sum(1))/Sxx
    q=stats.t.isf(0.025,n-2)
    cov=lambda se: (np.abs(b-2)<=q*se).mean()
    return b.std(), se_n.mean(), se_0.mean(), se_3.mean(), cov(se_n), cov(se_0), cov(se_3)
if __name__=='__main__':
    for g in [0,0.5,1,1.5,2,2.5,3]:
        print(g, [round(float(v),4) for v in run(g)])

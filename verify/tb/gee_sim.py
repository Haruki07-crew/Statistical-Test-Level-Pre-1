"""GEE のページの例（施設ごとに割り付けた 2 値データ）のシミュレーション。verify/tb/ch50.py から使う。"""
import numpy as np
def logit_fit(X,y,it=25):
    b=np.zeros(X.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-X@b)); W=p*(1-p); H=(X.T*W)@X; b=b+np.linalg.solve(H,X.T@(y-p))
    p=1/(1+np.exp(-X@b)); W=p*(1-p); H=(X.T*W)@X
    return b,np.linalg.inv(H),p
def run(sd_u, K=50, m=10, R=2000, seed=50):
    rng=np.random.default_rng(seed); cov_n=cov_r=0; ratio=[]
    for r in range(R):
        trt=np.repeat((np.arange(K)%2).astype(float),m); u=np.repeat(rng.normal(0,sd_u,K),m)
        eta=-0.5+0.8*trt+u; y=(rng.random(K*m)<1/(1+np.exp(-eta))).astype(float)
        X=np.c_[np.ones(K*m),trt]; b,Hinv,p=logit_fit(X,y)
        sc=(X*(y-p)[:,None]).reshape(K,m,2).sum(1); B=sc.T@sc; Vr=Hinv@B@Hinv
        # marginal true trt effect on logit scale is attenuated; use marginal truth computed separately
        ratio.append(np.sqrt(Vr[1,1]/Hinv[1,1]))
        yield b[1], np.sqrt(Hinv[1,1]), np.sqrt(Vr[1,1])
def marginal_truth(sd_u, n=2_000_000, seed=1):
    rng=np.random.default_rng(seed); u=rng.normal(0,sd_u,n)
    p0=(1/(1+np.exp(-(-0.5+u)))).mean(); p1=(1/(1+np.exp(-(0.3+u)))).mean()
    return np.log(p1/(1-p1))-np.log(p0/(1-p0))
if __name__=='__main__':
    for sd in [0,0.25,0.5,0.75,1.0,1.5]:
        th=marginal_truth(sd); res=np.array(list(run(sd,R=1000)))
        cn=(np.abs(res[:,0]-th)<=1.96*res[:,1]).mean(); cr=(np.abs(res[:,0]-th)<=1.96*res[:,2]).mean()
        print(sd, round(th,4), round(res[:,0].std(),4), round(res[:,1].mean(),4), round(res[:,2].mean(),4), cn, cr)

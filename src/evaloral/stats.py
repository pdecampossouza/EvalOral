from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon


def bootstrap_stat_ci(values, stat=np.median, n_boot=5000, seed=42, q=(0.025,0.975)):
    values=np.asarray(values,float)
    rng=np.random.default_rng(seed)
    boots=np.empty(n_boot,float)
    for i in range(n_boot):
        boots[i]=stat(rng.choice(values,size=len(values),replace=True))
    return float(np.quantile(boots,q[0])), float(np.quantile(boots,q[1]))


def bootstrap_paired_mean_diff(a,b,n_boot=5000,seed=42):
    a=np.asarray(a,float);b=np.asarray(b,float);d=a-b
    rng=np.random.default_rng(seed);boots=np.empty(n_boot,float)
    for i in range(n_boot): boots[i]=rng.choice(d,size=len(d),replace=True).mean()
    return float(d.mean()),float(np.quantile(boots,.025)),float(np.quantile(boots,.975))


def holm_adjust(pvalues):
    p=np.asarray(pvalues,float);m=len(p);order=np.argsort(p);adj=np.empty(m,float);running=0.0
    for rank,idx in enumerate(order):
        v=(m-rank)*p[idx];running=max(running,v);adj[idx]=min(running,1.0)
    return adj


def paired_wilcoxon(a,b):
    try: return float(wilcoxon(a,b,zero_method='wilcox',alternative='two-sided').pvalue)
    except ValueError: return 1.0

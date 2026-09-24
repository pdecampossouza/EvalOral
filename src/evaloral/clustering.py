from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


def prepare_unique_space(X,n_components=4,random_state=42):
    sc=StandardScaler().fit(X);Z0=sc.transform(X)
    pca=PCA(n_components=n_components,whiten=False,svd_solver='randomized',random_state=random_state).fit(Z0)
    return sc,pca,pca.transform(Z0)


def cluster_grid(Z,k_values=range(2,13),seeds=range(10),n_init=20):
    rows=[];parts={}
    for k in k_values:
        labels=[]
        for seed in seeds:
            km=KMeans(n_clusters=k,n_init=n_init,random_state=seed).fit(Z);labels.append(km.labels_)
            rows.append({'k':k,'seed':seed,'silhouette':silhouette_score(Z,km.labels_)})
        aris=[adjusted_rand_score(a,b) for a,b in itertools.combinations(labels,2)]
        for r in rows:
            if r['k']==k:r['mean_pairwise_ari']=float(np.mean(aris)) if aris else 1.0
        parts[k]=labels
    return pd.DataFrame(rows),parts


def representative_kmeans(Z,k,seed=42,n_init=20):
    return KMeans(n_clusters=k,n_init=n_init,random_state=seed).fit(Z)


def prototypes(Z,labels,centers,n_per=5):
    out={}
    for k in range(len(centers)):
        idx=np.where(labels==k)[0]
        d=np.linalg.norm(Z[idx]-centers[k],axis=1)
        out[k]=idx[np.argsort(d)[:n_per]].tolist()
    return out


def cluster_rule_concordance(cluster_labels,rule_ids,view_labels=None):
    c=np.asarray(cluster_labels);r=np.asarray(rule_ids)
    out={'ari_cluster_rule':adjusted_rand_score(c,r),'nmi_cluster_rule':normalized_mutual_info_score(c,r)}
    if view_labels is not None:
        v=np.asarray(view_labels)
        def purity(a,b):
            n=0
            for x in np.unique(a):
                bb=b[a==x];n+=pd.Series(bb).value_counts().max()
            return n/len(a)
        out['cluster_to_view_purity']=purity(c,v);out['rule_to_view_purity']=purity(r,v)
    return out

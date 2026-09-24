from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
from .evolving_nf import EvolvingNeuroFuzzyEvalOral

@dataclass
class StreamConfig:
    n_components:int=4
    whiten:bool=True
    sigma_init:float=1.0
    alpha_add:float=0.40
    activation_mode:str='coverage_gated_unim'
    label_conflict_policy:str='new_rule'
    label_conflict_min_confidence:float=0.65
    feature_weight_floor:float=0.05
    tau_merge:float=1.10
    enable_merging:bool=False


def shuffled_volunteers(groups,seed):
    v=np.array(sorted(pd.unique(groups)))
    np.random.default_rng(seed).shuffle(v)
    return v


def prepare_stream_space(X,groups,seed,n_components=4,whiten=True):
    """Fit preprocessing once for a volunteer-order stream.

    Useful for parameter sweeps in which only fuzzy-model parameters change.
    """
    vols=shuffled_volunteers(groups,seed)
    init=vols[:3];stream=vols[3:]
    tr=np.isin(groups,init)
    sc=StandardScaler().fit(X[tr]);Xz=sc.transform(X)
    pca=PCA(n_components=n_components,whiten=whiten,svd_solver='randomized',random_state=seed).fit(Xz[tr])
    Z=pca.transform(Xz)
    return {'seed':seed,'init':init,'stream':stream,'train_mask':tr,'scaler':sc,'pca':pca,'Z':Z}


def run_nf_prepared(Z,y,groups,init,stream,seed,config:StreamConfig,return_model=False):
    nf=EvolvingNeuroFuzzyEvalOral(
        n_features=config.n_components,n_classes=2,alpha_add=config.alpha_add,
        tau_merge=config.tau_merge if config.enable_merging else 1.10,
        buffer_size_similarity=200,buffer_size_separability=200,max_rules=None,
        merge_require_same_class=True,merge_max_js=0.15,
        feature_weight_floor=config.feature_weight_floor,sigma_init=config.sigma_init,
        activation_mode=config.activation_mode,label_conflict_policy=config.label_conflict_policy,
        label_conflict_min_confidence=config.label_conflict_min_confidence,random_state=seed,
        feature_names=[f'PC{i+1}' for i in range(config.n_components)],
        class_names=['frontal','occlusal'])
    for v in init:
        idx=np.where(groups==v)[0];nf.partial_fit(Z[idx],y[idx])
    pred=[];true=[];per_block=[]
    for v in stream:
        idx=np.where(groups==v)[0];p=nf.predict(Z[idx]);
        pred.extend(p.tolist());true.extend(y[idx].tolist())
        per_block.append({'volunteer':str(v),'n':len(idx),'macro_f1':f1_score(y[idx],p,average='macro',zero_division=0),'accuracy':accuracy_score(y[idx],p)})
        nf.partial_fit(Z[idx],y[idx])
    row={'seed':seed,'macro_f1':f1_score(true,pred,average='macro',zero_division=0),'accuracy':accuracy_score(true,pred),'final_rules':len(nf.rules),'init_volunteers':'|'.join(map(str,init)),'stream_order':'|'.join(map(str,stream))}
    if return_model:return row,nf,pd.DataFrame(per_block)
    return row


def run_one_stream(X,y,groups,seed,config:StreamConfig,return_model=False):
    prep=prepare_stream_space(X,groups,seed,config.n_components,config.whiten)
    result=run_nf_prepared(prep['Z'],y,groups,prep['init'],prep['stream'],seed,config,return_model=return_model)
    if return_model:
        row,nf,per_block=result
        return row,nf,prep['scaler'],prep['pca'],prep['Z'],per_block
    return result


def run_static_anchor(X,y,groups,seed,n_components=4,whiten=True):
    vols=shuffled_volunteers(groups,seed);init=vols[:3];stream=vols[3:];tr=np.isin(groups,init)
    sc=StandardScaler().fit(X[tr]);Xz=sc.transform(X);pca=PCA(n_components=n_components,whiten=whiten,svd_solver='randomized',random_state=seed).fit(Xz[tr]);Z=pca.transform(Xz)
    clf=LinearSVC(C=1.0,penalty='l2',loss='squared_hinge',dual='auto',tol=1e-4,class_weight='balanced',fit_intercept=True,max_iter=1000,random_state=seed).fit(Z[tr],y[tr])
    pred=[];true=[]
    for v in stream:
        idx=np.where(groups==v)[0];pred.extend(clf.predict(Z[idx]).tolist());true.extend(y[idx].tolist())
    return {'seed':seed,'macro_f1':f1_score(true,pred,average='macro',zero_division=0),'accuracy':accuracy_score(true,pred),'final_rules':np.nan}

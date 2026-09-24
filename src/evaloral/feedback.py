from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from .evolving_nf import EvolvingNeuroFuzzyEvalOral

CLASSES=['frontal','lateral','occlusal']; LMAP={c:i for i,c in enumerate(CLASSES)}


def majority_label(labels,fallback):
    vc=pd.Series(list(labels)).value_counts()
    if len(vc)==0:return fallback
    top=vc[vc==vc.max()].index.tolist();return fallback if fallback in top else sorted(top)[0]


def semantic_gallery_replay(feedback):
    rows=[]
    for _,r in feedback.iterrows():
        g=feedback[(feedback.rule_id==r.rule_id)&(feedback.image_id!=r.image_id)]
        pred=majority_label(g.corrected_view,r.original_rule_label)
        rows.append({'image_id':r.image_id,'rule_id':r.rule_id,'expert_label':r.corrected_view,'original_prediction':r.original_rule_label,'feedback_prediction_loo':pred})
    df=pd.DataFrame(rows)
    y=df.expert_label.map(LMAP);a=df.original_prediction.map(LMAP);b=df.feedback_prediction_loo.map(LMAP)
    return df,{'original_accuracy':accuracy_score(y,a),'feedback_loo_accuracy':accuracy_score(y,b),'original_macro_f1':f1_score(y,a,average='macro',zero_division=0),'feedback_loo_macro_f1':f1_score(y,b,average='macro',zero_division=0)}


def build_binary_nf(seed):
    return EvolvingNeuroFuzzyEvalOral(4,2,alpha_add=.4,tau_merge=1.10,feature_weight_floor=.05,sigma_init=1.0,activation_mode='coverage_gated_unim',label_conflict_policy='new_rule',label_conflict_min_confidence=.65,random_state=seed,class_names=['frontal','occlusal'],feature_names=['PC1','PC2','PC3','PC4'])


def rule_semantic_logo(X,meta,feedback,n_orders=30):
    yorig=meta.released_view.map({'frontal':0,'occlusal':1}).to_numpy();groups=meta.volunteer_id.astype(str).to_numpy();id2idx={i:j for j,i in enumerate(meta.image_id)};holds=sorted(feedback.volunteer_id.unique())
    fold={}
    for h in holds:
        tr=groups!=h;sc=StandardScaler().fit(X[tr]);Xz=sc.transform(X);pca=PCA(n_components=4,whiten=True,svd_solver='randomized',random_state=42).fit(Xz[tr]);fold[h]=(tr,sc,pca,pca.transform(Xz),{g:i for i,g in enumerate(np.where(tr)[0])})
    rows=[]
    for seed in range(n_orders):
        rng=np.random.default_rng(seed)
        for h in holds:
            tr,sc,pca,Ztr,g2l=fold[h];nf=build_binary_nf(seed);tg=list(pd.unique(groups[tr]));rng.shuffle(tg)
            for gid in tg:
                gi=np.where((groups==gid)&tr)[0];li=np.array([g2l[i] for i in gi]);nf.partial_fit(Ztr[li],yorig[gi])
            counts={r.rule_id:np.zeros(3) for r in nf.rules};ftr=feedback[feedback.volunteer_id!=h]
            ii=np.array([id2idx[i] for i in ftr.image_id]);Zfb=pca.transform(sc.transform(X[ii]));C=nf.rule_coverages(Zfb)
            for j,(_,fr) in enumerate(ftr.iterrows()):
                ri=int(np.argmax(C[j]));counts[nf.rules[ri].rule_id][LMAP[fr.corrected_view]]+=1
            fte=feedback[feedback.volunteer_id==h];ii=np.array([id2idx[i] for i in fte.image_id]);Zt=pca.transform(sc.transform(X[ii]));Ct=nf.rule_coverages(Zt)
            for j,(_,rr) in enumerate(fte.iterrows()):
                ri=int(np.argmax(Ct[j]));rule=nf.rules[ri];orig='frontal' if rule.dominant_class==0 else 'occlusal';cnt=counts[rule.rule_id]
                if cnt.sum()>0:
                    mx=cnt.max();cand=np.where(cnt==mx)[0].tolist();fb=LMAP[orig];si=fb if fb in cand else cand[0];upd=CLASSES[si]
                else:upd=orig
                rows.append({'seed':seed,'holdout_volunteer':h,'image_id':rr.image_id,'expert_label':rr.corrected_view,'original_rule_prediction':orig,'feedback_updated_rule_prediction':upd,'winning_rule_id':rule.rule_id,'coverage':float(Ct[j,ri]),'feedback_support_for_rule':int(cnt.sum()),'n_rules':len(nf.rules)})
    df=pd.DataFrame(rows)
    metrics=[]
    for seed,g in df.groupby('seed'):
        yt=g.expert_label.map(LMAP);yo=g.original_rule_prediction.map(LMAP);yu=g.feedback_updated_rule_prediction.map(LMAP)
        metrics.append({'seed':seed,'original_accuracy':accuracy_score(yt,yo),'feedback_updated_accuracy':accuracy_score(yt,yu),'original_macro_f1':f1_score(yt,yo,average='macro',zero_division=0),'feedback_updated_macro_f1':f1_score(yt,yu,average='macro',zero_division=0),'mean_rules':g.n_rules.mean()})
    return df,pd.DataFrame(metrics)


def cluster_logo(X,meta,feedback,n_orders=30):
    y=feedback.corrected_view.map(LMAP).to_numpy();id2idx={i:j for j,i in enumerate(meta.image_id)};idx=np.array([id2idx[i] for i in feedback.image_id]);Xf=X[idx];groups=feedback.volunteer_id.astype(str).to_numpy();rows=[]
    for seed in range(n_orders):
      for h in sorted(pd.unique(groups)):
        tr=groups!=h;te=groups==h;sc=StandardScaler().fit(Xf[tr]);A=sc.transform(Xf[tr]);B=sc.transform(Xf[te]);pca=PCA(n_components=4,whiten=True,svd_solver='randomized',random_state=42).fit(A);Ztr,Zte=pca.transform(A),pca.transform(B);ytr,yte=y[tr],y[te]
        def cmap(labels):
            mp={};glob=int(pd.Series(ytr).value_counts().idxmax())
            for k in range(3):
                z=ytr[labels==k];mp[k]=glob if len(z)==0 else int(pd.Series(z).value_counts().idxmax())
            return mp
        un=KMeans(3,n_init=20,random_state=seed).fit(Ztr);mp=cmap(un.labels_);pu=np.array([mp[k] for k in un.predict(Zte)])
        init=np.stack([Ztr[ytr==c].mean(0) for c in range(3)]);fb=KMeans(3,init=init,n_init=1,random_state=seed).fit(Ztr);mp2=cmap(fb.labels_);pf=np.array([mp2[k] for k in fb.predict(Zte)])
        for j,(_,rr) in enumerate(feedback.loc[te].reset_index(drop=True).iterrows()):rows.append({'seed':seed,'holdout_volunteer':h,'image_id':rr.image_id,'expert_label':rr.corrected_view,'unseeded_prediction':CLASSES[pu[j]],'feedback_seeded_prediction':CLASSES[pf[j]]})
    return pd.DataFrame(rows)

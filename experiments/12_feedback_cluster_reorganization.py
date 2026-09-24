#!/usr/bin/env python3
"""Descriptive before/after K-means membership reorganization on the 40 feedback images.

This is intentionally in-sample/descriptive. It is NOT the leave-one-volunteer-out
generalization test; see experiment 11 for that analysis.
"""
import json
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,f1_score,normalized_mutual_info_score,adjusted_rand_score
from evaloral.data import repo_root,load_view121
from evaloral.feedback import CLASSES,LMAP


def cmap(labels,y):
    mp={};glob=int(pd.Series(y).value_counts().idxmax())
    for k in range(3):
        z=y[labels==k];mp[k]=glob if len(z)==0 else int(pd.Series(z).value_counts().idxmax())
    return mp


def main():
    root=repo_root();X,_,m=load_view121(root);fb=pd.read_csv(root/'data/reference/author_feedback40.csv');out=root/'results/generated/12_feedback_cluster_reorganization';out.mkdir(parents=True,exist_ok=True)
    id2={i:j for j,i in enumerate(m.image_id)};idx=np.array([id2[i] for i in fb.image_id]);A=X[idx];y=fb.corrected_view.map(LMAP).to_numpy()
    sc=StandardScaler().fit(A);Az=sc.transform(A);pca=PCA(n_components=4,whiten=True,random_state=42).fit(Az);Z=pca.transform(Az);Z2=PCA(n_components=2,random_state=42).fit_transform(Az)
    un=KMeans(n_clusters=3,n_init=50,random_state=0).fit(Z);mp=cmap(un.labels_,y);pu=np.array([mp[k] for k in un.labels_])
    init=np.stack([Z[y==c].mean(0) for c in range(3)]);seeded=KMeans(n_clusters=3,init=init,n_init=1,random_state=0).fit(Z);mp2=cmap(seeded.labels_,y);ps=np.array([mp2[k] for k in seeded.labels_])
    changed=pu!=ps
    tab=fb[['image_id','volunteer_id','corrected_view','filepath']].copy();tab['PCA1_display']=Z2[:,0];tab['PCA2_display']=Z2[:,1];tab['unseeded_cluster_id']=un.labels_;tab['unseeded_semantic']=[CLASSES[i] for i in pu];tab['feedback_seeded_cluster_id']=seeded.labels_;tab['feedback_seeded_semantic']=[CLASSES[i] for i in ps];tab['semantic_assignment_changed']=changed.astype(int);tab.to_csv(out/'cluster_membership_before_after.csv',index=False)
    mat=np.zeros((3,3),int)
    for a,b in zip(pu,ps):mat[a,b]+=1
    pd.DataFrame(mat,index=CLASSES,columns=CLASSES).to_csv(out/'semantic_migration_matrix.csv')
    summary={'n_images':len(y),'semantic_assignments_changed':int(changed.sum()),'changed_fraction':float(changed.mean()),'unseeded_accuracy':accuracy_score(y,pu),'feedback_seeded_accuracy':accuracy_score(y,ps),'unseeded_macro_f1':f1_score(y,pu,average='macro',zero_division=0),'feedback_seeded_macro_f1':f1_score(y,ps,average='macro',zero_division=0),'unseeded_nmi':normalized_mutual_info_score(y,pu),'feedback_seeded_nmi':normalized_mutual_info_score(y,ps),'unseeded_ari':adjusted_rand_score(y,pu),'feedback_seeded_ari':adjusted_rand_score(y,ps)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(13,4));
    for ax,pred,title in [(axs[0],pu,'Before feedback: unseeded K-means'),(axs[1],ps,'After feedback: seeded K-means')]:
        for c,name in enumerate(CLASSES):
            z=pred==c;ax.scatter(Z2[z,0],Z2[z,1],label=f'{name} (n={z.sum()})',s=38)
        ax.scatter(Z2[changed,0],Z2[changed,1],facecolors='none',edgecolors='black',s=90,linewidths=1.1);ax.set_xlabel('PCA 1');ax.set_ylabel('PCA 2');ax.set_title(title);ax.legend(frameon=False,fontsize=7)
    im=axs[2].imshow(mat)
    for i in range(3):
        for j in range(3):axs[2].text(j,i,str(mat[i,j]),ha='center',va='center')
    axs[2].set_xticks(range(3),CLASSES,rotation=25,ha='right');axs[2].set_yticks(range(3),CLASSES);axs[2].set_xlabel('After');axs[2].set_ylabel('Before');axs[2].set_title(f'Migration ({changed.sum()}/{len(y)} changed)');fig.colorbar(im,ax=axs[2],fraction=.046,pad=.04)
    fig.suptitle('Code-derived cluster organization before and after human feedback');fig.tight_layout();fig.savefig(out/'Figure_cluster_before_after_feedback.png',dpi=300,bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from PIL import Image,ImageOps
from evaloral.data import repo_root,load_unique
from evaloral.clustering import prepare_unique_space,cluster_grid,representative_kmeans,prototypes

def main():
 root=repo_root();X,m=load_unique(root);out=root/'results/generated/02_unsupervised_clustering';out.mkdir(parents=True,exist_ok=True)
 sc,pca,Z=prepare_unique_space(X);np.save(out/'pca4_nonwhitened.npy',Z)
 grid,_=cluster_grid(Z);grid.to_csv(out/'k2_to_k12_seed_stability.csv',index=False)
 agg=grid.groupby('k').agg(silhouette_mean=('silhouette','mean'),silhouette_sd=('silhouette','std'),mean_pairwise_ari=('mean_pairwise_ari','first')).reset_index();agg.to_csv(out/'k_summary.csv',index=False)
 for k in [2,7]:
  km=representative_kmeans(Z,k);a=m.copy();a['cluster']=km.labels_;a.to_csv(out/f'k{k}_assignments.csv',index=False);pr=prototypes(Z,km.labels_,km.cluster_centers_,5);rows=[]
  for c,idxs in pr.items():
   for rank,i in enumerate(idxs,1):rows.append({'cluster':c,'rank':rank,'image_id':m.iloc[i].image_id,'filepath':m.iloc[i].filepath})
  pd.DataFrame(rows).to_csv(out/f'k{k}_prototypes.csv',index=False)
 # figure metrics
 fig,ax=plt.subplots(figsize=(6.8,4));ax.plot(agg.k,agg.silhouette_mean,marker='o',label='Silhouette');ax.plot(agg.k,agg.mean_pairwise_ari,marker='s',label='Mean pairwise ARI');ax.set_xlabel('k');ax.set_ylabel('Score');ax.set_title('Label-free clustering: structure and seed stability');ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/'Figure_cluster_k_sweep.png',dpi=250);plt.close(fig)
 # k7 prototype gallery
 pr=pd.read_csv(out/'k7_prototypes.csv');fig,axes=plt.subplots(7,5,figsize=(10,12));
 for c in range(7):
  g=pr[pr.cluster==c].sort_values('rank')
  for j,(_,r) in enumerate(g.iterrows()):
   ax=axes[c,j];im=Image.open(root/r.filepath).convert('RGB');ax.imshow(im);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'C{c} P{j+1}',fontsize=8)
 fig.suptitle('Representative oral photographs from label-free k=7 clusters');fig.tight_layout();fig.savefig(out/'Figure_cluster_prototypes_k7.png',dpi=250,bbox_inches='tight');plt.close(fig)
 print(agg.to_string(index=False))
if __name__=='__main__':main()

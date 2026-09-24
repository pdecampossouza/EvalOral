#!/usr/bin/env python3
"""Compare label-free fine clusters with final evolving fuzzy-rule regions."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaloral.data import repo_root,load_unique,load_view121
from evaloral.clustering import prepare_unique_space,representative_kmeans,cluster_rule_concordance


def main():
    root=repo_root(); out=root/'results/generated/08_cluster_rule_concordance'; out.mkdir(parents=True,exist_ok=True)
    Xu,um=load_unique(root); _,_,vm=load_view121(root)
    _,_,Z=prepare_unique_space(Xu); km=representative_kmeans(Z,7,seed=42,n_init=20)
    u=um[['sha256','image_id']].copy(); u['fine_cluster']=km.labels_
    ref=pd.read_csv(root/'data/reference/final_rule_assignments_view121.csv')
    # Match the DAI anchor to the content-unique partition by SHA-256, because
    # duplicated source photos can be represented by a different module ID in
    # the content-unique table.
    join=vm[['image_id','sha256','released_view']].merge(u[['sha256','fine_cluster']],on='sha256',how='left')
    join=join.merge(ref[['image_id','final_rule_id','final_rule_coverage']],on='image_id',how='left')
    join.to_csv(out/'view121_cluster_rule_assignments.csv',index=False)
    valid=join.dropna(subset=['fine_cluster','final_rule_id'])
    met=cluster_rule_concordance(valid.fine_cluster.astype(int),valid.final_rule_id.astype(int),valid.released_view)
    pd.DataFrame([met]).to_csv(out/'cluster_rule_concordance_metrics.csv',index=False)
    tab=pd.crosstab(valid.final_rule_id,valid.fine_cluster); tab.to_csv(out/'rule_by_fine_cluster_crosstab.csv')
    fig,ax=plt.subplots(figsize=(8,5)); im=ax.imshow(tab.values,aspect='auto');
    ax.set_xticks(range(len(tab.columns)),[f'C{x}' for x in tab.columns]); ax.set_yticks(range(len(tab.index)),[f'R{x}' for x in tab.index]);
    ax.set_xlabel('Label-free k=7 cluster'); ax.set_ylabel('Final fuzzy rule'); ax.set_title('Cluster–rule concordance on the view-labelled anchor subset')
    fig.colorbar(im,ax=ax,label='Images'); fig.tight_layout(); fig.savefig(out/'Figure_cluster_rule_concordance.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    print(pd.DataFrame([met]).to_string(index=False))

if __name__=='__main__': main()

#!/usr/bin/env python3
import argparse,pandas as pd,numpy as np,matplotlib.pyplot as plt
from evaloral.data import repo_root,load_view121
from evaloral.protocols import StreamConfig,run_one_stream
from evaloral.stats import bootstrap_paired_mean_diff,paired_wilcoxon,holm_adjust

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n-orders',type=int,default=30);a=ap.parse_args()
 root=repo_root();X,y,m=load_view121(root);g=m.volunteer.astype(int).to_numpy();out=root/'results/generated/04_ablation';out.mkdir(parents=True,exist_ok=True)
 variants={
  'legacy_geometry':StreamConfig(whiten=False,activation_mode='unim',label_conflict_policy='adapt',feature_weight_floor=.0),
  'pca_whitening':StreamConfig(whiten=True,activation_mode='unim',label_conflict_policy='adapt',feature_weight_floor=.0),
  'coverage_gate':StreamConfig(whiten=True,activation_mode='coverage_gated_unim',label_conflict_policy='adapt',feature_weight_floor=.05),
  'consequent_aware':StreamConfig(),
 }
 rows=[]
 for rep in range(a.n_orders):
  seed=20260824+rep
  for name,cfg in variants.items():rows.append({'variant':name,**run_one_stream(X,y,g,seed,cfg)})
 df=pd.DataFrame(rows);df.to_csv(out/'ablation_streams.csv',index=False)
 pairs=[('pca_whitening','legacy_geometry'),('coverage_gate','pca_whitening'),('consequent_aware','coverage_gate')];stats=[];rawp=[]
 for a,b in pairs:
  aa=df[df.variant==a].sort_values('seed').macro_f1.to_numpy();bb=df[df.variant==b].sort_values('seed').macro_f1.to_numpy();d,lo,hi=bootstrap_paired_mean_diff(aa,bb);p=paired_wilcoxon(aa,bb);stats.append({'comparison':f'{a} vs {b}','mean_delta_f1':d,'ci_low':lo,'ci_high':hi,'raw_p':p});rawp.append(p)
 adj=holm_adjust(rawp)
 for r,p in zip(stats,adj):r['holm_p']=p
 pd.DataFrame(stats).to_csv(out/'paired_effects.csv',index=False)
 order=list(variants);vals=[df[df.variant==v].macro_f1.mean() for v in order];fig,ax=plt.subplots(figsize=(7,4));ax.bar(range(len(order)),vals);ax.set_xticks(range(len(order)),[x.replace('_','\n') for x in order]);ax.set_ylabel('Mean macro-F1');ax.set_ylim(0,1);ax.set_title('Methodological ablation');fig.tight_layout();fig.savefig(out/'Figure_ablation.png',dpi=250);plt.close(fig)
 print(pd.DataFrame(stats).to_string(index=False))
if __name__=='__main__':main()

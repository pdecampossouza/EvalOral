#!/usr/bin/env python3
import argparse,itertools,pandas as pd,numpy as np,matplotlib.pyplot as plt
from evaloral.data import repo_root,load_view121
from evaloral.protocols import StreamConfig,prepare_stream_space,run_nf_prepared

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n-orders',type=int,default=10);a=ap.parse_args()
 root=repo_root();X,y,m=load_view121(root);g=m.volunteer.astype(int).to_numpy();out=root/'results/generated/05_hyperparameter_sensitivity';out.mkdir(parents=True,exist_ok=True);rows=[]
 params=list(itertools.product([.5,1.,1.5,2.],[.2,.3,.4,.5,.6,.7]))
 # Preprocessing depends on seed but not on sigma/alpha, so cache it once per
 # volunteer order. This preserves the paper protocol while making the 240-run
 # grid much faster than refitting PCA for every operating point.
 for rep in range(a.n_orders):
  seed=20260824+rep;prep=prepare_stream_space(X,g,seed,4,True)
  for sigma,alpha in params:
   cfg=StreamConfig(sigma_init=sigma,alpha_add=alpha)
   rows.append({'sigma_init':sigma,'alpha_add':alpha,**run_nf_prepared(prep['Z'],y,g,prep['init'],prep['stream'],seed,cfg)})
 df=pd.DataFrame(rows);df.to_csv(out/'sensitivity_240_runs.csv',index=False);agg=df.groupby(['sigma_init','alpha_add']).agg(median_macro_f1=('macro_f1','median'),median_rules=('final_rules','median')).reset_index();agg.to_csv(out/'sensitivity_summary.csv',index=False)
 p=agg.pivot(index='sigma_init',columns='alpha_add',values='median_macro_f1');fig,ax=plt.subplots(figsize=(7,4));im=ax.imshow(p.values,aspect='auto');ax.set_xticks(range(len(p.columns)),p.columns);ax.set_yticks(range(len(p.index)),p.index);ax.set_xlabel('alpha');ax.set_ylabel('sigma_init');ax.set_title('Median macro-F1 sensitivity');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'Figure_sigma_alpha_f1.png',dpi=250);plt.close(fig);print(agg.sort_values('median_macro_f1',ascending=False).head(10).to_string(index=False))
if __name__=='__main__':main()

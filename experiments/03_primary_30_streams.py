#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np,pandas as pd,matplotlib.pyplot as plt,json
from evaloral.data import repo_root,load_view121
from evaloral.protocols import StreamConfig,run_one_stream,run_static_anchor
from evaloral.stats import bootstrap_stat_ci

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n-orders',type=int,default=30);a=ap.parse_args()
 root=repo_root();X,y,m=load_view121(root);groups=m.volunteer.astype(int).to_numpy();out=root/'results/generated/03_primary_30_streams';out.mkdir(parents=True,exist_ok=True)
 cfg=StreamConfig();nf=[];svc=[]
 for rep in range(a.n_orders):
  seed=20260824+rep;nf.append(run_one_stream(X,y,groups,seed,cfg));svc.append(run_static_anchor(X,y,groups,seed))
 n=pd.DataFrame(nf);s=pd.DataFrame(svc);n.to_csv(out/f'evolving_nf_{a.n_orders}_streams.csv',index=False);s.to_csv(out/f'linearsvc_paired_rerun_{a.n_orders}_streams.csv',index=False)
 lo,hi=bootstrap_stat_ci(n.macro_f1);summary={'evolving_nf_median_macro_f1':float(n.macro_f1.median()),'evolving_nf_bootstrap_ci':[lo,hi],'evolving_nf_median_accuracy':float(n.accuracy.median()),'median_final_rules':float(n.final_rules.median()),'paired_linearsvc_rerun_median_macro_f1':float(s.macro_f1.median()),'manuscript_reported_linearsvc_anchor_macro_f1':0.964}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));
 fig,ax=plt.subplots(figsize=(6,4));ax.boxplot([n.macro_f1,s.macro_f1],tick_labels=['Evolving NF','LinearSVC\npaired rerun']);ax.set_ylabel('Prequential macro-F1');ax.set_ylim(0,1.03);ax.set_title(f'{a.n_orders} randomized volunteer-order streams');fig.tight_layout();fig.savefig(out/'Figure_primary_30_streams.png',dpi=250);plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

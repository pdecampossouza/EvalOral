#!/usr/bin/env python3
import argparse,time,pandas as pd,numpy as np
from evaloral.data import repo_root,load_view121
from evaloral.protocols import StreamConfig,prepare_stream_space,run_nf_prepared

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n-orders',type=int,default=10);a=ap.parse_args()
 root=repo_root();X,y,m=load_view121(root);g=m.volunteer.astype(int).to_numpy();out=root/'results/generated/06_merge_analysis';out.mkdir(parents=True,exist_ok=True);rows=[]
 for rep in range(a.n_orders):
  seed=20260824+rep;prep=prepare_stream_space(X,g,seed,4,True)
  for tau in [.75,.85,.90,.95,1.10]:
   cfg=StreamConfig(tau_merge=tau,enable_merging=tau<=1.0);t=time.perf_counter();r=run_nf_prepared(prep['Z'],y,g,prep['init'],prep['stream'],seed,cfg);r.update({'tau_merge':tau,'elapsed_s':time.perf_counter()-t});rows.append(r)
 df=pd.DataFrame(rows);df.to_csv(out/'merge_analysis.csv',index=False);print(df.groupby('tau_merge').agg(median_f1=('macro_f1','median'),median_rules=('final_rules','median'),median_elapsed_s=('elapsed_s','median')).to_string())
if __name__=='__main__':main()

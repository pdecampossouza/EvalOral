#!/usr/bin/env python3
"""Leave-one-volunteer-out feedback generalization and seeded clustering."""
import argparse,json
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,normalized_mutual_info_score,adjusted_rand_score
from evaloral.data import repo_root,load_view121
from evaloral.feedback import rule_semantic_logo,cluster_logo,LMAP


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n-orders',type=int,default=30);a=ap.parse_args()
    root=repo_root();X,_,meta=load_view121(root);fb=pd.read_csv(root/'data/reference/author_feedback40.csv');out=root/'results/generated/11_feedback_generalization';out.mkdir(parents=True,exist_ok=True)
    pred,met=rule_semantic_logo(X,meta,fb,a.n_orders);pred.to_csv(out/'rule_semantic_logo_predictions.csv',index=False);met.to_csv(out/'rule_semantic_logo_metrics.csv',index=False)
    cp=cluster_logo(X,meta,fb,a.n_orders);cp.to_csv(out/'cluster_logo_predictions.csv',index=False)
    cm=[]
    for seed,g in cp.groupby('seed'):
        y=g.expert_label.map(LMAP);u=g.unseeded_prediction.map(LMAP);s=g.feedback_seeded_prediction.map(LMAP)
        cm.append({'seed':seed,'unseeded_accuracy':accuracy_score(y,u),'feedback_seeded_accuracy':accuracy_score(y,s),'unseeded_macro_f1':f1_score(y,u,average='macro',zero_division=0),'feedback_seeded_macro_f1':f1_score(y,s,average='macro',zero_division=0),'unseeded_nmi':normalized_mutual_info_score(y,u),'feedback_seeded_nmi':normalized_mutual_info_score(y,s),'unseeded_ari':adjusted_rand_score(y,u),'feedback_seeded_ari':adjusted_rand_score(y,s)})
    cm=pd.DataFrame(cm);cm.to_csv(out/'cluster_logo_metrics.csv',index=False)
    summary={
      'n_orders':a.n_orders,
      'rule_original_median_accuracy':float(met.original_accuracy.median()),'rule_feedback_median_accuracy':float(met.feedback_updated_accuracy.median()),
      'rule_original_median_macro_f1':float(met.original_macro_f1.median()),'rule_feedback_median_macro_f1':float(met.feedback_updated_macro_f1.median()),
      'cluster_unseeded_median_accuracy':float(cm.unseeded_accuracy.median()),'cluster_seeded_median_accuracy':float(cm.feedback_seeded_accuracy.median()),
      'cluster_unseeded_median_macro_f1':float(cm.unseeded_macro_f1.median()),'cluster_seeded_median_macro_f1':float(cm.feedback_seeded_macro_f1.median()),
      'cluster_unseeded_median_nmi':float(cm.unseeded_nmi.median()),'cluster_seeded_median_nmi':float(cm.feedback_seeded_nmi.median()),
      'cluster_unseeded_median_ari':float(cm.unseeded_ari.median()),'cluster_seeded_median_ari':float(cm.feedback_seeded_ari.median())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

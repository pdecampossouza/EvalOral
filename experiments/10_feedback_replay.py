#!/usr/bin/env python3
"""Reproduce the 40-image author-side semantic feedback replay."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from evaloral.data import repo_root
from evaloral.feedback import semantic_gallery_replay


def main():
    root=repo_root();out=root/'results/generated/10_feedback_replay';out.mkdir(parents=True,exist_ok=True)
    fb=pd.read_csv(root/'data/reference/author_feedback40.csv');pred,met=semantic_gallery_replay(fb);pred.to_csv(out/'semantic_leave_one_exemplar_out.csv',index=False)
    summary=pd.read_csv(root/'data/reference/feedback_rule_summary.csv');summary.to_csv(out/'feedback_rule_summary.csv',index=False)
    (out/'summary.json').write_text(json.dumps({k:float(v) for k,v in met.items()},indent=2))
    x=range(len(summary));fig,ax=plt.subplots(figsize=(8,4));w=.36;ax.bar([i-w/2 for i in x],summary.purity_before,w,label='Before');ax.bar([i+w/2 for i in x],summary.purity_after,w,label='After feedback');ax.set_xticks(list(x),[f'R{x}' for x in summary.rule_id]);ax.set_ylim(0,1.05);ax.set_ylabel('Audited-gallery purity');ax.set_title('Human semantic calibration of rule galleries');ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/'Figure_feedback_gallery_purity.png',dpi=300);plt.close(fig)
    print(json.dumps(met,indent=2))

if __name__=='__main__':main()

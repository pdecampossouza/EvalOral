#!/usr/bin/env python3
"""Recreate the paper-linked rule-history summaries and representative galleries.

The default mode uses the exact rule IDs/final maximum-coverage assignments bundled
under data/reference. These are manuscript reference outputs, not prequential
predictions. Use experiments/03_primary_30_streams.py for predictive evaluation.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from evaloral.data import repo_root


def main():
    root=repo_root(); out=root/'results/generated/07_rule_history'; out.mkdir(parents=True,exist_ok=True)
    hist=pd.read_csv(root/'data/reference/rule_history_table6.csv')
    assign=pd.read_csv(root/'data/reference/final_rule_assignments_view121.csv')
    hist.to_csv(out/'rule_history_reference_table.csv',index=False)
    rule_ids=hist.rule_id.tolist()
    rows=[]
    for rid in rule_ids:
        g=assign[assign.final_rule_id==rid].sort_values('final_rule_coverage',ascending=False).head(5)
        for rank,(_,r) in enumerate(g.iterrows(),1):
            rows.append({'rule_id':rid,'rank':rank,'image_id':r.image_id,'released_view':r.released_view,
                         'coverage':r.final_rule_coverage,'filepath':r.filepath})
    gal=pd.DataFrame(rows); gal.to_csv(out/'rule_gallery_top5_by_coverage.csv',index=False)
    fig,axes=plt.subplots(len(rule_ids),5,figsize=(11,2.0*len(rule_ids)))
    if len(rule_ids)==1: axes=[axes]
    for i,rid in enumerate(rule_ids):
        g=gal[gal.rule_id==rid].sort_values('rank')
        h=hist[hist.rule_id==rid].iloc[0]
        for j in range(5):
            ax=axes[i][j]; ax.set_xticks([]); ax.set_yticks([])
            if j<len(g):
                r=g.iloc[j]; ax.imshow(Image.open(root/r.filepath).convert('RGB'))
                ax.set_title(f"{r.released_view} | cov={r.coverage:.2f}",fontsize=7)
            if j==0:
                ax.set_ylabel(f"R{rid} {h.consequent}\nsupport={h.support}\nbirth={h.birth_step}\nstale={h.stale_steps}",fontsize=8)
    fig.suptitle('Representative photographs linked to high-support evolving fuzzy rules',fontsize=12)
    fig.tight_layout(); fig.savefig(out/'Figure_rule_history_galleries.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    print(hist.to_string(index=False))

if __name__=='__main__': main()

#!/usr/bin/env python3
"""Export and visualize the literal fuzzy rules reported in the manuscript."""
import pandas as pd,matplotlib.pyplot as plt
from evaloral.data import repo_root


def main():
    root=repo_root();out=root/'results/generated/13_literal_rules';out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(root/'data/reference/literal_rules_table10.csv');df.to_csv(out/'literal_rules_table10.csv',index=False)
    show=[7,13,12,10];fig,axes=plt.subplots(len(show),1,figsize=(10,7.6))
    for ax,rid in zip(axes,show):
        r=df[df.rule_id==rid].iloc[0];ax.axis('off');prem=(f"IF PC1~N({r.PC1_center:.3f},{r.PC1_sigma:.3f}); PC2~N({r.PC2_center:.3f},{r.PC2_sigma:.3f}); " f"PC3~N({r.PC3_center:.3f},{r.PC3_sigma:.3f}); PC4~N({r.PC4_center:.3f},{r.PC4_sigma:.3f})")
        ax.text(.01,.88,f'R{rid} — Gaussian premise retained',fontsize=11,fontweight='bold',va='top');ax.text(.01,.58,prem,fontsize=9.3,va='top',wrap=True);ax.text(.03,.30,f'BEFORE: THEN {r.before_feedback}',fontsize=10);ax.text(.03,.10,f'AFTER FEEDBACK: THEN {r.after_feedback} (human support {r.human_support:.2f})',fontsize=10,fontweight='bold');ax.axhline(.02,linewidth=.7)
    fig.suptitle('Literal evolving fuzzy rules before and after semantic feedback');fig.tight_layout();fig.savefig(out/'Figure_literal_rules_before_after.png',dpi=300,bbox_inches='tight');plt.close(fig);print(df.to_string(index=False))

if __name__=='__main__':main()

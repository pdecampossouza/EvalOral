from __future__ import annotations
import numpy as np
import pandas as pd


def snapshot_frame(model, step_label=None):
    rows=model.rule_snapshot()
    df=pd.DataFrame(rows)
    if step_label is not None:df.insert(0,'monitor_round',step_label)
    return df


def final_assignments(model,Z,meta):
    cov=model.rule_coverages(Z);ri=np.argmax(cov,axis=1)
    rows=[]
    for i,(_,m) in enumerate(meta.reset_index(drop=True).iterrows()):
        rule=model.rules[int(ri[i])]
        rows.append({'image_id':m.image_id,'volunteer_id':getattr(m,'volunteer_id',''),'released_view':getattr(m,'released_view',''),'final_rule_id':rule.rule_id,'final_rule_coverage':float(cov[i,ri[i]])})
    return pd.DataFrame(rows)


def literal_rules(model):
    w=model.feature_weights_
    rows=[]
    for r in model.rules:
        rows.append({'rule_id':r.rule_id,'support':r.support,'birth_step':r.birth_step,'last_update_step':r.last_update_step,'consequent':model.class_names[r.dominant_class],'confidence':float(r.class_probs.max()),**{f'PC{j+1}_center':float(r.center[j]) for j in range(len(r.center))},**{f'PC{j+1}_sigma':float(r.sigma[j]) for j in range(len(r.sigma))},**{f'PC{j+1}_relevance':float(w[j]) for j in range(len(w))},'rule_text':model.rule_text(r)})
    return pd.DataFrame(rows)

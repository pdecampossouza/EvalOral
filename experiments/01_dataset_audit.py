#!/usr/bin/env python3
from pathlib import Path
import pandas as pd,json
from evaloral.data import repo_root,load_source,load_unique,load_view121

def main():
 root=repo_root();X,m=load_source(root);Xu,u=load_unique(root);Xv,y,v=load_view121(root)
 out=root/'results/generated/01_dataset_audit';out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for mod,g in m.groupby('source_module'):rows.append({'source_module':mod,'records':len(g),'content_unique':g.sha256.nunique()})
 pd.DataFrame(rows).to_csv(out/'module_counts.csv',index=False)
 dup=m.groupby('sha256').filter(lambda g:len(g)>1).sort_values(['sha256','source_module']);dup.to_csv(out/'cross_module_duplicate_records.csv',index=False)
 summary={'source_records':len(m),'content_unique':len(u),'module_qualified_volunteers':int(m.groupby(['source_module','volunteer']).ngroups),'view121':len(v),'frontal':int((v.released_view=='frontal').sum()),'occlusal':int((v.released_view=='occlusal').sum()),'duplicate_record_excess':int(len(m)-len(u))}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()

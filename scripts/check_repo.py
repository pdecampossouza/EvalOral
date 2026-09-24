#!/usr/bin/env python3
"""Fast integrity check for the bundled EvalOral release."""
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,cond,detail=''):
    checks.append((name,bool(cond),detail));print(f"[{'OK' if cond else 'FAIL'}] {name}"+(f': {detail}' if detail else ''))

src=pd.read_csv(ROOT/'data/processed/source_manifest.csv');uni=pd.read_csv(ROOT/'data/processed/content_unique_manifest.csv');v=pd.read_csv(ROOT/'data/processed/view121_manifest.csv')
check('441 source records',len(src)==441,str(len(src)))
check('257 content-unique photographs',len(uni)==257,str(len(uni)))
check('121 view-labelled anchor images',len(v)==121,str(len(v)))
check('56 frontal',int((v.released_view=='frontal').sum())==56,str((v.released_view=='frontal').sum()))
check('65 occlusal',int((v.released_view=='occlusal').sum())==65,str((v.released_view=='occlusal').sum()))
X=np.load(ROOT/'data/processed/features_source441.npy');Xu=np.load(ROOT/'data/processed/features_unique257.npy');Xv=np.load(ROOT/'data/processed/features_view121.npy')
check('source feature shape',X.shape==(441,6272),str(X.shape));check('unique feature shape',Xu.shape==(257,6272),str(Xu.shape));check('view feature shape',Xv.shape==(121,6272),str(Xv.shape))
fb=pd.read_csv(ROOT/'data/reference/author_feedback40.csv');check('40 feedback exemplars',len(fb)==40,str(len(fb)))
counts=fb.corrected_view.value_counts().to_dict();check('feedback label distribution',counts=={'lateral':14,'occlusal':14,'frontal':12},str(counts))
missing=[r.filepath for _,r in src.iterrows() if not (ROOT/r.filepath).is_file()];check('all canonical raw files present',not missing,f'{len(missing)} missing')
# Hash only the smaller 121 anchor here; experiment 00 verifies all 441.
bad=[]
for _,r in v.iterrows():
    p=ROOT/r.filepath
    if hashlib.sha256(p.read_bytes()).hexdigest()!=r.sha256: bad.append(r.image_id)
check('anchor SHA-256 hashes',not bad,f'{len(bad)} mismatch(es)')
status={'passed':sum(x[1] for x in checks),'total':len(checks),'failed':[x[0] for x in checks if not x[1]]}
print(json.dumps(status,indent=2));sys.exit(0 if not status['failed'] else 1)

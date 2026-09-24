#!/usr/bin/env python3
"""Verify the bundled canonical data and optionally recompute 6,272-D features."""
from pathlib import Path
import argparse,hashlib
import numpy as np,pandas as pd
from evaloral.data import repo_root,dataset_summary
from evaloral.features import compute_feature

def main():
 p=argparse.ArgumentParser();p.add_argument('--recompute-features',action='store_true');a=p.parse_args();root=repo_root();meta=pd.read_csv(root/'data/processed/source_manifest.csv')
 print(dataset_summary(root))
 missing=[r.filepath for _,r in meta.iterrows() if not (root/r.filepath).exists()];print('missing raw images:',len(missing))
 bad=[]
 for _,r in meta.iterrows():
  h=hashlib.sha256((root/r.filepath).read_bytes()).hexdigest()
  if h!=r.sha256:bad.append(r.image_id)
 print('hash mismatches:',len(bad))
 if a.recompute_features:
  X=np.stack([compute_feature(root/p) for p in meta.filepath]).astype('float32');np.save(root/'data/processed/features_source441_recomputed.npy',X);print('saved',X.shape)
if __name__=='__main__':main()

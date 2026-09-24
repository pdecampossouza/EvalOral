import numpy as np,pandas as pd
from evaloral.data import repo_root
from evaloral.features import compute_feature

def test_feature_reconstruction_one_image():
    root=repo_root();m=pd.read_csv(root/'data/processed/source_manifest.csv');X=np.load(root/'data/processed/features_source441.npy')
    x=compute_feature(root/m.iloc[0].filepath)
    assert x.shape==(6272,)
    # Feature cache was produced by the same deterministic transform.
    assert np.allclose(x,X[0],rtol=1e-5,atol=1e-6)

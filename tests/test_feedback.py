import pandas as pd
from evaloral.data import repo_root
from evaloral.feedback import semantic_gallery_replay

def test_feedback_reference_and_replay():
    root=repo_root();fb=pd.read_csv(root/'data/reference/author_feedback40.csv')
    assert len(fb)==40
    assert fb.corrected_view.value_counts().to_dict()=={'lateral':14,'occlusal':14,'frontal':12}
    _,m=semantic_gallery_replay(fb)
    assert abs(m['original_accuracy']-.575)<1e-12
    assert abs(m['feedback_loo_accuracy']-.75)<1e-12

from evaloral.data import load_view121
from evaloral.protocols import StreamConfig,run_one_stream

def test_one_primary_stream_smoke():
    X,y,m=load_view121();g=m.volunteer.astype(int).to_numpy()
    r=run_one_stream(X,y,g,20260824,StreamConfig())
    assert 0<=r['macro_f1']<=1
    assert 0<=r['accuracy']<=1
    assert r['final_rules']>=1

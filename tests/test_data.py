from evaloral.data import repo_root,load_source,load_unique,load_view121

def test_canonical_counts_and_shapes():
    X,m=load_source(); Xu,u=load_unique(); Xv,y,v=load_view121()
    assert len(m)==441 and X.shape==(441,6272)
    assert len(u)==257 and Xu.shape==(257,6272)
    assert len(v)==121 and Xv.shape==(121,6272)
    assert (v.released_view=='frontal').sum()==56
    assert (v.released_view=='occlusal').sum()==65
    assert set(y)=={0,1}

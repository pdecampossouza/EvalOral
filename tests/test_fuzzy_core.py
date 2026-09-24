import numpy as np
from evaloral.evolving_nf import EvolvingNeuroFuzzyEvalOral,gaussian_memberships_per_feature,UnimNary

def test_gaussian_and_unim_ranges():
    a=gaussian_memberships_per_feature(np.array([0.,1.]),np.zeros(2),np.ones(2))
    assert np.all((0<=a)&(a<=1))
    u=UnimNary(.5)
    assert u.aggregate(np.array([.2,.3]))[1]=='AND'
    assert u.aggregate(np.array([.7,.8]))[1]=='OR'
    assert u.aggregate(np.array([.2,.8]))[1]=='COMP'

def test_model_creates_and_predicts_rules():
    X=np.array([[-2.,0.],[-1.8,.1],[2.,0.],[1.8,-.1]])
    y=np.array([0,0,1,1])
    m=EvolvingNeuroFuzzyEvalOral(2,2,alpha_add=.4,sigma_init=1.0,activation_mode='coverage_gated_unim',label_conflict_policy='new_rule',random_state=1)
    m.partial_fit(X,y)
    p=m.predict(X)
    assert len(m.rules)>=2
    assert p.shape==(4,)
    assert m.rule_coverages(X).shape==(4,len(m.rules))

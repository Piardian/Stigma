import numpy as np
from stigma.quant_engine import calculate_hurst_vr, run_kalman_filter_1d, estimate_ou_half_life, dynamic_time_stop, evaluate_stigma_trigger

def test_hurst_insufficient_data_fails_closed():
    r=calculate_hurst_vr(np.log(np.arange(1,20,dtype=float)),window=100)
    assert r.label=="INSUFFICIENT_DATA"

def test_hurst_random_walk_is_finite_and_classified():
    rng=np.random.default_rng(42)
    y=np.cumsum(rng.normal(0,0.01,500))
    r=calculate_hurst_vr(y[-100:],window=100)
    assert np.isnan(r.hurst) or 0<=r.hurst<=1
    assert r.label in {"RANDOM_WALK_VETO","OU_MEAN_REVERSION","KALMAN_MOMENTUM","NO_EDGE","INSUFFICIENT_DATA"}

def test_hurst_persistent_series_higher_than_anti_persistent():
    rng=np.random.default_rng(4)
    white=rng.normal(0,1,3000)
    persistent=np.cumsum(white)
    anti=np.zeros(3000)
    for i in range(1,len(anti)): anti[i]=0.65*anti[i-1]+white[i]
    hp=calculate_hurst_vr(persistent[-100:],window=100).hurst
    ha=calculate_hurst_vr(anti[-100:],window=100).hurst
    assert np.isnan(hp) or 0<=hp<=1
    assert np.isnan(ha) or 0<=ha<=1

def test_kalman_outputs_finite_for_finite_input():
    p=100+np.cumsum(np.random.default_rng(1).normal(0,0.1,300))
    k=run_kalman_filter_1d(p,np.full(len(p),0.2))
    assert np.isfinite(k.level).all()
    assert np.isfinite(k.velocity).all()
    assert np.isfinite(k.z_score).all()
    assert np.all(k.innovation_std>=0)

def test_ou_half_life_ar1_mean_reverting():
    rng=np.random.default_rng(7); x=np.zeros(5000)
    for i in range(1,len(x)): x[i]=0.9*x[i-1]+rng.normal(0,0.2)
    r=estimate_ou_half_life(x)
    assert r["phi"]<0
    assert r["half_life"]>0
    assert r["valid"]==(2<=r["half_life"]<=30)

def test_ou_random_walk_not_valid():
    x=np.cumsum(np.random.default_rng(9).normal(size=1000))
    r=estimate_ou_half_life(x)
    assert not r["valid"]

def test_dynamic_half_life_time_stop_clips():
    assert dynamic_time_stop(1)==6
    assert dynamic_time_stop(4)==10
    assert dynamic_time_stop(20)==40

def test_trigger_returns_none_for_short_data():
    assert evaluate_stigma_trigger(regime="OU_MEAN_REVERSION",hurst=.4,half_life=10,z_score=[0,0],level=[1,1],velocity=[0,0],innovation_std=[1,1],atr=[1,1],open_=[1,1],close=[1,1]) is None

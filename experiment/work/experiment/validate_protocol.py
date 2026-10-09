"""Checks needed to interpret the generated problem before training."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from synthetic import generate
from models import make_model, count_parameters
from common import save_json

def main():
    cases=[]
    for rate in [.01,.20]:
      for lag in [2,64]:
        x,y,m=generate(512,300001,rate,lag); score=np.broadcast_to(m,y.shape).astype(bool)
        oracle=np.zeros_like(y);oracle[:,lag:]=x[:,lag:,1]*(x[:,:-lag,0]>0)
        assert np.array_equal(y,oracle)
        small=generate(64,100000,rate,lag);big=generate(256,100000,rate,lag)
        assert np.array_equal(small[0],big[0][:64]) and np.array_equal(small[1],big[1][:64])
        cases.append(dict(rate=rate,lag=lag,test_prevalence=float(y[score].mean()),
                          oracle_ap=float(average_precision_score(y[score],oracle[score])),
                          query_only_ap=float(average_precision_score(y[score],x[:,:,1][score])),
                          query_count=int(x[:,:,1][score].sum()),positive_count=int(y[score].sum())))
    dims={d:{name:count_parameters(make_model(name,d)) for name in ['mlp','tcn','gru','mamba']} for d in [8,54]}
    for counts in dims.values(): assert max(counts.values())/min(counts.values())<1.05
    for lag in [2,64]:
        a=generate(512,300001,.01,lag); b=generate(512,300001,.2,lag)
        assert np.array_equal(a[0][:,:,0],b[0][:,:,0]);assert np.all(a[0][:,:,1]<=b[0][:,:,1])
    save_json('work/experiment/protocol_validation.json',dict(cases=cases,parameters=dims,
              nested_subsets=True,paired_density=True,no_test_metrics_from_learned_models=True))
    print(json.dumps({'checks':'passed','parameters':dims,'cases':cases}))
if __name__=='__main__':main()

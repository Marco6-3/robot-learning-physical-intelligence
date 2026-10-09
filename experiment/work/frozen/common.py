"""Fixed-budget training and record-aware evaluation utilities."""
from __future__ import annotations
import copy, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, roc_auc_score

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def best_threshold(y,p):
    precision,recall,threshold=precision_recall_curve(y,p)
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(threshold[np.argmax(f)]) if len(threshold) else 0.5

def metrics(y,p,threshold):
    y=np.asarray(y).astype(int).reshape(-1); p=np.asarray(p).reshape(-1)
    yh=p>=threshold
    tp=int(np.sum(yh & (y==1))); fp=int(np.sum(yh & (y==0))); fn=int(np.sum(~yh & (y==1)))
    return dict(ap=float(average_precision_score(y,p)) if y.sum() else 0.,
                f1=float(f1_score(y,yh,zero_division=0)), prevalence=float(y.mean()),
                precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
                brier=float(np.mean((p-y)**2)),tp=tp,fp=fp,fn=fn,n=int(len(y)),threshold=float(threshold))

@torch.no_grad()
def predict(model,x,batch=32):
    model.eval(); out=[]
    for s in range(0,len(x),batch):
        out.append(torch.sigmoid(model(torch.as_tensor(x[s:s+batch],device=DEVICE))).cpu().numpy())
    return np.concatenate(out)

def train(model,x,y,mask,val_x,val_y,val_mask,seed,lr,steps=400,batch=16,eval_every=100):
    """Same update count and data for every architecture; select checkpoint only on validation AP."""
    seed_all(seed); model=model.to(DEVICE)
    tx=torch.as_tensor(x,device=DEVICE); ty=torch.as_tensor(y,device=DEVICE)
    tm=torch.as_tensor(np.broadcast_to(mask,y.shape).copy(),device=DEVICE)
    npos=float((ty*tm).sum()); nneg=float(tm.sum())-npos
    pos_weight=torch.tensor(nneg/max(npos,1),device=DEVICE)
    # Official Mamba A_log/D no-weight-decay and common bias/norm treatment.
    from models import optimizer_parameter_groups
    opt=torch.optim.AdamW(optimizer_parameter_groups(model,1e-4),lr=lr)
    rng=np.random.default_rng(seed+1729); best=-1.; best_state=None; best_step=None; trace=[]
    if DEVICE=='cuda': torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0=time.perf_counter()
    vm=np.broadcast_to(val_mask,val_y.shape).astype(bool)
    for step in range(1,steps+1):
        model.train(); idx=torch.as_tensor(rng.integers(0,len(x),size=batch),device=DEVICE)
        opt.zero_grad(set_to_none=True); logits=model(tx[idx])
        raw=torch.nn.functional.binary_cross_entropy_with_logits(logits,ty[idx],reduction='none',pos_weight=pos_weight)
        loss=(raw*tm[idx]).sum()/tm[idx].sum().clamp_min(1)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step%eval_every==0 or step==steps:
            vp=predict(model,val_x); ap=float(average_precision_score(val_y[vm],vp[vm]))
            trace.append({'step':step,'train_loss':float(loss.detach()),'val_ap':ap})
            if ap>best:
                best=ap; best_step=step; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if DEVICE=='cuda': torch.cuda.synchronize()
    seconds=time.perf_counter()-t0
    peak=int(torch.cuda.max_memory_allocated()) if DEVICE=='cuda' else None
    model.load_state_dict(best_state)
    return model,dict(trace=trace,best_val_ap=best,best_step=best_step,model_config=model.config,
                      parameters=sum(p.numel() for p in model.parameters()),seconds=seconds,peak_allocated_bytes=peak,
                      pos_weight=float(pos_weight),steps=steps,batch=batch,lr=lr,device=DEVICE)

def spans(y):
    z=np.r_[0,np.asarray(y,dtype=int),0]; d=np.diff(z)
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))

def event_metrics(y,p,threshold,hz):
    """One-to-one overlap event matching; latency only among detected true events."""
    actual=spans(y); alarms=spans(np.asarray(p)>=threshold)
    matches=[]; used=set()
    for a,b in actual:
        candidates=[(max(a,c),i,c,d) for i,(c,d) in enumerate(alarms) if i not in used and max(a,c)<min(b,d)]
        if candidates:
            _,i,c,d=min(candidates); used.add(i); matches.append(max(0,c-a)/hz)
    tp=len(matches); fp=len(alarms)-tp; fn=len(actual)-tp
    return dict(event_tp=tp,event_fp=fp,event_fn=fn,event_f1=2*tp/max(2*tp+fp+fn,1),
                false_alarms_per_min=fp/(len(y)/hz/60),
                detected_latency_sec_mean=float(np.mean(matches)) if matches else None,
                detected_latencies_sec=matches, duration_sec=len(y)/hz)

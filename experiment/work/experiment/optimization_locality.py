"""Fresh-data interventions on sampling and causal local convolution.

No fitting before --precheck seals the declared protocol and dependencies.
The two samplers estimate the same finite-pool weighted BCE risk; their Adam
updates, clipped gradients and positive presentation counts are not identical.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from common import DEVICE, seed_all, save_json, sha256, best_threshold, metrics
from models import make_model, count_parameters, optimizer_parameter_groups
from cue_memory import generate, shuffle_cue_bit

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'outputs' / '补充协议_采样与局部卷积.md'
SEAL = PROTOCOL.with_name(PROTOCOL.stem + '_登记.json')
VARIANTS = ('gru', 'gru_conv1', 'gru_conv4', 'mamba_conv4', 'mamba_conv1')
SAMPLERS = ('uniform', 'balanced')
SEEDS = list(range(10))
LRS = (0.001, 0.003)
N_TRAIN, N_VAL, N_TEST = 4096, 8192, 65536
STEPS, BATCH = 400, 32


class ConvGRU(nn.Module):
    def __init__(self, base, kernel):
        super().__init__()
        self.base = base
        self.conv = nn.Conv1d(32, 32, kernel, groups=32, padding=kernel-1)
        self.config = {**base.config, 'name': f'gru_conv{kernel}', 'causal_depthwise_kernel': kernel,
                       'activation': 'SiLU', 'convolution_position': 'after input projection, before GRU'}

    def forward(self, x):
        z = self.base.input_proj(x).transpose(1, 2)
        z = F.silu(self.conv(z)[..., :x.shape[1]].transpose(1, 2))
        states, _ = self.base.gru(z)
        return self.base.head(states).squeeze(-1)


def family(variant):
    return 'gru' if variant.startswith('gru') else 'mamba'


def construct(variant, seed):
    """Explicitly pair common parameter tensors, despite different RNG use."""
    seed_all(seed)
    if family(variant) == 'gru':
        base = make_model('gru', 8)
        if variant == 'gru':
            return base
        # Both start from the exact same GRU base. All common-shaped conv
        # parameters (bias) are copied from a conv4 template; kernel weights
        # retain PyTorch's standard fan-in initialization for their own shape.
        template = ConvGRU(base, 4)
        if variant == 'gru_conv4':
            return template
        seed_all(seed + 8123)
        target = ConvGRU(base, 1)
        with torch.no_grad():
            target.conv.bias.copy_(template.conv.bias)
        return target
    template = make_model('mamba', 8, d_conv=4)
    if variant == 'mamba_conv4':
        template.config['name'] = variant
        return template
    seed_all(seed + 8123)
    target = make_model('mamba', 8, d_conv=1)
    source = template.state_dict()
    state = target.state_dict()
    for name in state:
        if state[name].shape == source[name].shape:
            state[name] = source[name].clone()
    target.load_state_dict(state)
    target.config['name'] = variant
    return target


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def batches(y, seed, sampler, steps=STEPS):
    rng = np.random.default_rng(seed + 1729)
    pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        raise ValueError('Both empirical classes are required')
    for _ in range(steps):
        if sampler == 'uniform':
            yield rng.integers(0, len(y), size=BATCH)
        elif sampler == 'balanced':
            # Order has no effect: models have no cross-record operations.
            yield np.concatenate([rng.choice(pos, BATCH//2), rng.choice(neg, BATCH//2)])
        else:
            raise ValueError(sampler)


@torch.no_grad()
def endpoint_predict(model, x, meta, batch=128):
    model.eval()
    out = []
    for start in range(0, len(x), batch):
        end = min(start+batch, len(x))
        logits = model(torch.as_tensor(x[start:end], device=DEVICE))
        rows = torch.arange(end-start, device=DEVICE)
        endpoints = torch.as_tensor(meta['endpoint'][start:end], device=DEVICE)
        out.append(logits[rows, endpoints].sigmoid().cpu().numpy())
    return np.concatenate(out)


def train_one(variant, sampler, seed, lr, x, meta, vx, vmeta):
    model = construct(variant, seed)
    initial_hash = state_hash(model)
    seed_all(seed)
    model = model.to(DEVICE)
    tx = torch.as_tensor(x, device=DEVICE)
    ty = torch.as_tensor(meta['y'], device=DEVICE)
    te = torch.as_tensor(meta['endpoint'], device=DEVICE)
    pi = float(meta['y'].mean())
    pos_weight = torch.tensor((1-pi)/pi, device=DEVICE)
    opt = torch.optim.AdamW(optimizer_parameter_groups(model, 1e-4), lr=lr)
    best, best_step, best_state = -1., None, None
    trace, batch_positive_counts = [], []
    visits = np.zeros(len(x), dtype=np.int64)
    index_hash = hashlib.sha256()
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    rows = torch.arange(BATCH, device=DEVICE)
    for step, idx_np in enumerate(batches(meta['y'], seed, sampler), 1):
        index_hash.update(idx_np.astype(np.int64).tobytes())
        np.add.at(visits, idx_np, 1)
        batch_positive_counts.append(int(meta['y'][idx_np].sum()))
        idx = torch.as_tensor(idx_np, device=DEVICE)
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(tx[idx])[rows, te[idx]]
        if sampler == 'uniform':
            loss = F.binary_cross_entropy_with_logits(logits, ty[idx], pos_weight=pos_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, ty[idx]) * (2*(1-pi))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient')
        opt.step()
        if step % 100 == 0:
            vp = endpoint_predict(model, vx, vmeta)
            ap = float(average_precision_score(vmeta['y'], vp))
            trace.append({'step': step, 'train_loss': float(loss.detach()), 'val_ap': ap})
            if ap > best:
                best, best_step = ap, step
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    peak = int(torch.cuda.max_memory_allocated()) if DEVICE == 'cuda' else None
    model.load_state_dict(best_state)
    positive = meta['y'] > 0
    info = dict(trace=trace, best_val_ap=best, best_step=best_step, steps=STEPS, batch=BATCH,
                lr=lr, device=DEVICE, seconds=seconds, peak_allocated_bytes=peak,
                initial_state_sha256=initial_hash, batch_indices_sha256=index_hash.hexdigest(),
                train_prevalence=pi, train_positives=int(positive.sum()), pos_weight=float(pos_weight),
                balanced_loss_multiplier=2*(1-pi), batch_positive_counts=batch_positive_counts,
                positive_presentations=int(np.array(batch_positive_counts).sum()),
                total_presentations=STEPS*BATCH,
                zero_positive_batches=int((np.array(batch_positive_counts)==0).sum()),
                unique_positive_visited=int(((visits>0)&positive).sum()),
                unique_negative_visited=int(((visits>0)&~positive).sum()),
                visits_sha256=hashlib.sha256(visits.tobytes()).hexdigest())
    return model, info


def dependencies():
    return [Path(__file__), PROTOCOL, Path(__file__).with_name('models.py'),
            Path(__file__).with_name('common.py'), Path(__file__).with_name('cue_memory.py')]


def precheck():
    if SEAL.exists():
        raise RuntimeError('Existing registration must be preserved')
    report = {'training_launched': False, 'models': {}, 'checks': []}
    x, _, _, meta = generate(256, 510000, .01, 2)
    for name in VARIANTS:
        model = construct(name, 7).cpu()
        assert state_hash(model) == state_hash(construct(name, 7))
        tx = torch.as_tensor(x[:3])
        result = model(tx)
        altered = tx.clone()
        altered[:, 100:] += 4
        with torch.no_grad():
            torch.testing.assert_close(result[:, :100], model(altered)[:, :100], atol=1e-6, rtol=1e-5)
        result.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        report['models'][name] = {'parameters': count_parameters(model), 'config': model.config,
                                  'causal': True, 'finite_gradients': True}
    for seed in (0, 9, 901):
        base = construct('gru', seed).state_dict()
        for variant in ('gru_conv1', 'gru_conv4'):
            other = construct(variant, seed).base.state_dict()
            for key in base:
                torch.testing.assert_close(base[key], other[key], atol=0, rtol=0)
        for first, second in [('gru_conv4','gru_conv1'), ('mamba_conv4','mamba_conv1')]:
            a, b = construct(first,seed).state_dict(), construct(second,seed).state_dict()
            for key in a:
                if a[key].shape == b[key].shape:
                    torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    # Direct numerical check of the same finite empirical objective and its
    # un-clipped full-pool gradient, separate from stochastic Adam dynamics.
    y = np.array([1,1,1,0,0,0,0,0,0,0], dtype=np.float64)
    logits = torch.linspace(-2,2,10,dtype=torch.float64,requires_grad=True)
    ty = torch.as_tensor(y)
    pi = float(y.mean())
    raw = F.binary_cross_entropy_with_logits(logits,ty,reduction='none')
    u = (raw*torch.where(ty>0, torch.full_like(ty,(1-pi)/pi), torch.ones_like(ty))).mean()
    b = (1-pi)*(raw[ty>0].mean()+raw[ty==0].mean())
    torch.testing.assert_close(u,b,atol=1e-14,rtol=1e-14)
    gu = torch.autograd.grad(u,logits,retain_graph=True)[0]
    gb = torch.autograd.grad(b,logits)[0]
    torch.testing.assert_close(gu,gb,atol=1e-14,rtol=1e-14)
    _, _, _, tm = generate(N_TRAIN,510000,.01,2)
    bs = list(batches(tm['y'],0,'balanced'))
    assert all(tm['y'][indices].sum()==16 for indices in bs)
    report['checks'] = ['all common-shaped parameters explicitly paired',
                        'same variant/seed has identical initial state',
                        'no future-input leakage', 'balanced batches exactly16/16',
                        'same empirical risk and pre-clipping gradient expectation']
    report['data'] = {}
    for role, n, seed in [('train0',N_TRAIN,510000),('tune',N_TRAIN,510901),
                          ('validation',N_VAL,520001),('test',N_TEST,530001)]:
        _,_,_,m = generate(n,seed,.01,2)
        report['data'][role] = dict(n=n,seed=seed,positives=int(m['y'].sum()),
                                    queries=int(m['query'].sum()),
                                    query_only_ap=float(average_precision_score(m['y'],m['query'])))
    save_json(Path(__file__).with_name('optimization_locality_precheck.json'),report)
    save_json(SEAL, {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                    'status': 'adaptive follow-up declared before new data model fitting',
                    'previous_evidence': 'v1 and real results, v2 sparse short seed instability and preliminary long floor',
                    'files': {str(p.relative_to(ROOT)):sha256(p) for p in dependencies()},
                    'planned_tuning_runs':20,'planned_main_runs':100,'training_launched':False})
    print(json.dumps(report,ensure_ascii=False,indent=2))


def verify_seal():
    seal = json.loads(SEAL.read_text(encoding='utf-8'))
    for path, digest in seal['files'].items():
        if sha256(ROOT/path) != digest:
            raise RuntimeError(f'Sealed dependency changed: {path}')
    return seal


def run(out):
    seal = verify_seal()
    out.mkdir(parents=True,exist_ok=True)
    manifest = dict(task='sampling-and-locality-v3', variants=list(VARIANTS), samplers=list(SAMPLERS),
                    seeds=SEEDS,rate=.01,lag=2,n_train=N_TRAIN,n_val=N_VAL,n_test=N_TEST,
                    steps=STEPS,batch=BATCH,learning_rates=list(LRS),
                    train_data_seed_base=510000,tune_data_seed=510901,val_data_seed=520001,test_data_seed=530001,
                    lr_selection='one LR per family, mean best validation AP across all family variants and samplers',
                    registration=seal,planned_tuning_runs=20,planned_main_runs=100)
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text(encoding='utf-8')) == manifest
    else:
        save_json(out/'manifest.json',manifest)
    vx,_,_,vm = generate(N_VAL,520001,.01,2)
    tx,_,_,tm = generate(N_TRAIN,510901,.01,2)
    trials=[]
    for variant in VARIANTS:
        for sampler in SAMPLERS:
            for lr in LRS:
                path=out/f'tune_{variant}_{sampler}_lr{lr:g}.json'
                if path.exists():
                    trial=json.loads(path.read_text(encoding='utf-8'))
                else:
                    model,training=train_one(variant,sampler,901,lr,tx,tm,vx,vm)
                    trial=dict(variant=variant,sampler=sampler,seed=901,lr=lr,training=training,
                               parameters=count_parameters(model),config=model.config)
                    save_json(path,trial)
                    print(json.dumps({'tuned':variant,'sampler':sampler,'lr':lr,'val_ap':training['best_val_ap']}),flush=True)
                    del model
                trials.append(trial)
    family_selection={}
    for name in ('gru','mamba'):
        scores={str(lr):float(np.mean([t['training']['best_val_ap'] for t in trials
                                       if family(t['variant'])==name and t['lr']==lr])) for lr in LRS}
        family_selection[name]={'selected_lr':float(max(scores,key=scores.get)),'mean_validation_ap':scores}
    save_json(out/'family_lr_selection.json',family_selection)
    del tx,tm
    ex,_,_,em=generate(N_TEST,530001,.01,2)
    shuffled=shuffle_cue_bit(ex,em)
    for seed in SEEDS:
        tx,_,_,tm=generate(N_TRAIN,510000+seed,.01,2)
        for variant in VARIANTS:
            for sampler in SAMPLERS:
                key=f'{variant}_{sampler}_seed{seed}'
                path=out/(key+'.json')
                if path.exists():
                    continue
                lr=family_selection[family(variant)]['selected_lr']
                model,training=train_one(variant,sampler,seed,lr,tx,tm,vx,vm)
                vp=endpoint_predict(model,vx,vm)
                threshold=best_threshold(vm['y'],vp)
                ep=endpoint_predict(model,ex,em)
                sp=endpoint_predict(model,shuffled,em)
                qm=em['query']>0
                result_metrics=metrics(em['y'],ep,threshold)
                result_metrics.update(query_ap=float(average_precision_score(em['y'][qm],ep[qm])),
                                      cue_shuffled_ap=float(average_precision_score(em['y'],sp)),
                                      cue_shuffled_query_ap=float(average_precision_score(em['y'][qm],sp[qm])))
                np.savez_compressed(out/(key+'_predictions.npz'),y=em['y'],p=ep,query=em['query'],
                                    cue_shuffled_p=sp,endpoint=em['endpoint'],cue=em['cue'],bit=em['bit'])
                torch.save(model.state_dict(),out/(key+'.pt'))
                result=dict(task='sampling-and-locality-v3',variant=variant,sampler=sampler,seed=seed,
                            n_train=N_TRAIN,rate=.01,lag=2,parameters=count_parameters(model),config=model.config,
                            metrics=result_metrics,training=training)
                save_json(path,result)
                print(json.dumps({'completed':key,'ap':result_metrics['ap'],'query_ap':result_metrics['query_ap'],
                                  'positive_presentations':training['positive_presentations']}),flush=True)
                del model


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--precheck',action='store_true')
    group.add_argument('--run',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'work'/'results'/'optimization_locality_v3')
    args=parser.parse_args()
    precheck() if args.precheck else run(args.out)

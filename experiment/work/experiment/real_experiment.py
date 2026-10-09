"""Preregistered external real-data detection experiment; no causally generated labels.

ARQ-CRISP 2021 is observational, naturally moderate-prevalence slip data. It does
not identify causal memory demand or establish the sparse-1% synthetic hypothesis.
Run --prepare-only to freeze the configuration and audit chunks without training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback
import hashlib
import platform
import numpy as np
import torch
from common import DEVICE, seed_all, save_json, sha256, train, predict, metrics, best_threshold, spans
from models import make_model, count_parameters

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "real_data"
OUT = HERE.parent / "results" / "real"
MODELS = ["mlp", "tcn", "gru", "mamba"]
LRS = [.001, .003]
SEEDS = list(range(5))
TUNE_SEED = 901
STEPS, BATCH, LENGTH, WARMUP, STRIDE = 400, 16, 128, 64, 64

def log(event, **kwargs):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"time_unix": time.time(), "event": event, **kwargs}
    with (OUT / "run_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)

def data_records():
    return json.loads((DATA / "arq2021_audit.json").read_text(encoding="utf-8"))["trials"]

def load_trials():
    """Six-sample trailing mean at original endpoints 5,11,...; no future inputs."""
    trials = {}
    for record in data_records():
        path = DATA / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"Processed data hash changed: {path}")
        with np.load(path) as z:
            n = len(z["x"]) // 6
            endpoints = np.arange(5, 6*n, 6)
            x = z["x"][:6*n].reshape(n, 6, 54).mean(axis=1)
            y = z["slip"][endpoints].astype(np.float32)
            valid = z["valid"][:6*n].reshape(n, 6).all(axis=1)
            timestamps = z["timestamp_s"][endpoints]
            # Actual covered time, not n / nominal_hz. First bin spans five
            # observed intervals; following bins span six intervals each.
            coverage = np.diff(np.r_[z["timestamp_s"][0], timestamps])
            trials[record["trial_id"]] = {"x": x, "y": y, "valid": valid,
                "timestamp_s": timestamps, "coverage_s": coverage,
                "source_endpoint_index": endpoints, "record": record}
    return trials

def configurations():
    split = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
    index = {r["trial_id"]: r for r in data_records()}
    primary = split["held_trial"]
    result = []
    for budget in ["low", "full"]:
        training = [t for t in primary["train"] if budget == "full" or index[t]["experiment"] <= 2]
        result.append({"id": f"held_trial_{budget}", "suite": "held_trial", "budget": budget,
                       "train": training, "validation": primary["validation"], "test": primary["test"], "models":MODELS})
    for fold in split["leave_one_object_out"]:
        result.append({"id": "loo_"+fold["held_out_object"], "suite":"loo", "budget":"full",
                       **fold, "models":["gru","mamba"]})
    return result

def fit_scaler(trials, ids):
    # All input features from the selected training trials, including unlabeled
    # feature rows. Neither held-out features nor labels influence the scaler.
    x = np.concatenate([trials[i]["x"] for i in ids]).astype(np.float64)
    return x.mean(axis=0).astype(np.float32), np.maximum(x.std(axis=0), 1.0).astype(np.float32)

def chunks(trials, ids, mean, scale):
    xs, ys, masks, maps = [], [], [], []
    for trial_id in ids:
        trial = trials[trial_id]
        x = (trial["x"] - mean) / scale
        n = len(x)
        for score_start in range(0, n, STRIDE):
            start = score_start-WARMUP
            stop = score_start+STRIDE
            actual_start, actual_stop = max(start,0), min(stop,n)
            left = actual_start-start
            xx = np.zeros((LENGTH,54),dtype=np.float32)
            yy = np.zeros(LENGTH,dtype=np.float32)
            mask = np.zeros(LENGTH,dtype=bool)
            xx[left:left+actual_stop-actual_start] = x[actual_start:actual_stop]
            yy[left:left+actual_stop-actual_start] = trial["y"][actual_start:actual_stop]
            nscore = min(STRIDE,n-score_start)
            mask[WARMUP:WARMUP+nscore] = trial["valid"][score_start:score_start+nscore]
            xs.append(xx); ys.append(yy); masks.append(mask)
            maps.append({"trial_id":trial_id,"score_start":score_start,"score_count":nscore})
    return {"x":np.stack(xs),"y":np.stack(ys),"mask":np.stack(masks),"maps":maps}

def reconstruct(predictions, chunked, trials, ids):
    result = {i:np.full(len(trials[i]["y"]),np.nan,dtype=np.float32) for i in ids}
    seen = {i:np.zeros(len(trials[i]["y"]),dtype=np.int8) for i in ids}
    for p,m in zip(predictions,chunked["maps"]):
        trial_id, start, count = m["trial_id"],m["score_start"],m["score_count"]
        result[trial_id][start:start+count] = p[WARMUP:WARMUP+count]
        seen[trial_id][start:start+count] += 1
    for i in ids:
        assert (seen[i] == 1).all() and np.isfinite(result[i]).all(), f"Duplicate/missing endpoint: {i}"
    return result

def event_metrics_timestamp(trial, p, threshold):
    """Greedy one-to-one overlapping event matches within contiguous valid runs.

    Latency is conditional on a detected event; an alarm already active at the
    labeled onset has zero latency. Unknown-label runs cannot bridge an event.
    """
    y,t,valid = trial["y"],trial["timestamp_s"],trial["valid"]
    total_tp = total_fp = total_fn = 0
    latencies = []
    for first,last in spans(valid):
        actual, alarms = spans(y[first:last]), spans(p[first:last] >= threshold)
        used = set()
        for a,b in actual:
            candidates = [(max(a,c),i,c) for i,(c,d) in enumerate(alarms)
                          if i not in used and max(a,c) < min(b,d)]
            if candidates:
                _,i,c = min(candidates)
                used.add(i)
                latencies.append(float(max(0,t[first+c]-t[first+a])))
        total_tp += len(used); total_fp += len(alarms)-len(used); total_fn += len(actual)-len(used)
    duration = float(trial["coverage_s"][valid].sum())
    return {"event_tp":total_tp,"event_fp":total_fp,"event_fn":total_fn,
            "event_f1":2*total_tp/max(2*total_tp+total_fp+total_fn,1),
            "false_alarms_per_min":total_fp/(duration/60) if duration else None,
            "detected_latencies_sec":latencies,
            "detected_latency_sec_mean":float(np.mean(latencies)) if latencies else None,
            "duration_sec":duration}

def evaluate_trials(trials, ids, probabilities, threshold, directory):
    directory.mkdir(parents=True,exist_ok=True)
    rows=[]; all_y=[]; all_p=[]; latencies=[]
    for i in ids:
        tr,p = trials[i],probabilities[i]
        y,v = tr["y"],tr["valid"]
        m=metrics(y[v],p[v],threshold) if v.any() else None
        event=event_metrics_timestamp(tr,p,threshold)
        rows.append({"trial_id":i,"object":tr["record"]["object"],"frame":m,"event":event})
        all_y.append(y[v]);all_p.append(p[v]);latencies.extend(event["detected_latencies_sec"])
        np.savez_compressed(directory/(i+".npz"),p=p,y=y,valid=v,timestamp_s=tr["timestamp_s"],
                            coverage_s=tr["coverage_s"],source_endpoint_index=tr["source_endpoint_index"])
    frame = metrics(np.concatenate(all_y),np.concatenate(all_p),threshold)
    tp=sum(r["event"]["event_tp"] for r in rows)
    fp=sum(r["event"]["event_fp"] for r in rows)
    fn=sum(r["event"]["event_fn"] for r in rows)
    duration=sum(r["event"]["duration_sec"] for r in rows)
    return {"frame":frame,"event":{"event_tp":tp,"event_fp":fp,"event_fn":fn,
            "event_f1":2*tp/max(2*tp+fp+fn,1),"false_alarms_per_min":fp/(duration/60),
            "detected_latency_sec_mean":float(np.mean(latencies)) if latencies else None,
            "detected_latencies_sec":latencies,"duration_sec":duration},"trials":rows}

def build_manifest(trials):
    configs=configurations()
    # Parameter counts and code hashes freeze exact factory defaults before any training.
    model_sizes={name:count_parameters(make_model(name,54)) for name in MODELS}
    manifest={"protocol":"ARQ2021-real-slip-current-v1", "purpose":"External moderate-density current-slip detection; not proof of sparse-event or long-latent-memory mechanisms",
        "data_revision":json.loads((DATA/'sources/source_manifest.json').read_text())["revision"],
        "data_audit_sha256":sha256(DATA/'arq2021_audit.json'),"split_sha256":sha256(DATA/'splits.json'),
        "code_hashes":{p.name:sha256(p) for p in [Path(__file__),HERE/'common.py',HERE/'models.py']},
        "models_parameters":model_sizes,"configurations":configs,
        "training":{"learning_rates":LRS,"tuning_seed":TUNE_SEED,"final_seeds":SEEDS,"steps":STEPS,"batch":BATCH,
                    "checkpoint_selection":"maximum validation average precision every100 updates", "tuning_selection":"higher validation AP; first listed lr wins exact tie",
                    "threshold":"maximum validation F1 separately for each final model/seed", "class_weight":"training-only negative/positive scored-label ratio; no test rebalancing"},
        "input":{"trailing_bin_width":6,"native_first_endpoint":5,"nominal_downsampled_hz":30,"features":"raw54 only",
                 "transform":"train-only per-channel global zscore, std floor1.0 raw sensor count", "chunk_length":LENGTH,
                 "warmup_masked":WARMUP,"scored_nonoverlap_steps":STRIDE,"padding":"zero AFTER normalization; never scored",
                 "labels":"current endpoint slip1or5; bin valid iff all6 native labels valid", "state_reset":"each128step chunk starts zero state; maximum represented input history4.27sec nominal, varies within scored half",
                 "timing":"actual endpoint timestamps and observed bin coverage; no fixedHz event timing"},
        "evaluation":{"primary":"held_trial low and full; all4 models","robustness":"leave-one-object-out3folds full, GRU/Mamba only",
                      "events":"one-to-one greedy overlap within each contiguous valid run; latency conditional on detection",
                      "inference_timing":"3 complete batched prediction passes median incl host/device transfer; implementation-specific"},
        "runtime":{"python":platform.python_version(),"torch":torch.__version__,"device":DEVICE,
                   "gpu":torch.cuda.get_device_name(0) if DEVICE=='cuda' else None},
        "data_summary":{"trials":len(trials),"endpoints":sum(len(t['y'])for t in trials.values()),
                        "valid_endpoints":sum(int(t['valid'].sum())for t in trials.values()),
                        "slip_prevalence_valid":float(sum(t['y'][t['valid']].sum()for t in trials.values())/sum(t['valid'].sum()for t in trials.values()))}}
    serialized=json.dumps(manifest,sort_keys=True,ensure_ascii=False).encode('utf-8')
    manifest['fingerprint']=hashlib.sha256(serialized).hexdigest()
    return manifest

def prepare(trials):
    manifest=build_manifest(trials)
    target=OUT/'pretraining_manifest.json'
    if target.exists():
        old=json.loads(target.read_text(encoding='utf-8'))
        if old['fingerprint'] != manifest['fingerprint'] and any(OUT.glob('**/result.json')):
            raise RuntimeError('Protocol/code/data changed after completed runs; use a new output directory, never silently mix results')
    save_json(target,manifest)
    audit=[]
    for config in configurations():
        mean,scale=fit_scaler(trials,config['train'])
        for split in ['train','validation','test']:
            c=chunks(trials,config[split],mean,scale)
            reconstructed=reconstruct(c['y'],c,trials,config[split])
            assert all(np.array_equal(reconstructed[i],trials[i]['y'])for i in config[split])
            nvalid=int(c['mask'].sum())
            assert nvalid==sum(int(trials[i]['valid'].sum())for i in config[split])
            audit.append({'configuration':config['id'],'split':split,'trials':len(config[split]),'chunks':len(c['x']),
                          'valid_scored_frames':nvalid,'slip_prevalence':float(c['y'][c['mask']].mean())})
    save_json(OUT/'chunk_validation.json',{'all_real_endpoints_scored_exactly_once':True,'no_overlap_across_trial_sets':True,'records':audit})
    log('prepared',fingerprint=manifest['fingerprint'],data=manifest['data_summary'],model_parameters=manifest['models_parameters'])
    return manifest

def fit_one(trials,config,name,seed,lr,phase,manifest,sets,mean,scale):
    tag=f'{phase}_seed{seed}_lr{lr:g}'
    run_dir=OUT/config['id']/name/tag
    result_path=run_dir/'result.json'
    if result_path.exists():
        old=json.loads(result_path.read_text(encoding='utf-8'))
        if old['fingerprint']!=manifest['fingerprint']:
            raise RuntimeError('Attempted resume with mismatched fingerprint')
        if not (run_dir/'checkpoint.pt').exists():
            raise RuntimeError('Completed run is missing its checkpoint')
        return old
    run_dir.mkdir(parents=True,exist_ok=True)
    log('start',configuration=config['id'],model=name,phase=phase,seed=seed,lr=lr)
    seed_all(seed)
    model=make_model(name,54)
    tr,va=sets['train'],sets['validation']
    model,training=train(model,tr['x'],tr['y'],tr['mask'],va['x'],va['y'],va['mask'],seed=seed,lr=lr,steps=STEPS,batch=BATCH)
    val_predictions=predict(model,va['x'])
    threshold=best_threshold(va['y'][va['mask']],val_predictions[va['mask']])
    result={'fingerprint':manifest['fingerprint'],'configuration':config['id'],'model':name,'phase':phase,
            'seed':seed,'lr':lr,'parameters':count_parameters(model),'training':training,
            'validation':metrics(va['y'][va['mask']],val_predictions[va['mask']],threshold)}
    torch.save({'state_dict':model.state_dict(),'mean':mean,'scale':scale,'model':name,'input_dim':54,
                'fingerprint':manifest['fingerprint'],'threshold':threshold},run_dir/'checkpoint.pt')
    if phase=='final':
        validation_probs=reconstruct(val_predictions,va,trials,config['validation'])
        result['validation_trials']=evaluate_trials(trials,config['validation'],validation_probs,threshold,run_dir/'validation_predictions')
        te=sets['test'];timing=[]
        for repeat in range(3):
            if DEVICE=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();test_predictions=predict(model,te['x'])
            if DEVICE=='cuda':torch.cuda.synchronize()
            timing.append(time.perf_counter()-start)
        test_probs=reconstruct(test_predictions,te,trials,config['test'])
        result['test']=evaluate_trials(trials,config['test'],test_probs,threshold,run_dir/'test_predictions')
        result['inference']={'three_complete_pass_seconds':timing,'median_pass_seconds':float(np.median(timing)),
                             'scored_real_endpoints':sum(len(trials[i]['y'])for i in config['test']),
                             'definition':'batched128-step chunks including transfer; unfused implementation-specific throughput, not streaming control latency'}
    # Write completion marker last. A crash before it reruns the same seed and fixed configuration.
    save_json(result_path,result)
    log('complete',configuration=config['id'],model=name,phase=phase,seed=seed,lr=lr,
        validation_ap=result['validation']['ap'],test_ap=result.get('test',{}).get('frame',{}).get('ap'),seconds=training['seconds'])
    del model
    if DEVICE=='cuda':torch.cuda.empty_cache()
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',choices=['held_trial','loo','all'],default='held_trial')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    trials=load_trials();manifest=prepare(trials)
    if args.prepare_only:return
    for config in configurations():
        if args.suite!='all' and config['suite']!=args.suite:continue
        mean,scale=fit_scaler(trials,config['train'])
        sets={k:chunks(trials,config[k],mean,scale)for k in ['train','validation','test']}
        save_json(OUT/config['id']/'normalization.json',{'mean':mean.tolist(),'scale':scale.tolist(),'fitted_training_trials':config['train']})
        for name in config['models']:
            tunes=[fit_one(trials,config,name,TUNE_SEED,lr,'tune',manifest,sets,mean,scale)for lr in LRS]
            winner=max(tunes,key=lambda r:r['validation']['ap'])
            lr=winner['lr']
            save_json(OUT/config['id']/name/'selection.json',{'fingerprint':manifest['fingerprint'],'selected_lr':lr,
                        'rule':'maximum validationAP tune_seed901, firstlistedlr exacttie','candidates':[{'lr':r['lr'],'validation_ap':r['validation']['ap']}for r in tunes]})
            for seed in SEEDS:
                fit_one(trials,config,name,seed,lr,'final',manifest,sets,mean,scale)
    log('suite_complete',suite=args.suite)

if __name__=='__main__':
    try:main()
    except Exception as error:
        log('failure',error=str(error),traceback=traceback.format_exc())
        raise

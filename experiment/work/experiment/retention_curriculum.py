"""Fresh-test study of one-bit retention training paths.

This module does not launch training unless --run is explicitly passed. It
creates one model and one AdamW optimizer per run, retaining their state across
all six stages. The imported common/models modules are never modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch.nn import functional as F

from common import DEVICE, best_threshold, metrics, predict, save_json, seed_all, sha256
from models import count_parameters, make_model, optimizer_parameter_groups


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "outputs" / "补充协议_记忆训练路径.md"
SEAL = ROOT / "outputs" / "补充协议_记忆训练路径_登记.json"
PRECHECK = Path(__file__).with_name("retention_curriculum_precheck.json")
LAGS = (2, 4, 8, 16, 32, 64)
ARMS = ("direct", "curriculum")
MODELS = ("gru", "mamba")
LEARNING_RATES = (0.001, 0.003)
FINAL_SEEDS = tuple(range(5))
STEPS_PER_STAGE = 400
STEPS = 2400
BATCH = 32
TRAIN_N, VAL_N, TEST_N = 4096, 4096, 16384
VAL_SEED, TEST_SEED = 620001, 630001
RATE, LENGTH = 0.20, 128
INFERENCE_BATCH = 128
SUCCESS_QUERY_AP = 0.90
SUCCESS_CUE_DROP = 0.30


def generate(n: int, seed: int, lag: int):
    """Same one-cue/one-bit construction as v2, generalized lags, fresh seeds."""
    if lag not in LAGS:
        raise ValueError(f"Unregistered lag {lag}")
    streams = np.random.SeedSequence(seed).spawn(5)
    symbol = np.random.default_rng(streams[0]).choice([-1., 1.], size=(n, LENGTH)).astype(np.float32)
    endpoint = np.random.default_rng(streams[1]).integers(96, 128, size=n)
    bit = np.random.default_rng(streams[2]).choice([-1., 1.], size=n).astype(np.float32)
    query = (np.random.default_rng(streams[3]).random(n) < 2 * RATE).astype(np.float32)
    noise = np.random.default_rng(streams[4]).normal(size=(n, LENGTH, 5)).astype(np.float32)
    rows = np.arange(n)
    cue = endpoint - lag
    symbol[rows, cue] = bit
    marker, gate = np.zeros((n, LENGTH), np.float32), np.zeros((n, LENGTH), np.float32)
    marker[rows, cue] = 1
    gate[rows, endpoint] = query
    x = np.concatenate([symbol[..., None], marker[..., None], gate[..., None], noise], axis=-1)
    scalar_y = (query * (bit > 0)).astype(np.float32)
    y, mask = np.zeros_like(marker), np.zeros_like(marker)
    y[rows, endpoint], mask[rows, endpoint] = scalar_y, 1
    metadata = {"endpoint": endpoint, "cue": cue, "bit": bit, "query": query, "y": scalar_y}
    return x, y, mask, metadata


def lag_at_step(arm: str, step: int) -> int:
    if arm not in ARMS or not 1 <= step <= STEPS:
        raise ValueError("Unknown arm or invalid update index")
    return 64 if arm == "direct" else LAGS[(step - 1) // STEPS_PER_STAGE]


def endpoint_predictions(model, x, metadata):
    p = predict(model, x, batch=INFERENCE_BATCH)
    return p[np.arange(len(x)), metadata["endpoint"]]


def cue_shuffle(x, metadata):
    ablation = x.copy()
    ablation[np.arange(len(x)), metadata["cue"], 0] = np.roll(metadata["bit"], 1)
    return ablation


def serialize_data_counts(meta):
    return {"records": len(meta["y"]), "positive_records": int(meta["y"].sum()),
            "query_records": int(meta["query"].sum()), "actual_positive_rate": float(meta["y"].mean()),
            "query_baseline_ap": float(average_precision_score(meta["y"], meta["query"])),
            "query_conditioned_positive_fraction": float(meta["y"][meta["query"] > 0].mean()),
            "oracle_ap": float(average_precision_score(meta["y"], meta["y"]))}


def precheck():
    """Only CPU generator checks and registration, never model fitting."""
    if SEAL.exists():
        raise RuntimeError("Registration exists; preserve it instead of silently changing a frozen protocol")
    report = {"study": "retention-training-path-v1", "training_launched": False,
              "gpu_operations_launched": False, "checks": []}
    reference = generate(96, 610000, 64)
    for lag in LAGS:
        x, y, mask, meta = generate(96, 610000, lag)
        prefix = generate(32, 610000, lag)
        for full, short in zip((x, y, mask), prefix[:3]):
            np.testing.assert_array_equal(full[:32], short)
        np.testing.assert_array_equal(y, reference[1])
        np.testing.assert_array_equal(mask, reference[2])
        np.testing.assert_array_equal(x[:, :, 2:], reference[0][:, :, 2:])
        for key in ("endpoint", "query", "bit", "y"):
            np.testing.assert_array_equal(meta[key], reference[3][key])
        assert np.all(x[:, :, 1].sum(axis=1) == 1)
        assert np.all(mask.sum(axis=1) == 1)
        np.testing.assert_array_equal(meta["endpoint"] - meta["cue"], np.full(96, lag))
        shuffled = cue_shuffle(x, meta)
        allowed = np.zeros_like(x, bool)
        allowed[np.arange(96), meta["cue"], 0] = True
        np.testing.assert_array_equal(x[~allowed], shuffled[~allowed])
        # For paired lags only their cue symbols and cue-marker positions may differ.
        allowed_pair = np.zeros_like(x, bool)
        for which in (meta, reference[3]):
            allowed_pair[np.arange(96), which["cue"], 0:2] = True
        np.testing.assert_array_equal(x[~allowed_pair], reference[0][~allowed_pair])
        report["checks"].append({"lag": lag, "same_labels_masks_endpoints_bits_queries_noise": True,
                                  "only_cue_position_variant": True, "nested_prefix": True,
                                  "one_marked_bit_and_one_loss_endpoint": True,
                                  "ablation_changes_only_marked_symbol": True})
    for arm in ARMS:
        counts = {str(lag): sum(lag_at_step(arm, step) == lag for step in range(1, STEPS+1)) for lag in LAGS}
        expected = {str(lag): (STEPS if lag == 64 else 0) for lag in LAGS} if arm == "direct" else {str(lag): 400 for lag in LAGS}
        assert counts == expected
        report.setdefault("schedule", {})[arm] = counts
    # Formula equivalence to v2 at shared generator seeds for supported lags;
    # actual study data use disjoint absolute seed namespaces below.
    from cue_memory import generate as original_generate
    for lag in (2, 64):
        current = generate(32, 610000, lag)
        old = original_generate(32, 610000, .20, lag)
        for a, b in zip(current[:3], old[:3]):
            np.testing.assert_array_equal(a, b)
    report["exact_formula_match_to_v2_at_same_seed"] = True
    counts = {}
    for name, n, seed in [("validation", VAL_N, VAL_SEED), ("test", TEST_N, TEST_SEED),
                           ("tuning_train", TRAIN_N, 610901)]:
        _, _, _, meta = generate(n, seed, 64)
        counts[name] = {"generator_seed": seed, **serialize_data_counts(meta)}
    # Index sampling is independent of arm and stage, hence actual exposure
    # counts must match for every prefix of the paired update sequences.
    for seed in FINAL_SEEDS:
        _, _, _, meta = generate(TRAIN_N, 610000 + seed, 64)
        a = np.random.default_rng(seed + 1729).integers(0, TRAIN_N, size=(STEPS, BATCH))
        b = np.random.default_rng(seed + 1729).integers(0, TRAIN_N, size=(STEPS, BATCH))
        np.testing.assert_array_equal(a, b)
        counts[f"train_seed{seed}"] = {"generator_seed": 610000+seed, **serialize_data_counts(meta),
                                       "planned_record_exposures": int(a.size),
                                       "planned_positive_exposures": int(meta["y"][a].sum()),
                                       "planned_query_exposures": int(meta["query"][a].sum())}
    report["data_counts"] = counts
    report["paired_record_and_positive_exposure_schedule"] = True
    report["status"] = "all CPU generator invariants passed; no model training"
    save_json(PRECHECK, report)
    seal = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "study": "retention-training-path-v1", "protocol_sha256": sha256(PROTOCOL),
            "script_sha256": sha256(__file__), "models_sha256": sha256(Path(__file__).with_name("models.py")),
            "common_sha256": sha256(Path(__file__).with_name("common.py")), "precheck_sha256": sha256(PRECHECK),
            "planned_tuning_runs": 8, "planned_final_runs": 20, "training_launched": False,
            "context": "New mechanism study proposed after prior v1/v2 results and long-lag training-floor evidence were available; not an amendment claimed to precede those findings.",
            "fresh_test_generator_seed": TEST_SEED,
            "lr_selection": "one LR per model shared by both arms, selected by mean best long-lag validation AP across the two tuning arms"}
    save_json(SEAL, seal)
    PROTOCOL.with_suffix(".sha256").write_text(sha256(PROTOCOL) + "  " + PROTOCOL.name + "\n", encoding="utf-8")
    print(json.dumps({"registration": seal, "precheck_status": report["status"], "data_counts": counts}, ensure_ascii=False, indent=2))


def verify_registration():
    if not SEAL.is_file():
        raise RuntimeError("Run CPU-only --precheck and freeze protocol before --run")
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    paths = {"protocol_sha256": PROTOCOL, "script_sha256": Path(__file__),
             "models_sha256": Path(__file__).with_name("models.py"),
             "common_sha256": Path(__file__).with_name("common.py")}
    for key, path in paths.items():
        if sha256(path) != seal[key]:
            raise RuntimeError(f"Registered source changed: {path}. Preserve and explicitly amend before running.")
    return seal


def train_one(name, arm, seed, lr, run_dir, validation):
    """One continuous model/optimizer lifetime across 2400 updates."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data_seed = 610000 + seed
    vx, _, _, vmeta = validation
    x, y, mask, tmeta = generate(TRAIN_N, data_seed, lag_at_step(arm, 1))
    seed_all(seed)
    model = make_model(name, 8).to(DEVICE)
    opt = torch.optim.AdamW(optimizer_parameter_groups(model, 1e-4), lr=lr)
    tx, ty, tm = (torch.as_tensor(array, device=DEVICE) for array in (x, y, mask))
    positives = int(tmeta["y"].sum())
    pos_weight = torch.tensor((TRAIN_N - positives) / max(positives, 1), device=DEVICE)
    rng = np.random.default_rng(seed + 1729)
    best_ap, best_step, best_state = -1., None, None
    trace, stages = [], []
    positive_exposures = query_exposures = 0
    stage_positive = stage_query = 0
    seen_records = np.zeros(TRAIN_N, bool)
    optimizer_object_id = id(opt)
    model_object_id = id(model)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    current_lag = lag_at_step(arm, 1)
    for step in range(1, STEPS + 1):
        next_lag = lag_at_step(arm, step)
        if next_lag != current_lag:
            x, same_y, same_mask, same_meta = generate(TRAIN_N, data_seed, next_lag)
            np.testing.assert_array_equal(same_y, y)
            np.testing.assert_array_equal(same_mask, mask)
            for key in ("bit", "query", "endpoint", "y"):
                np.testing.assert_array_equal(same_meta[key], tmeta[key])
            tx = torch.as_tensor(x, device=DEVICE)
            current_lag = next_lag
        assert id(opt) == optimizer_object_id and id(model) == model_object_id
        draw = rng.integers(0, TRAIN_N, size=BATCH)
        seen_records[draw] = True
        np_positive, np_query = int(tmeta["y"][draw].sum()), int(tmeta["query"][draw].sum())
        positive_exposures += np_positive
        query_exposures += np_query
        stage_positive += np_positive
        stage_query += np_query
        idx = torch.as_tensor(draw, device=DEVICE)
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(tx[idx])
        raw = F.binary_cross_entropy_with_logits(logits, ty[idx], pos_weight=pos_weight, reduction="none")
        loss = (raw * tm[idx]).sum() / tm[idx].sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Nonfinite gradient at step {step}")
        opt.step()
        if step % 100 == 0:
            vp = endpoint_predictions(model, vx, vmeta)
            val_ap = float(average_precision_score(vmeta["y"], vp))
            qm = vmeta["query"] > 0
            val_query_ap = float(average_precision_score(vmeta["y"][qm], vp[qm]))
            row = {"step": step, "stage": (step-1)//STEPS_PER_STAGE + 1, "training_lag": current_lag,
                   "validation_lag": 64, "train_loss": float(loss.detach()), "grad_norm_before_clip": float(grad_norm),
                   "val_ap": val_ap, "val_query_ap": val_query_ap,
                   "record_exposures": step*BATCH, "positive_exposures": positive_exposures,
                   "query_exposures": query_exposures, "unique_training_records_seen": int(seen_records.sum())}
            trace.append(row)
            if val_ap > best_ap:
                best_ap, best_step = val_ap, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            save_json(run_dir / "training_trace.json", {"complete": False, "trace": trace, "best_step": best_step,
                                                        "best_val_ap": best_ap, "optimizer_instances_created": 1})
        if step % STEPS_PER_STAGE == 0:
            stages.append({"stage": step//STEPS_PER_STAGE, "start_step": step-STEPS_PER_STAGE+1,
                           "end_step": step, "training_lag": current_lag,
                           "record_exposures": STEPS_PER_STAGE*BATCH,
                           "positive_exposures": stage_positive, "query_exposures": stage_query,
                           "val_ap": trace[-1]["val_ap"], "val_query_ap": trace[-1]["val_query_ap"]})
            stage_positive = stage_query = 0
            print(json.dumps({"stage_completed": name, "arm": arm, "seed": seed, "lr": lr,
                              "step": step, "training_lag": current_lag, "val_ap": trace[-1]["val_ap"],
                              "val_query_ap": trace[-1]["val_query_ap"]}), flush=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    optimizer_steps = sorted({int(state["step"].item()) for state in opt.state.values() if "step" in state})
    if optimizer_steps != [STEPS]:
        raise RuntimeError(f"Optimizer step continuity failed: {optimizer_steps}")
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(final_state, run_dir / "final.pt")
    torch.save(opt.state_dict(), run_dir / "final_optimizer.pt")
    torch.save(best_state, run_dir / "best.pt")
    model.load_state_dict(best_state)
    meta = {"trace": trace, "stages": stages, "best_val_ap": best_ap, "best_step": best_step,
            "best_stage": (best_step-1)//STEPS_PER_STAGE+1, "best_training_lag": lag_at_step(arm, best_step),
            "final_val_ap": trace[-1]["val_ap"], "final_val_query_ap": trace[-1]["val_query_ap"],
            "optimizer_instances_created": 1, "model_instances_created": 1,
            "final_optimizer_step_counters": optimizer_steps, "optimizer_reset_between_stages": False,
            "model_config": model.config, "parameters": count_parameters(model), "seconds": seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if DEVICE == "cuda" else None,
            "pos_weight": float(pos_weight), "steps": STEPS, "batch": BATCH, "lr": lr, "device": DEVICE,
            "record_exposures": STEPS*BATCH, "positive_exposures": positive_exposures,
            "query_exposures": query_exposures, "unique_training_records_seen": int(seen_records.sum()),
            "training_data_seed": data_seed, "training_data_counts": serialize_data_counts(tmeta),
            "loss": "uniform record sampling; endpoint masked BCE with fixed training-set n_negative/n_positive weighting"}
    save_json(run_dir / "training_trace.json", {"complete": True, **meta})
    return model, meta


def collect_metrics(model, x, metadata, threshold):
    p = endpoint_predictions(model, x, metadata)
    shuffled_p = endpoint_predictions(model, cue_shuffle(x, metadata), metadata)
    qm = metadata["query"] > 0
    m = metrics(metadata["y"], p, threshold)
    m.update({"query_ap": float(average_precision_score(metadata["y"][qm], p[qm])),
              "query_n": int(qm.sum()), "cue_shuffled_ap": float(average_precision_score(metadata["y"], shuffled_p)),
              "cue_shuffled_query_ap": float(average_precision_score(metadata["y"][qm], shuffled_p[qm]))})
    m["cue_shuffle_query_ap_drop"] = m["query_ap"]-m["cue_shuffled_query_ap"]
    m["auxiliary_long_task_success"] = bool(m["query_ap"] > SUCCESS_QUERY_AP and
                                             m["cue_shuffle_query_ap_drop"] > SUCCESS_CUE_DROP)
    return m, p, shuffled_p


def completed(path):
    if not (path / "result.json").is_file():
        return False
    for required in ("best.pt", "final.pt", "final_optimizer.pt", "training_trace.json"):
        if not (path / required).is_file():
            raise RuntimeError(f"Completion marker exists but checkpoint/log missing: {path/required}")
    return True


def run(out):
    seal = verify_registration()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Keep the same default cuDNN precision setting as common.train; explicit
    # matmul TF32 is disabled there and here. All models/data use float32.
    manifest = {"study": "retention-training-path-v1", "registration": seal,
                "rate": RATE, "length": LENGTH, "lags": list(LAGS), "test_lag": 64,
                "arms": list(ARMS), "models": list(MODELS), "final_seeds": list(FINAL_SEEDS),
                "n_train": TRAIN_N, "n_validation": VAL_N, "n_test": TEST_N,
                "train_generator_seed_rule": "610000+initialization_seed", "validation_seed": VAL_SEED,
                "test_seed": TEST_SEED, "tuning_seed": 901, "learning_rates": list(LEARNING_RATES),
                "steps": STEPS, "batch": BATCH, "stages": 6, "updates_per_stage": STEPS_PER_STAGE,
                "evaluation_every": 100, "validation_lag": 64,
                "inference_batch": INFERENCE_BATCH,
                "auxiliary_success": {"query_ap_greater_than": SUCCESS_QUERY_AP,
                                      "cue_shuffle_query_ap_drop_greater_than": SUCCESS_CUE_DROP,
                                      "stable_success_requires": "all five final seeds satisfy both thresholds"},
                "lr_selection": "per-model shared LR across arms: maximize mean best long-validation AP of direct and curriculum tuning runs",
                "checkpoint_selection": "best all-record endpoint AP on long-lag validation only; earliest wins exact ties",
                "planned_tuning_runs": 8, "planned_final_runs": 20,
                "torch": torch.__version__, "numpy": np.__version__, "device": DEVICE,
                "interpretation": "Training-path mechanism on fresh one-bit records; curriculum sees short-lag variants, not equal fixed-task exposure."}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Existing manifest differs; preserve prior study instead of mixing outputs")
    else:
        save_json(manifest_path, manifest)
    validation = generate(VAL_N, VAL_SEED, 64)
    for name in MODELS:
        model_dir = out / name
        selection_path = model_dir / "selection.json"
        if not selection_path.exists():
            tuning = {}
            for lr in LEARNING_RATES:
                for arm in ARMS:
                    folder = model_dir / arm / f"tune_seed901_lr{lr:g}"
                    if completed(folder):
                        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                    else:
                        print(json.dumps({"start": name, "arm": arm, "phase": "tuning", "seed": 901, "lr": lr}), flush=True)
                        model, training = train_one(name, arm, 901, lr, folder, validation)
                        result = {"study": manifest["study"], "phase": "tuning", "model": name, "arm": arm,
                                  "seed": 901, "lr": lr, "config": model.config, "training": training,
                                  "test_evaluated": False}
                        save_json(folder / "result.json", result)
                        del model
                    tuning[(lr, arm)] = result["training"]["best_val_ap"]
            selection_rows = [{"lr": lr, "direct_best_val_ap": tuning[(lr, "direct")],
                               "curriculum_best_val_ap": tuning[(lr, "curriculum")],
                               "mean_best_val_ap_across_arms": float(np.mean([tuning[(lr, arm)] for arm in ARMS]))}
                              for lr in LEARNING_RATES]
            selected = max(selection_rows, key=lambda row: row["mean_best_val_ap_across_arms"])
            selection = {"model": name, "selected_lr": selected["lr"], "shared_by_both_arms": True,
                         "criterion": manifest["lr_selection"], "trials": selection_rows,
                         "manifest_sha256": sha256(manifest_path), "test_used": False}
            save_json(selection_path, selection)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection["manifest_sha256"] != sha256(manifest_path):
            raise RuntimeError("Stored learning-rate selection belongs to a different manifest")
        lr = selection["selected_lr"]
        for seed in FINAL_SEEDS:
            for arm in ARMS:
                folder = model_dir / arm / f"final_seed{seed}_lr{lr:g}"
                if completed(folder):
                    if not (folder / "test_predictions.npz").is_file():
                        raise RuntimeError(f"Missing completed final predictions: {folder}")
                    continue
                print(json.dumps({"start": name, "arm": arm, "phase": "final", "seed": seed, "lr": lr}), flush=True)
                model, training = train_one(name, arm, seed, lr, folder, validation)
                vx, _, _, vmeta = validation
                vp = endpoint_predictions(model, vx, vmeta)
                threshold = best_threshold(vmeta["y"], vp)
                ex, _, _, emeta = generate(TEST_N, TEST_SEED, 64)
                result_metrics, ep, abp = collect_metrics(model, ex, emeta, threshold)
                np.savez_compressed(folder / "test_predictions.npz", y=emeta["y"], p=ep,
                                    query=emeta["query"], cue_shuffled_p=abp,
                                    endpoint=emeta["endpoint"], cue=emeta["cue"], bit=emeta["bit"])
                np.savez_compressed(folder / "validation_predictions.npz", y=vmeta["y"], p=vp, query=vmeta["query"])
                result = {"study": manifest["study"], "phase": "final", "model": name, "arm": arm,
                          "seed": seed, "lr": lr, "rate": RATE, "n_train": TRAIN_N, "test_lag": 64,
                          "parameters": count_parameters(model), "config": model.config,
                          "metrics": result_metrics, "training": training,
                          "validation_threshold": threshold, "test_data_counts": serialize_data_counts(emeta),
                          "manifest_sha256": sha256(manifest_path)}
                save_json(folder / "result.json", result)
                print(json.dumps({"completed": name, "arm": arm, "seed": seed, "ap": result_metrics["ap"],
                                  "query_ap": result_metrics["query_ap"],
                                  "cue_shuffled_query_ap": result_metrics["cue_shuffled_query_ap"]}), flush=True)
                del model, ex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--precheck", action="store_true", help="CPU data invariants and protocol registration only")
    mode.add_argument("--run", action="store_true", help="Explicit launch of the registered 28 training runs")
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "results" / "retention_curriculum")
    args = parser.parse_args()
    precheck() if args.precheck else run(args.out)

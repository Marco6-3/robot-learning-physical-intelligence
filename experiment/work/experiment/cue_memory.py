"""Preregistered one-bit cue memory task; no training unless --run is passed.

Version 2 is a distinct algorithmic task, preserving v1 files and negative
results. Exactly one marked bit matters in each independent record; lag changes
retention duration without changing the number of relevant bits.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from common import seed_all, save_json, sha256, train, predict, metrics, best_threshold
from models import make_model, count_parameters


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "outputs" / "补充协议_单状态记忆.md"
SEAL = ROOT / "outputs" / "补充协议_单状态记忆_登记.json"
SEEDS = list(range(5))
RATES = (0.01, 0.20)
LAGS = (2, 64)
TRAIN_SIZES = (4096, 16384)
MODELS = ("gru", "mamba")
LEARNING_RATES = (0.001, 0.003)
LENGTH = 128
STEPS = 400
BATCH = 32
VAL_N = 2048
TEST_N = 16384


def generate(n: int, seed: int, prevalence: float, lag: int):
    """Independent RNG streams make prefixes nested over n and paired over lag.

    Return dense [record,time] y/mask for common.train. Exactly one time per
    record is scored, even for query=0 records. Metadata retains scalar labels.
    """
    if lag not in LAGS or prevalence not in RATES:
        raise ValueError("This preregistered task allows only lag2/64 and prevalence .01/.20")
    streams = np.random.SeedSequence(seed).spawn(5)
    symbol = np.random.default_rng(streams[0]).choice([-1., 1.], size=(n, LENGTH)).astype(np.float32)
    endpoint = np.random.default_rng(streams[1]).integers(96, 128, size=n)
    bit = np.random.default_rng(streams[2]).choice([-1., 1.], size=n).astype(np.float32)
    query = (np.random.default_rng(streams[3]).random(n) < 2 * prevalence).astype(np.float32)
    noise = np.random.default_rng(streams[4]).normal(size=(n, LENGTH, 5)).astype(np.float32)
    cue = endpoint - lag
    rows = np.arange(n)
    symbol[rows, cue] = bit
    marker = np.zeros((n, LENGTH), dtype=np.float32)
    marker[rows, cue] = 1
    gate = np.zeros_like(marker)
    gate[rows, endpoint] = query
    x = np.concatenate([symbol[..., None], marker[..., None], gate[..., None], noise], axis=-1)
    scalar_y = (query * (bit > 0)).astype(np.float32)
    y = np.zeros_like(marker)
    mask = np.zeros_like(marker)
    y[rows, endpoint] = scalar_y
    mask[rows, endpoint] = 1
    metadata = {"endpoint": endpoint, "cue": cue, "bit": bit, "query": query, "y": scalar_y}
    return x, y, mask, metadata


def scalar_predictions(model, x, metadata):
    dense = predict(model, x, batch=32)
    return dense[np.arange(len(x)), metadata["endpoint"]]


def shuffle_cue_bit(x, metadata):
    shuffled = x.copy()
    shuffled[np.arange(len(x)), metadata["cue"], 0] = np.roll(metadata["bit"], 1)
    return shuffled


def precheck():
    """CPU-only generator invariants and analytic-oracle checks; no model fit."""
    report = {"task": "single-cue-one-bit-memory-v2", "checks": [], "training_launched": False}
    for rate in RATES:
        pair = []
        for lag in LAGS:
            x, y, mask, meta = generate(256, 200001, rate, lag)
            smaller = generate(64, 200001, rate, lag)
            for large, small in zip((x, y, mask), smaller[:3]):
                np.testing.assert_array_equal(large[:64], small)
            assert np.all(mask.sum(axis=1) == 1)
            assert np.all(x[:, :, 1].sum(axis=1) == 1)
            np.testing.assert_array_equal(y[np.arange(256), meta["endpoint"]], meta["y"])
            np.testing.assert_array_equal(x[np.arange(256), meta["cue"], 0], meta["bit"])
            assert np.all(meta["endpoint"] - meta["cue"] == lag)
            np.testing.assert_array_equal(x[np.arange(256), meta["endpoint"], 2], meta["query"])
            shuffled = shuffle_cue_bit(x, meta)
            altered = np.zeros_like(x, dtype=bool)
            altered[np.arange(256), meta["cue"], 0] = True
            np.testing.assert_array_equal(x[~altered], shuffled[~altered])
            # Construct two records differing only in the marked bit. Current
            # and immediately preceding frames are identical even as q=1 label flips.
            qrows = np.flatnonzero(meta["query"] > 0)
            flipped = x.copy()
            flipped[qrows, meta["cue"][qrows], 0] *= -1
            for offset in (0, 1):
                np.testing.assert_array_equal(x[qrows, meta["endpoint"][qrows] - offset],
                                              flipped[qrows, meta["endpoint"][qrows] - offset])
            assert np.all((meta["bit"][qrows] > 0) != (-meta["bit"][qrows] > 0))
            pair.append((x, y, mask, meta))
            report["checks"].append({"rate": rate, "lag": lag, "nested_prefix": True,
                                      "one_cue_one_endpoint": True, "recent_two_frames_counterfactual": True,
                                      "cue_shuffle_changes_only_marked_symbol": True})
        for key in ("endpoint", "bit", "query", "y"):
            np.testing.assert_array_equal(pair[0][3][key], pair[1][3][key])
        np.testing.assert_array_equal(pair[0][0][:, :, 2:], pair[1][0][:, :, 2:])
        _, _, _, test = generate(TEST_N, 300001, rate, 64)
        report.setdefault("test_oracles", []).append({
            "rate": rate, "n": TEST_N, "positive_records": int(test["y"].sum()),
            "query_records": int(test["query"].sum()),
            "oracle_ap": float(average_precision_score(test["y"], test["y"])),
            "query_only_baseline_ap": float(average_precision_score(test["y"], test["query"])),
            "query_conditioned_positive_fraction": float(test["y"][test["query"] > 0].mean()),
        })
    report["status"] = "all generator checks passed; no model training performed by precheck"
    save_json(Path(__file__).with_name("cue_memory_precheck.json"), report)
    seal = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "single-cue-one-bit-memory-v2",
        "protocol_sha256": sha256(PROTOCOL), "generator_and_run_script_sha256": sha256(__file__),
        "models_sha256": sha256(Path(__file__).with_name("models.py")),
        "common_sha256_at_registration": sha256(Path(__file__).with_name("common.py")),
        "precheck_sha256": sha256(Path(__file__).with_name("cue_memory_precheck.json")),
        "reason": "Identifiability audit: v1 long lag also requires dense multi-bit history; v2 holds relevant memory capacity at one bit.",
        "results_visible_when_amendment_was_requested": "Design request specified v2 before root inspected target-model rankings; root had seen v1 MLP outputs. Before the registration file was completed, root inspected short-lag GRU seed4 and Mamba seeds0/1. The fixed v2 design and 112 planned runs were not changed based on those values.",
        "training_launched": False,
    }
    if SEAL.exists():
        raise RuntimeError("Registration already exists: preserve it; do not silently re-register after seeing results")
    save_json(SEAL, seal)
    PROTOCOL.with_suffix(".sha256").write_text(sha256(PROTOCOL) + "  " + PROTOCOL.name + "\n", encoding="utf-8")
    print(json.dumps({"precheck": report, "registration": seal}, ensure_ascii=False, indent=2))


def verify_registration():
    if not SEAL.exists():
        raise RuntimeError("Run --precheck before any training to seal the declared protocol")
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    expected = {"protocol_sha256": PROTOCOL, "generator_and_run_script_sha256": Path(__file__),
                "models_sha256": Path(__file__).with_name("models.py")}
    for key, path in expected.items():
        if sha256(path) != seal[key]:
            raise RuntimeError(f"Registered file changed: {path}. Record an explicit amendment before proceeding.")
    return seal


def run(out: Path):
    seal = verify_registration()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"generator": "single-cue-one-bit-memory-v2", "length": LENGTH,
                "score": "one randomly selected endpoint per record in inclusive[96,127]",
                "prevalence": list(RATES), "lags": list(LAGS), "relevant_bits_per_record": 1,
                "train_records": list(TRAIN_SIZES), "validation_records": VAL_N, "test_records": TEST_N,
                "train_seeds": SEEDS, "models": list(MODELS), "steps": STEPS, "batch": BATCH,
                "eval_every": 100, "learning_rates": list(LEARNING_RATES), "tune_seed": 901,
                "planned_tuning_runs": 32, "planned_main_runs": 80, "test_used_for_selection": False,
                "predictions_shape": "scalar vectors [record]; keep all records, including query=0",
                "preregistration": seal,
                "common_sha256_at_run": sha256(Path(__file__).with_name("common.py")),
                "interpretation": "Algorithmic one-bit retention; distinct from v1 dense-symbol delay and from physical tactile control."}
    if (out / "manifest.json").exists():
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError("Existing result manifest differs; preserve the prior experiment")
    else:
        save_json(out / "manifest.json", manifest)
    for rate in RATES:
        for lag in LAGS:
            vx, vy, vm, vmeta = generate(VAL_N, 200001, rate, lag)
            ex, _, _, emeta = generate(TEST_N, 300001, rate, lag)
            for n in TRAIN_SIZES:
                for name in MODELS:
                    key = f"p{rate:g}_lag{lag}_n{n}_{name}"
                    hp_path = out / (key + "_tuning.json")
                    if hp_path.exists():
                        hp = json.loads(hp_path.read_text(encoding="utf-8"))
                    else:
                        xx, yy, mask, _ = generate(n, 100901, rate, lag)
                        trials = []
                        for lr in LEARNING_RATES:
                            seed_all(901)
                            model = make_model(name, 8)
                            model, meta = train(model, xx, yy, mask, vx, vy, vm,
                                                901, lr, steps=STEPS, batch=BATCH, eval_every=100)
                            trials.append({"lr": lr, "model_config": model.config, **meta})
                            del model
                        hp = {"selected_lr": max(trials, key=lambda trial: trial["best_val_ap"])["lr"], "trials": trials}
                        save_json(hp_path, hp)
                        print(json.dumps({"tuned": key, "lr": hp["selected_lr"]}), flush=True)
                    for seed in SEEDS:
                        path = out / (key + f"_seed{seed}.json")
                        if path.exists():
                            continue
                        xx, yy, mask, _ = generate(n, 100000 + seed, rate, lag)
                        seed_all(seed)
                        model = make_model(name, 8)
                        model, meta = train(model, xx, yy, mask, vx, vy, vm, seed,
                                            hp["selected_lr"], steps=STEPS, batch=BATCH, eval_every=100)
                        vp = scalar_predictions(model, vx, vmeta)
                        threshold = best_threshold(vmeta["y"], vp)
                        ep = scalar_predictions(model, ex, emeta)
                        abp = scalar_predictions(model, shuffle_cue_bit(ex, emeta), emeta)
                        result_metrics = metrics(emeta["y"], ep, threshold)
                        qm = emeta["query"] > 0
                        result_metrics.update({
                            "query_ap": float(average_precision_score(emeta["y"][qm], ep[qm])),
                            "query_n": int(qm.sum()),
                            "cue_shuffled_ap": float(average_precision_score(emeta["y"], abp)),
                            "cue_shuffled_query_ap": float(average_precision_score(emeta["y"][qm], abp[qm])),
                        })
                        result = {"task": "single-cue-one-bit-memory-v2", "rate": rate, "lag": lag,
                                  "n_train": n, "model": name, "seed": seed, "parameters": count_parameters(model),
                                  "config": model.config, "metrics": result_metrics, "training": meta}
                        np.savez_compressed(out / (key + f"_seed{seed}_predictions.npz"),
                                            y=emeta["y"], p=ep, query=emeta["query"], cue_shuffled_p=abp,
                                            endpoint=emeta["endpoint"], cue=emeta["cue"], bit=emeta["bit"])
                        torch.save(model.state_dict(), out / (key + f"_seed{seed}.pt"))
                        save_json(path, result)
                        print(json.dumps({"completed": key, "seed": seed, "ap": result_metrics["ap"],
                                          "query_ap": result_metrics["query_ap"]}), flush=True)
                        del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--precheck", action="store_true", help="CPU generator checks and immutable preregistration; no training")
    mode.add_argument("--run", action="store_true", help="Explicitly launch all registered GPU training runs")
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "results" / "cue_memory_v2")
    args = parser.parse_args()
    if args.precheck:
        precheck()
    else:
        run(args.out)

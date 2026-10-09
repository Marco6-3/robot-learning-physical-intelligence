"""Exact paired analysis of the separately registered one-bit memory supplement.

V2 was sealed after some v1 GRU/Mamba results were actually observed. Its design
was proposed beforehand and was not changed by those results. It is not the
original blind v1 preregistration. No v2 model results existed before its seal.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score

from analyze_synthetic import RankedClusterAP, hypothesis_contrasts, save_json, summarize_interval, write_csv


def scalar_metadata(n, seed, rate):
    """Reconstruct only metadata using the sealed v2 generator's separate streams."""
    streams = np.random.SeedSequence(seed).spawn(5)
    endpoint = np.random.default_rng(streams[1]).integers(96, 128, size=n)
    bit = np.random.default_rng(streams[2]).choice([-1., 1.], size=n).astype(np.float32)
    query = (np.random.default_rng(streams[3]).random(n) < 2 * rate).astype(np.float32)
    return {"endpoint": endpoint, "bit": bit, "query": query, "y": (query * (bit > 0)).astype(np.float32)}


def run(args):
    source, output = Path(args.source), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["generator"] != "single-cue-one-bit-memory-v2":
        raise ValueError("This script analyzes only the sealed scalar-endpoint v2 task")
    rates, lags, sizes = manifest["prevalence"], manifest["lags"], manifest["train_records"]
    seeds, models = manifest["train_seeds"], manifest["models"]
    if rates != [.01, .20] or lags != [2, 64] or sizes != [4096, 16384] or seeds != list(range(5)) or set(models) != {"gru", "mamba"}:
        raise ValueError("Factorial design differs from the fixed v2 protocol")
    conditions = list(itertools.product(rates, lags, sizes))
    n_records = manifest["test_records"]
    if n_records != 16384:
        raise ValueError("Unexpected v2 test size")
    expected, missing = [], []
    for key in itertools.product(rates, lags, sizes, models, seeds):
        rate, lag, size, model, seed = key
        stem = f"p{rate:g}_lag{lag}_n{size}_{model}_seed{seed}"
        jp, pp = source / (stem + ".json"), source / (stem + "_predictions.npz")
        expected.append((key, jp, pp))
        missing.extend(str(path) for path in [jp, pp] if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Incomplete fixed v2 study: {len(missing)} missing artifacts; first: {missing[:8]}")

    ones = np.ones(n_records)
    caches, rows, metadata_references = {}, [], {}
    max_ap_error = 0.
    for key, json_path, prediction_path in expected:
        rate, lag, size, model, seed = key
        result = json.loads(json_path.read_text(encoding="utf-8"))
        if (result["rate"], result["lag"], result["n_train"], result["model"], result["seed"]) != key:
            raise ValueError(f"Metadata mismatch: {json_path}")
        with np.load(prediction_path, allow_pickle=False) as stored:
            arrays = {k: stored[k] for k in ["y", "p", "query", "cue_shuffled_p", "endpoint", "cue", "bit"]}
        if any(value.shape != (n_records,) for value in arrays.values()):
            raise ValueError(f"Unexpected scalar record shape: {prediction_path}")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"Nonfinite artifact: {prediction_path}")
        if rate not in metadata_references:
            metadata_references[rate] = scalar_metadata(n_records, 300001, rate)
        for column, reference in metadata_references[rate].items():
            np.testing.assert_array_equal(arrays[column], reference, err_msg=f"Mismatched common test records: {prediction_path}")
        np.testing.assert_array_equal(arrays["endpoint"] - arrays["cue"], np.full(n_records, lag))
        y, p, query = (arrays[column][:, None] for column in ["y", "p", "query"])
        full, conditioned = RankedClusterAP(y, p), RankedClusterAP(y, p, query > 0)
        ap, query_ap = full(ones), conditioned(ones)
        errors = [abs(ap - result["metrics"]["ap"]), abs(query_ap - result["metrics"]["query_ap"])]
        max_ap_error = max(max_ap_error, *errors)
        if max(errors) > 1e-9:
            raise ValueError("Saved AP disagrees with exact record-weighted calculation")
        q = arrays["query"] > 0
        ab_ap = average_precision_score(arrays["y"], arrays["cue_shuffled_p"])
        ab_q_ap = average_precision_score(arrays["y"][q], arrays["cue_shuffled_p"][q])
        np.testing.assert_allclose([ab_ap, ab_q_ap], [result["metrics"]["cue_shuffled_ap"], result["metrics"]["cue_shuffled_query_ap"]], atol=1e-9)
        caches[key] = (full, conditioned)
        rows.append({"rate": rate, "lag": lag, "n_train": size, "model": model, "seed": seed,
                     "parameters": result["parameters"], **result["metrics"],
                     "cue_shuffle_ap_drop": ap - float(ab_ap), "cue_shuffle_query_ap_drop": query_ap - float(ab_q_ap),
                     "seconds": result["training"]["seconds"], "steps": result["training"]["steps"],
                     "batch": result["training"]["batch"]})

    summaries = []
    for condition in conditions:
        for model in models:
            selected = [row for row in rows if (row["rate"], row["lag"], row["n_train"], row["model"]) == (*condition, model)]
            item = {"rate": condition[0], "lag": condition[1], "n_train": condition[2], "model": model,
                    "seeds": len(selected), "parameters": selected[0]["parameters"]}
            for metric in ["ap", "query_ap", "f1", "precision", "recall", "cue_shuffled_ap", "cue_shuffled_query_ap",
                           "cue_shuffle_ap_drop", "cue_shuffle_query_ap_drop", "seconds"]:
                values = np.array([row[metric] for row in selected])
                item[metric + "_mean"], item[metric + "_seed_sd"] = float(values.mean()), float(values.std(ddof=1))
            summaries.append(item)

    baselines = []
    for rate in rates:
        for split, n, data_seed in [("validation", manifest["validation_records"], 200001), ("test", n_records, 300001)]:
            metadata = scalar_metadata(n, data_seed, rate)
            y, query = metadata["y"], metadata["query"]
            baselines.append({"split": split, "rate": rate, "records": n, "positive_records": int(y.sum()),
                              "query_records": int(query.sum()), "prevalence": float(y.mean()),
                              "score_query_ap": float(average_precision_score(y, query)),
                              "query_conditioned_constant_ap": float(y[query > 0].mean()),
                              "oracle_ap": float(average_precision_score(y, y))})

    point_d = {condition: np.mean([np.array([caches[(*condition, "gru", seed)][mi](ones)
                                           - caches[(*condition, "mamba", seed)][mi](ones) for mi in range(2)])
                                   for seed in seeds], axis=0) for condition in conditions}
    point_h = hypothesis_contrasts(point_d, rates, lags, sizes)
    rng = np.random.default_rng(args.bootstrap_seed)
    boot_d = np.empty((args.draws, len(conditions), 2), dtype=np.float64)
    start = time.perf_counter()
    for draw in range(args.draws):
        record_weights = rng.multinomial(n_records, np.full(n_records, 1 / n_records))
        seed_weights = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)))
        for ci, condition in enumerate(conditions):
            aggregate = np.zeros(2)
            for seed, weight in zip(seeds, seed_weights):
                if weight:
                    gru, mamba = caches[(*condition, "gru", seed)], caches[(*condition, "mamba", seed)]
                    aggregate += weight * np.array([gru[mi](record_weights) - mamba[mi](record_weights) for mi in range(2)])
            boot_d[draw, ci] = aggregate / len(seeds)
        if (draw + 1) % 100 == 0:
            print(json.dumps({"cue_bootstrap_draws": draw + 1, "total": args.draws,
                              "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    boot_h = hypothesis_contrasts({condition: boot_d[:, ci] for ci, condition in enumerate(conditions)}, rates, lags, sizes)
    hypotheses = []
    descriptions = ["D(sparse,short,small)", "mean_n[D(sparse,long,n)-D(sparse,short,n)]",
                    "(sparse-dense memory difference at small N) - (same at large N)"]
    for hi, name in enumerate(["H1", "H2", "H3"]):
        for mi, metric in enumerate(["ap", "query_ap"]):
            interval = summarize_interval(boot_h[hi, :, mi], 1 - .05 / 3 if mi == 0 else .95)
            supports = None
            if mi == 0 and interval["undefined_draws"] == 0:
                supports = (interval["lower"] > args.margin if hi == 0 else
                            interval["upper"] < -args.margin if hi == 1 else
                            interval["lower"] > 0 or interval["upper"] < 0)
            hypotheses.append({"hypothesis": name, "metric": metric, "contrast": descriptions[hi],
                               "estimate": float(point_h[hi, mi]), **interval,
                               "primary_within_v2": mi == 0, "criterion_supported": supports,
                               "margin": args.margin if hi < 2 else 0,
                               "direction": "positive" if hi == 0 else "negative" if hi == 1 else "two-sided"})
    differences = []
    for ci, condition in enumerate(conditions):
        for mi, metric in enumerate(["ap", "query_ap"]):
            differences.append({"rate": condition[0], "lag": condition[1], "n_train": condition[2], "metric": metric,
                                "gru_minus_mamba": float(point_d[condition][mi]),
                                **summarize_interval(boot_d[:, ci, mi], .95), "confirmatory": False})
    report = {
        "task": manifest["generator"], "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "preregistration_as_recorded": manifest["preregistration"],
        "registration_timing_clarification": "V2's concrete task design was requested before target-architecture rankings were inspected. During implementation, before the V2 protocol file was frozen, the root agent actually viewed V1 GRU seed4 and Mamba seeds0,1. V2's specified design was unchanged; no V2 learned-model results existed before its freeze. Treat V2 as a separately registered mechanism supplement, not as part of the original blind V1 preregistration.",
        "input_runs": len(rows), "cells": summaries, "hypotheses": hypotheses, "differences": differences,
        "data_counts_and_deterministic_baselines": baselines,
        "baselines": [{"rate": row["rate"], "lag": lag,
                       "score_query_all_frame_ap": row["score_query_ap"],
                       "query_only_constant_score_ap": row["query_conditioned_constant_ap"],
                       "oracle_ap": row["oracle_ap"]}
                      for row in baselines if row["split"] == "test" for lag in lags],
        "bootstrap": {"draws": args.draws, "random_seed": args.bootstrap_seed,
                      "method": "paired training-seed and independent-record bootstrap; same weights across all conditions/models",
                      "ap": "exact tied-score record-weighted non-interpolated AP; no approximation",
                      "primary_confidence": 1 - .05 / 3,
                      "multiplicity_scope": "three primary AP contrasts inside V2 only; does not cover V1 plus V2 or other analyses",
                      "seconds": time.perf_counter() - start},
        "validation": {"max_saved_vs_recomputed_ap_abs_error": max_ap_error,
                       "exact_test_metadata_reconstruction": True,
                       "labels_queries_endpoints_bits_paired_across_all_conditions": True},
        "limitations": [
            "The sparse validation set has only about 20 positives among 2048 records; exact counts are reported. Validation-based LR, checkpoint and threshold choices can therefore be noisy.",
            "Intervals condition on the already selected hyperparameters/checkpoints; the small validation set is not resampled and the complete selection process is not rerun.",
            "Only five training seeds; percentile-bootstrap tail coverage is approximate, especially at 98.333% confidence.",
            "The supplement was frozen after actual partial V1 target-architecture results were observed; the exact timing is disclosed above.",
            "H1 compares fixed data budgets, not a complete sample-efficiency curve. With 400 updates of batch32, there are only12800 training presentations and the larger16384-record pool cannot be exhaustively visited.",
            "The 1% rate is positive independent endpoint records, not continuous-time event occupancy. There is no valid Event-F1, onset latency or false alarms/min from these isolated endpoints.",
            "H3 is an interaction on the AP scale, which depends on prevalence; query-conditioned AP is a diagnostic, not an additional corrected confirmation family.",
            "One-bit retention isolates required information capacity better than V1's delay queue, but is still an algorithmic mechanism and not physical contact or closed-loop control.",
        ],
    }
    save_json(output / "analysis.json", report)
    write_csv(output / "per_seed_metrics.csv", rows)
    write_csv(output / "cell_summary.csv", summaries)
    write_csv(output / "primary_contrasts.csv", hypotheses)
    write_csv(output / "differences.csv", differences)
    write_csv(output / "data_counts_and_baselines.csv", baselines)
    np.savez_compressed(output / "bootstrap_draws.npz", differences=boot_d, hypotheses=boot_h,
                        conditions=np.array(conditions), metric_names=np.array(["ap", "query_ap"]))
    print(json.dumps({"analysis": str(output / "analysis.json"), "hypotheses": hypotheses}), flush=True)


def self_test():
    rng = np.random.default_rng(56)
    y, p, q = rng.integers(0, 2, 37), np.round(rng.random(37), 1), rng.random(37) > .4
    y = y * q
    maximum = 0.
    for selected in [np.ones(37, bool), q]:
        rank = RankedClusterAP(y[:, None], p[:, None], selected[:, None])
        for _ in range(30):
            indices = rng.integers(0, len(y), len(y))
            if not y[indices][selected[indices]].sum():
                continue
            weights = np.bincount(indices, minlength=len(y))
            exact = average_precision_score(y[indices][selected[indices]], p[indices][selected[indices]])
            maximum = max(maximum, abs(rank(weights) - exact))
            np.testing.assert_allclose(rank(weights), exact, atol=1e-14)
    expected = scalar_metadata(16384, 300001, .01)
    assert int(expected["y"].sum()) == 157 and int(expected["query"].sum()) == 345
    print(json.dumps({"self_test": "passed", "scalar_weighted_ap_max_abs_error": maximum,
                      "known_frozen_sparse_test_metadata": "157 positives /345 queries confirmed"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="work/results/cue_memory_v2")
    parser.add_argument("--out", default="work/results/cue_memory_v2_analysis")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261011)
    parser.add_argument("--margin", type=float, default=.03)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    self_test() if arguments.self_test else run(arguments)

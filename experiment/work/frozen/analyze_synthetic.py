"""Exact paired seed-and-record bootstrap for the frozen synthetic experiment.

Only saved predictions are read. No models are trained, selected, or changed.
Cached score ranks make record-weighted AP exactly equivalent to physically
replicating sampled records, including tied scores; frames are never subsampled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RankedClusterAP:
    """Binary, non-interpolated AP with nonnegative integer record weights."""

    def __init__(self, y, p, selection=None):
        y, p = np.asarray(y), np.asarray(p)
        if y.ndim != 2 or p.shape != y.shape:
            raise ValueError("Expected equal [record,time] label and score arrays")
        if not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Nonfinite scores or nonbinary labels")
        record = np.broadcast_to(np.arange(y.shape[0])[:, None], y.shape)
        selection = np.ones(y.shape, bool) if selection is None else np.asarray(selection, bool)
        yy, pp, rr = y[selection], p[selection], record[selection]
        if not len(yy):
            raise ValueError("Empty AP selection")
        order = np.argsort(-pp, kind="stable")
        self.y = yy[order].astype(np.float64)
        self.record = rr[order].astype(np.int32)
        sorted_p = pp[order]
        self.ends = np.r_[np.flatnonzero(sorted_p[1:] != sorted_p[:-1]), len(sorted_p) - 1]
        self.n_records = len(y)
        self.n = len(yy)

    def __call__(self, record_weights):
        weights = np.asarray(record_weights, dtype=np.float64)[self.record]
        positives = np.cumsum(weights * self.y)[self.ends]
        total = positives[-1]
        if total == 0:
            # Undefined draws are tracked and prevent confirmatory conclusions.
            return float("nan")
        all_counts = np.cumsum(weights)[self.ends]
        positive_increments = np.diff(np.r_[0., positives])
        precision = np.divide(positives, all_counts, out=np.zeros_like(positives), where=all_counts > 0)
        return float(np.dot(precision, positive_increments) / total)


def spans(y):
    diff = np.diff(np.r_[0, np.asarray(y, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def sequence_events(y, p, threshold, hz):
    """One-to-one overlap matching per record; adjacent positive frames merge."""
    total_tp = total_fp = total_fn = 0
    signed_latencies = []
    for labels, scores in zip(y, p):
        truth, alarms = spans(labels), spans(scores >= threshold)
        used = set()
        for start, stop in truth:
            candidates = [(max(start, a), i, a) for i, (a, b) in enumerate(alarms)
                          if i not in used and max(start, a) < min(stop, b)]
            if candidates:
                _, i, alarm_start = min(candidates)
                used.add(i)
                signed_latencies.append((alarm_start - start) / hz)
        total_tp += len(used)
        total_fp += len(alarms) - len(used)
        total_fn += len(truth) - len(used)
    duration = y.size / hz
    latency = np.asarray(signed_latencies)
    return {
        "event_tp": total_tp, "event_fp": total_fp, "event_fn": total_fn,
        "event_f1": 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1),
        "event_recall": total_tp / max(total_tp + total_fn, 1),
        "false_alarms_per_min": total_fp / (duration / 60),
        "detected_latency_sec_median": float(np.median(np.maximum(latency, 0))) if len(latency) else None,
        "detected_signed_onset_difference_sec_median": float(np.median(latency)) if len(latency) else None,
        "event_duration_sec": duration,
    }


def hypothesis_contrasts(d, rates, lags, sizes):
    sparse, dense = min(rates), max(rates)
    short, long = min(lags), max(lags)
    small, large = min(sizes), max(sizes)
    h1 = d[(sparse, short, small)]
    h2 = np.mean([d[(sparse, long, n)] - d[(sparse, short, n)] for n in sizes], axis=0)

    def difference_of_differences(n):
        return (d[(sparse, long, n)] - d[(sparse, short, n)]
                - d[(dense, long, n)] + d[(dense, short, n)])

    h3 = difference_of_differences(small) - difference_of_differences(large)
    return np.asarray([h1, h2, h3])


def summarize_interval(values, confidence):
    values = np.asarray(values)
    valid = np.isfinite(values)
    if not valid.any():
        return {"lower": None, "upper": None, "undefined_draws": int(len(values)), "confidence": confidence}
    alpha = 1 - confidence
    lo, hi = np.quantile(values[valid], [alpha / 2, 1 - alpha / 2])
    return {"lower": float(lo), "upper": float(hi), "undefined_draws": int((~valid).sum()), "confidence": confidence}


def load_inputs(source):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rates, lags, sizes = manifest["prevalence"], manifest["lags"], manifest["train_records"]
    seeds, models = manifest["train_seeds"], manifest["models"]
    if [len(rates), len(lags), len(sizes), len(seeds), len(models)] != [2, 2, 2, 5, 4]:
        raise ValueError("Confirmatory analysis requires the fixed 32 cells x 5 seeds")
    if not {"gru", "mamba", "mlp", "tcn"} == set(models):
        raise ValueError("Unexpected model set")
    expected, missing = [], []
    for rate, lag, size, model, seed in itertools.product(rates, lags, sizes, models, seeds):
        stem = f"p{rate:g}_lag{lag}_n{size}_{model}_seed{seed}"
        jp, pp = source / (stem + ".json"), source / (stem + "_predictions.npz")
        expected.append(((rate, lag, size, model, seed), jp, pp))
        missing.extend(str(path) for path in (jp, pp) if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Incomplete fixed study: {len(missing)} missing artifacts; first: {missing[:8]}")
    return manifest, expected, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def run(args):
    source, destination = Path(args.source), Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    manifest, expected, manifest_hash = load_inputs(source)
    rates, lags, sizes = manifest["prevalence"], manifest["lags"], manifest["train_records"]
    seeds, models = manifest["train_seeds"], manifest["models"]
    conditions = list(itertools.product(rates, lags, sizes))
    n_records = manifest["test_records"]
    n_frames = manifest["length"] - manifest["score_start"]
    ones = np.ones(n_records)
    rows, baselines, cached, labels_by_condition = [], {}, {}, {}
    validation_max_abs = 0.
    for key, json_path, prediction_path in expected:
        rate, lag, size, model, seed = key
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if (metadata["rate"], metadata["lag"], metadata["n_train"], metadata["model"], metadata["seed"]) != key:
            raise ValueError(f"Metadata mismatch in {json_path}")
        with np.load(prediction_path, allow_pickle=False) as arrays:
            y, p, query = arrays["y"], arrays["p"], arrays["query"]
            if y.shape != (n_records, n_frames) or p.shape != y.shape or query.shape != y.shape:
                raise ValueError(f"Unexpected array shape in {prediction_path}")
            cell = (rate, lag, size)
            old = labels_by_condition.get(cell)
            if old is not None and (not np.array_equal(y, old[0]) or not np.array_equal(query, old[1])):
                raise ValueError("Test records are not identical across models/seeds in a cell")
            labels_by_condition.setdefault(cell, (y.copy(), query.copy()))
            if np.any((y == 1) & (query == 0)):
                raise ValueError("Positive labels outside visible queries")
            full_rank, query_rank = RankedClusterAP(y, p), RankedClusterAP(y, p, query > 0)
            ap, query_ap = full_rank(ones), query_rank(ones)
            errors = [abs(ap - metadata["metrics"]["ap"]), abs(query_ap - metadata["metrics"]["query_ap"])]
            validation_max_abs = max(validation_max_abs, *errors)
            if max(errors) > 1e-9:
                raise ValueError(f"Saved AP disagrees with exact rank calculation in {json_path}")
            if model in {"gru", "mamba"}:
                cached[key] = (full_rank, query_rank)
            metrics = metadata["metrics"]
            events = sequence_events(y, p, metrics["threshold"], manifest["nominal_hz"])
            rows.append({"rate": rate, "lag": lag, "n_train": size, "model": model, "seed": seed,
                         "parameters": metadata["parameters"], **metrics, **events,
                         "seconds": metadata["training"]["seconds"]})
            baseline_key = (rate, lag)
            if baseline_key not in baselines:
                baselines[baseline_key] = {
                    "rate": rate, "lag": lag, "positive_frames": int(y.sum()),
                    "query_frames": int(query.sum()), "scored_frames": int(y.size),
                    "actual_prevalence": float(y.mean()),
                    "score_query_all_frame_ap": float(average_precision_score(y.ravel(), query.ravel())),
                    "query_only_constant_score_ap": float(y[query > 0].mean()),
                    "oracle_ap": float(average_precision_score(y.ravel(), y.ravel())),
                }

    # Paired cross-condition resampling relies on the same ordered test records.
    for rate, lag in itertools.product(rates, lags):
        references = [labels_by_condition[(rate, lag, size)] for size in sizes]
        if any(not np.array_equal(ref[0], references[0][0]) or not np.array_equal(ref[1], references[0][1]) for ref in references[1:]):
            raise ValueError("Test arrays differ across training sizes")
    for rate in rates:
        q0 = labels_by_condition[(rate, lags[0], sizes[0])][1]
        if any(not np.array_equal(q0, labels_by_condition[(rate, lag, sizes[0])][1]) for lag in lags):
            raise ValueError("Query draws differ across lag conditions")
    qs = labels_by_condition[(min(rates), lags[0], sizes[0])][1]
    qd = labels_by_condition[(max(rates), lags[0], sizes[0])][1]
    if np.any(qs > qd):
        raise ValueError("Sparse queries are not nested in dense queries")

    summaries = []
    for condition in conditions:
        for model in models:
            selected = [r for r in rows if (r["rate"], r["lag"], r["n_train"], r["model"]) == (*condition, model)]
            item = {"rate": condition[0], "lag": condition[1], "n_train": condition[2], "model": model,
                    "seeds": len(selected), "parameters": selected[0]["parameters"]}
            for metric in ["ap", "query_ap", "f1", "event_f1", "event_recall", "false_alarms_per_min",
                           "key_shuffled_ap", "key_shuffled_query_ap", "seconds"]:
                values = np.array([row[metric] for row in selected], dtype=float)
                item[metric + "_mean"] = float(values.mean())
                item[metric + "_seed_sd"] = float(values.std(ddof=1))
            summaries.append(item)

    point = {}
    for condition in conditions:
        point[condition] = np.mean([np.array([cached[(*condition, "gru", seed)][m](ones)
                                             - cached[(*condition, "mamba", seed)][m](ones) for m in range(2)])
                                    for seed in seeds], axis=0)
    point_h = hypothesis_contrasts(point, rates, lags, sizes)
    rng = np.random.default_rng(args.bootstrap_seed)
    boot_d = np.empty((args.draws, len(conditions), 2), np.float64)
    start = time.perf_counter()
    for draw in range(args.draws):
        record_weights = rng.multinomial(n_records, np.full(n_records, 1 / n_records))
        seed_weights = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)))
        for ci, condition in enumerate(conditions):
            aggregate = np.zeros(2)
            for seed, weight in zip(seeds, seed_weights):
                if not weight:
                    continue
                gru, mamba = cached[(*condition, "gru", seed)], cached[(*condition, "mamba", seed)]
                aggregate += weight * np.array([gru[m](record_weights) - mamba[m](record_weights) for m in range(2)])
            boot_d[draw, ci] = aggregate / len(seeds)
        if (draw + 1) % 100 == 0:
            print(json.dumps({"bootstrap_draws": draw + 1, "total": args.draws, "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    boot_h = hypothesis_contrasts({condition: boot_d[:, i] for i, condition in enumerate(conditions)}, rates, lags, sizes)
    # Shape is hypothesis x draw x metric.
    simultaneous = 1 - 0.05 / 3
    hypothesis_rows = []
    descriptions = ["D(sparse,short,small)", "mean_n[D(sparse,long,n)-D(sparse,short,n)]",
                    "(sparse-dense memory difference at small N) - (same at large N)"]
    for hi, name in enumerate(["H1", "H2", "H3"]):
        for mi, metric in enumerate(["ap", "query_ap"]):
            interval = summarize_interval(boot_h[hi, :, mi], simultaneous if mi == 0 else 0.95)
            supported = None
            if mi == 0 and interval["undefined_draws"] == 0:
                supported = (interval["lower"] > args.margin if hi == 0 else
                             interval["upper"] < -args.margin if hi == 1 else
                             interval["lower"] > 0 or interval["upper"] < 0)
            hypothesis_rows.append({"hypothesis": name, "metric": metric, "contrast": descriptions[hi],
                                    "estimate": float(point_h[hi, mi]), **interval,
                                    "confirmatory": mi == 0, "criterion_supported": supported,
                                    "margin": args.margin if hi < 2 else 0,
                                    "direction": "positive" if hi == 0 else "negative" if hi == 1 else "two-sided"})
    difference_rows = []
    for ci, condition in enumerate(conditions):
        for mi, metric in enumerate(["ap", "query_ap"]):
            difference_rows.append({"rate": condition[0], "lag": condition[1], "n_train": condition[2],
                                    "metric": metric, "gru_minus_mamba": float(point[condition][mi]),
                                    **summarize_interval(boot_d[:, ci, mi], .95), "confirmatory": False})
    result = {
        "manifest_sha256": manifest_hash, "input_runs": len(rows), "factorial_cells": len(summaries),
        "bootstrap": {"draws": args.draws, "random_seed": args.bootstrap_seed,
                      "method": "crossed paired bootstrap of 5 training replicates and 512 shared test records",
                      "record_weights_shared_across_models_conditions_and_training_replicates": True,
                      "seed_weights_shared_across_models_and_conditions": True,
                      "ap": "exact non-interpolated AP, tie-aware cluster weighting, no frame subsampling",
                      "primary_interval_confidence": simultaneous,
                      "multiplicity": "Bonferroni nominal simultaneous 95% coverage for three primary AP contrasts",
                      "seconds": time.perf_counter() - start},
        "validation": {"max_saved_vs_recomputed_ap_abs_error": validation_max_abs,
                       "identical_test_arrays_across_models_seeds_sizes": True},
        "limitations": [
            "Five training replicates give a weak estimate of training-process tails; percentile bootstrap coverage is approximate.",
            "Intervals condition on selected hyperparameters, fixed validation data, architecture sizes and training budget; hyperparameter selection is not rerun.",
            "The same fixed test dataset is shared across seeds; seeds are not independent test datasets.",
            "Query AP and cell differences have descriptive 95% intervals and are not additional confirmatory tests.",
            "H1 compares two fixed data budgets rather than estimating an entire sample-efficiency curve.",
            "Delayed retrieval is an algorithmic test with memory-capacity demands, not physical contact simulation or future forecasting.",
            "Event metrics merge adjacent positive frames and use one-to-one temporal overlap, with no matching across record boundaries.",
            "Nonnegative latency is conditional on detected events; signed onset difference is also recorded and is not evidence of forecasting.",
        ],
        "baselines": list(baselines.values()), "hypotheses": hypothesis_rows,
        "differences": difference_rows, "cells": summaries,
    }
    save_json(destination / "analysis.json", result)
    write_csv(destination / "per_seed_metrics.csv", rows)
    write_csv(destination / "cell_summary.csv", summaries)
    write_csv(destination / "primary_contrasts.csv", hypothesis_rows)
    write_csv(destination / "differences.csv", difference_rows)
    write_csv(destination / "deterministic_baselines.csv", list(baselines.values()))
    np.savez_compressed(destination / "bootstrap_draws.npz", differences=boot_d, hypotheses=boot_h,
                        conditions=np.array(conditions), metric_names=np.array(["ap", "query_ap"]))
    print(json.dumps({"analysis": str(destination / "analysis.json"), "hypotheses": hypothesis_rows}), flush=True)


def self_test():
    rng = np.random.default_rng(117)
    max_error, cases = 0., 0
    for ties in [False, True]:
        for query_only in [False, True]:
            y = rng.integers(0, 2, size=(17, 13))
            p = rng.random(y.shape)
            if ties:
                p = np.round(p, 1)
            mask = rng.random(y.shape) < .6 if query_only else np.ones(y.shape, bool)
            cache = RankedClusterAP(y, p, mask)
            for _ in range(25):
                sampled = rng.integers(0, len(y), size=len(y))
                weights = np.bincount(sampled, minlength=len(y))
                actual = cache(weights)
                expected = average_precision_score(y[sampled][mask[sampled]], p[sampled][mask[sampled]])
                max_error = max(max_error, abs(actual - expected))
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
                cases += 1
    # All-negative draw must remain explicitly undefined, not silently zero.
    cache = RankedClusterAP(np.array([[0, 0], [1, 0]]), np.array([[.2, .1], [.8, .1]]))
    assert np.isnan(cache([2, 0]))
    print(json.dumps({"self_test": "passed", "exact_vs_explicit_record_replication_cases": cases,
                      "max_abs_error": max_error, "all_negative_draw": "explicit_nan"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="work/results/synthetic")
    parser.add_argument("--out", default="work/results/synthetic_analysis")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261009)
    parser.add_argument("--margin", type=float, default=.03)
    parser.add_argument("--self-test", action="store_true")
    parsed = parser.parse_args()
    self_test() if parsed.self_test else run(parsed)

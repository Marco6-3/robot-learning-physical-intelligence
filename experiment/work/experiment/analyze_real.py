"""Trial-paired descriptive analysis of saved ARQ2021 current-slip predictions.

No training or checkpoint selection. Objects are fixed strata, not a sampled
object population. The three LOO folds are shown individually and macro-averaged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score

from analyze_synthetic import RankedClusterAP, save_json, spans, summarize_interval, write_csv


def trial_events(arrays, threshold):
    y, p = arrays["y"], arrays["p"]
    valid, t, coverage = arrays["valid"], arrays["timestamp_s"], arrays["coverage_s"]
    tp = fp = fn = left_censored = 0
    all_delays, observed_delays, signed_delays, ious = [], [], [], []
    for first, last in spans(valid):
        truth, alarms = spans(y[first:last]), spans(p[first:last] >= threshold)
        used = set()
        for a, b in truth:
            left_censored += int(a == 0)
            candidates = [(max(a, c), i, c, d) for i, (c, d) in enumerate(alarms)
                          if i not in used and max(a, c) < min(b, d)]
            if candidates:
                _, index, c, d = min(candidates)
                used.add(index)
                signed = float(t[first + c] - t[first + a])
                all_delays.append(max(0., signed))
                if a > 0:
                    signed_delays.append(signed)
                    observed_delays.append(max(0., signed))
                intersection = coverage[first + max(a, c):first + min(b, d)].sum()
                union = coverage[first + min(a, c):first + max(b, d)].sum()
                ious.append(float(intersection / union))
        tp += len(used)
        fp += len(alarms) - len(used)
        fn += len(truth) - len(used)
    negative = valid & (y == 0)
    return {
        "event_tp": tp, "event_fp": fp, "event_fn": fn,
        "left_censored_truth_segments": left_censored,
        "valid_duration_sec": float(coverage[valid].sum()),
        "negative_duration_sec": float(coverage[negative].sum()),
        "negative_alarm_duration_sec": float(coverage[negative & (p >= threshold)].sum()),
        "alarm_duration_sec": float(coverage[valid & (p >= threshold)].sum()),
        "all_matched_nonnegative_delays": all_delays,
        "observed_onset_nonnegative_delays": observed_delays,
        "observed_onset_signed_differences": signed_delays,
        "matched_temporal_ious": ious,
    }


def aggregate_events(events):
    sums = {key: sum(row[key] for row in events) for key in [
        "event_tp", "event_fp", "event_fn", "left_censored_truth_segments",
        "valid_duration_sec", "negative_duration_sec", "negative_alarm_duration_sec", "alarm_duration_sec"]}
    tp, fp, fn = sums["event_tp"], sums["event_fp"], sums["event_fn"]
    duration, negative = sums["valid_duration_sec"], sums["negative_duration_sec"]
    result = {**sums, "event_f1": 2 * tp / max(2 * tp + fp + fn, 1),
              "event_recall": tp / max(tp + fn, 1), "event_miss_fraction": fn / max(tp + fn, 1),
              "false_alarms_per_min": fp / (duration / 60) if duration else None,
              "negative_alarm_time_fraction": sums["negative_alarm_duration_sec"] / negative if negative else None,
              "alarm_time_fraction": sums["alarm_duration_sec"] / duration if duration else None}
    arrays = {key: np.asarray([value for row in events for value in row[key]]) for key in [
        "all_matched_nonnegative_delays", "observed_onset_nonnegative_delays",
        "observed_onset_signed_differences", "matched_temporal_ious"]}
    for key, values in arrays.items():
        result[key + "_n"] = len(values)
        result[key + "_median"] = float(np.median(values)) if len(values) else None
        result[key + "_p90"] = float(np.quantile(values, .9)) if len(values) else None
    return result


def padded_ap(trial_arrays):
    n, length = len(trial_arrays), max(len(arrays["y"]) for arrays in trial_arrays)
    y, p, valid = np.zeros((n, length)), np.zeros((n, length)), np.zeros((n, length), bool)
    for index, arrays in enumerate(trial_arrays):
        m = len(arrays["y"])
        y[index, :m], p[index, :m], valid[index, :m] = arrays["y"], arrays["p"], arrays["valid"]
    return RankedClusterAP(y, p, valid)


def read_trial(path):
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in ["y", "p", "valid", "timestamp_s", "coverage_s", "source_endpoint_index"]}
    n = len(arrays["y"])
    if any(value.shape != (n,) for value in arrays.values()) or not n:
        raise ValueError(f"Invalid trial shape: {path}")
    if not all(np.isfinite(arrays[key]).all() for key in ["y", "p", "timestamp_s", "coverage_s"]):
        raise ValueError(f"Nonfinite data: {path}")
    if not np.isin(arrays["y"], [0, 1]).all() or arrays["valid"].dtype != bool:
        raise ValueError(f"Invalid labels or mask: {path}")
    if not (np.diff(arrays["timestamp_s"]) > 0).all() or not (arrays["coverage_s"] > 0).all():
        raise ValueError(f"Nonpositive time coverage: {path}")
    return arrays


def run(args):
    source, output = Path(args.source), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "pretraining_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configurations = manifest["configurations"]
    if args.suite != "all":
        configurations = [c for c in configurations if c["suite"] == args.suite]
    seeds = manifest["training"]["final_seeds"]
    if len(seeds) != 5:
        raise ValueError("Expected five frozen training seeds")
    audit_path = Path(args.data_audit)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    objects = {row["trial_id"]: row["object"] for row in audit["trials"]}
    caches, reference_trials, seed_rows, trial_rows, baselines = {}, {}, [], [], []
    max_ap_error = 0.
    # Validate completeness before computing any comparisons.
    expected = []
    missing = []
    for config in configurations:
        for model in config["models"]:
            selection_path = source / config["id"] / model / "selection.json"
            if not selection_path.is_file():
                missing.append(str(selection_path))
                continue
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if selection["fingerprint"] != manifest["fingerprint"]:
                raise ValueError("Selection fingerprint differs from frozen manifest")
            for seed in seeds:
                run_dir = selection_path.parent / f"final_seed{seed}_lr{selection['selected_lr']:g}"
                result_path = run_dir / "result.json"
                if not result_path.is_file():
                    missing.append(str(result_path))
                for trial_id in config["test"]:
                    path = run_dir / "test_predictions" / (trial_id + ".npz")
                    if not path.is_file():
                        missing.append(str(path))
                expected.append((config, model, seed, run_dir, result_path))
    if missing:
        raise FileNotFoundError(f"Incomplete fixed real-data suite: {len(missing)} missing artifacts; first: {missing[:8]}")

    for config, model, seed, run_dir, result_path in expected:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["fingerprint"] != manifest["fingerprint"] or result["phase"] != "final":
            raise ValueError(f"Wrong fingerprint or phase: {result_path}")
        if (result["configuration"], result["model"], result["seed"]) != (config["id"], model, seed):
            raise ValueError(f"Wrong model/seed metadata: {result_path}")
        threshold = result["validation"]["threshold"]
        if threshold != result["test"]["frame"]["threshold"]:
            raise ValueError("Test threshold differs from validation choice")
        trials, events = [], []
        for trial_id in config["test"]:
            arrays = read_trial(run_dir / "test_predictions" / (trial_id + ".npz"))
            if trial_id in reference_trials:
                if any(not np.array_equal(arrays[key], reference_trials[trial_id][key])
                       for key in arrays if key != "p"):
                    raise ValueError(f"Ground truth or timing changed for {trial_id}")
            else:
                reference_trials[trial_id] = {key: value.copy() for key, value in arrays.items() if key != "p"}
            event = trial_events(arrays, threshold)
            events.append(event)
            v = arrays["valid"]
            frame_ap = float(average_precision_score(arrays["y"][v], arrays["p"][v])) if arrays["y"][v].sum() else None
            trial_rows.append({"configuration": config["id"], "model": model, "seed": seed,
                               "trial_id": trial_id, "object": objects[trial_id], "ap": frame_ap,
                               "valid_frames": int(v.sum()), "positive_frames": int(arrays["y"][v].sum()),
                               **aggregate_events([event])})
            trials.append(arrays)
        rank = padded_ap(trials)
        ap = rank(np.ones(len(trials)))
        error = abs(ap - result["test"]["frame"]["ap"])
        max_ap_error = max(max_ap_error, error)
        if error > 1e-9:
            raise ValueError("Recomputed AP does not match saved test AP")
        summary = aggregate_events(events)
        for key in ["event_tp", "event_fp", "event_fn"]:
            if summary[key] != result["test"]["event"][key]:
                raise ValueError(f"Event matching differs from frozen evaluator: {key}")
        np.testing.assert_allclose(summary["valid_duration_sec"], result["test"]["event"]["duration_sec"], atol=1e-8)
        seed_rows.append({"configuration": config["id"], "suite": config["suite"], "model": model,
                          "seed": seed, "n_train_trials": len(config["train"]), "n_test_trials": len(config["test"]),
                          "parameters": result["parameters"], **result["test"]["frame"], **summary,
                          "training_seconds": result["training"]["seconds"]})
        if model in {"gru", "mamba"}:
            caches[(config["id"], model, seed)] = rank

    for config in configurations:
        baseline_trials = [{**reference_trials[i], "p": np.ones_like(reference_trials[i]["y"])} for i in config["test"]]
        baselines.append({"configuration": config["id"], "baseline": "always_alarm",
                          "ap": padded_ap(baseline_trials)(np.ones(len(baseline_trials))),
                          **aggregate_events([trial_events(row, .5) for row in baseline_trials])})

    summaries = []
    for config in configurations:
        for model in config["models"]:
            rows = [row for row in seed_rows if row["configuration"] == config["id"] and row["model"] == model]
            summary = {"configuration": config["id"], "suite": config["suite"], "model": model,
                       "seeds": len(rows), "n_train_trials": len(config["train"]), "n_test_trials": len(config["test"]),
                       "parameters": rows[0]["parameters"]}
            for metric in ["ap", "f1", "precision", "recall", "event_f1", "event_recall", "event_miss_fraction",
                           "false_alarms_per_min", "negative_alarm_time_fraction", "alarm_time_fraction", "training_seconds",
                           "all_matched_nonnegative_delays_median", "observed_onset_nonnegative_delays_median",
                           "observed_onset_signed_differences_median", "matched_temporal_ious_median"]:
                values = np.array([row[metric] for row in rows if row[metric] is not None])
                summary[metric + "_mean"] = float(values.mean()) if len(values) else None
                summary[metric + "_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
                if len(values) < len(rows):
                    summary[metric + "_defined_seeds"] = len(values)
            summaries.append(summary)

    # Same held-trial cohort uses identical resampled trial weights at low/full N.
    # Different LOO objects are separate fixed strata, never object-resampled.
    cohorts = {tuple(config["test"]): None for config in configurations}
    strata = {ids: [np.array([i for i, trial_id in enumerate(ids) if objects[trial_id] == obj], dtype=int)
                    for obj in sorted({objects[trial_id] for trial_id in ids})] for ids in cohorts}
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap = np.zeros((args.draws, len(configurations), 2), dtype=np.float64)
    point = np.zeros((len(configurations), 2), dtype=np.float64)
    for ci, config in enumerate(configurations):
        for mi, model in enumerate(["gru", "mamba"]):
            point[ci, mi] = np.mean([caches[(config["id"], model, seed)](np.ones(len(config["test"]))) for seed in seeds])
    start = time.perf_counter()
    for draw in range(args.draws):
        seed_weights = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)))
        record_weights = {}
        for ids, groups in strata.items():
            weights = np.zeros(len(ids), dtype=int)
            for group in groups:
                weights[group] = rng.multinomial(len(group), np.full(len(group), 1 / len(group)))
            record_weights[ids] = weights
        for ci, config in enumerate(configurations):
            weights = record_weights[tuple(config["test"])]
            for mi, model in enumerate(["gru", "mamba"]):
                bootstrap[draw, ci, mi] = sum(weight * caches[(config["id"], model, seed)](weights)
                                             for seed, weight in zip(seeds, seed_weights) if weight) / len(seeds)
        if (draw + 1) % 100 == 0:
            print(json.dumps({"real_bootstrap_draws": draw + 1, "total": args.draws,
                              "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    contrast_rows = []
    for ci, config in enumerate(configurations):
        contrast_rows.append({"configuration": config["id"], "scope": "fixed object(s), trial-level conditional uncertainty",
                              "gru_ap": float(point[ci, 0]), "mamba_ap": float(point[ci, 1]),
                              "gru_minus_mamba_ap": float(point[ci, 0] - point[ci, 1]),
                              **summarize_interval(bootstrap[:, ci, 0] - bootstrap[:, ci, 1], .95), "confirmatory": False})
    loo_indices = [i for i, config in enumerate(configurations) if config["suite"] == "loo"]
    if len(loo_indices) == 3:
        loo_point = point[loo_indices].mean(axis=0)
        loo_bootstrap = bootstrap[:, loo_indices].mean(axis=1)
        contrast_rows.append({"configuration": "loo_macro_three_fixed_objects",
                              "scope": "equal-weight average of these three fixed objects; not new-object population inference",
                              "gru_ap": float(loo_point[0]), "mamba_ap": float(loo_point[1]),
                              "gru_minus_mamba_ap": float(loo_point[0] - loo_point[1]),
                              **summarize_interval(loo_bootstrap[:, 0] - loo_bootstrap[:, 1], .95), "confirmatory": False})

    report = {"manifest_fingerprint": manifest["fingerprint"],
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "suite": args.suite, "completed_runs": len(seed_rows), "cells": summaries,
              "comparisons": contrast_rows, "always_alarm_baselines": baselines,
              "bootstrap": {"draws": args.draws, "seed": args.bootstrap_seed,
                            "method": "paired training-seed and object-stratified trial resampling; exact cluster-weighted AP",
                            "objects_resampled": False, "confidence": .95,
                            "same_test_cohort_resampled_identically_across_budgets": True,
                            "seconds": time.perf_counter() - start},
              "validation": {"max_saved_vs_recomputed_ap_abs_error": max_ap_error,
                             "event_counts_match_frozen_evaluator": True,
                             "prediction_padding_excluded": True,
                             "actual_timestamp_bin_coverage_used": True},
              "limitations": [
                  "Observational current-slip detection with natural prevalence; it does not test the 1% sparse H1 or identify causal long-memory demand.",
                  "Only three objects; each object's trials are resampled conditionally within that object. No claim about an independently sampled object population.",
                  "Five training seeds and percentile bootstrap give approximate uncertainty conditional on fixed data splits, hyperparameters and trained models.",
                  "95% intervals are descriptive, not a multiplicity-adjusted confirmatory hypothesis family.",
                  "Overlap Event-F1 can reward long active alarms; always-alarm baseline, negative-label alarm duration and temporal IoU expose this pathology.",
                  "False alarms/min counts unmatched alarm episodes, not false-positive frames; denominator is actual valid bin coverage from timestamps.",
                  "Latency is conditional on detection. Known-onset latency excludes slip already active at the beginning of a valid span; missed events and censored segments remain reported.",
                  "An alarm that started before a slip has clipped delay zero; signed differences and alarm occupancy are also reported, and neither establishes future forecasting.",
                  "Labels are manual independent annotations with unknown precise timing error; sub-bin timing claims are unsupported.",
              ]}
    save_json(output / "analysis.json", report)
    write_csv(output / "per_seed_metrics.csv", seed_rows)
    write_csv(output / "per_trial_metrics.csv", trial_rows)
    write_csv(output / "cell_summary.csv", summaries)
    write_csv(output / "gru_mamba_comparisons.csv", contrast_rows)
    write_csv(output / "always_alarm_baselines.csv", baselines)
    np.savez_compressed(output / "bootstrap_draws.npz", architecture_ap=bootstrap,
                        configurations=np.array([c["id"] for c in configurations]), models=np.array(["gru", "mamba"]))
    print(json.dumps({"analysis": str(output / "analysis.json"), "comparisons": contrast_rows}), flush=True)


def self_test():
    # All-on alarm can match one event perfectly while alarming on all negatives.
    arrays = {"y": np.array([0, 0, 1, 1, 0, 0]), "p": np.ones(6), "valid": np.ones(6, bool),
              "timestamp_s": np.arange(6) * .1, "coverage_s": np.full(6, .1)}
    result = aggregate_events([trial_events(arrays, .5)])
    assert result["event_f1"] == 1 and result["false_alarms_per_min"] == 0
    assert result["negative_alarm_time_fraction"] == 1
    np.testing.assert_allclose(result["observed_onset_signed_differences_median"], -.2)
    # Invalid gaps create separate episodes and no cross-gap matching.
    arrays["valid"] = np.array([1, 1, 0, 1, 1, 1], bool)
    result = aggregate_events([trial_events(arrays, .5)])
    assert result["event_tp"] == 1 and result["event_fp"] == 1
    assert result["left_censored_truth_segments"] == 1
    assert result["observed_onset_nonnegative_delays_n"] == 0
    # AP padding and label masks must equal flattened true endpoints exactly.
    second = {"y": np.array([1, 0]), "p": np.array([.8, .3]), "valid": np.array([1, 1], bool)}
    rank = padded_ap([arrays, second])
    expected = average_precision_score(np.r_[arrays["y"][arrays["valid"]], second["y"]],
                                       np.r_[arrays["p"][arrays["valid"]], second["p"]])
    np.testing.assert_allclose(rank([1, 1]), expected, atol=1e-14)
    print(json.dumps({"self_test": "passed", "checks": ["always-alarm pathology exposed", "invalid-span boundaries",
                                                         "left-censored onset excluded", "padding-free AP"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="work/results/real")
    parser.add_argument("--out", default="work/results/real_analysis")
    parser.add_argument("--data-audit", default="work/real_data/arq2021_audit.json")
    parser.add_argument("--suite", choices=["all", "held_trial", "loo"], default="all")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261010)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    self_test() if arguments.self_test else run(arguments)

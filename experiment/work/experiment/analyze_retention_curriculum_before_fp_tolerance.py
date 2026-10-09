"""Analyze the frozen 28-run training-path study without executing any models.

The primary family contains two curriculum-minus-direct query-AP effects, one
for each model, with 97.5% marginal percentile intervals (Bonferroni nominal
family coverage 95%). Architecture comparisons and interactions are descriptive.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score

from analyze_synthetic import RankedClusterAP, summarize_interval


METRICS = ("ap", "query_ap", "cue_shuffled_ap", "cue_shuffled_query_ap", "cue_shuffle_query_ap_drop")
MODELS = ("gru", "mamba")
ARMS = ("direct", "curriculum")
LAGS = (2, 4, 8, 16, 32, 64)
STEPS = list(range(100, 2401, 100))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_trace(result):
    training, arm = result["training"], result["arm"]
    trace = training["trace"]
    if [row["step"] for row in trace] != STEPS:
        raise ValueError("Expected all 24 validation points; no adaptive omission")
    if training["steps"] != 2400 or training["batch"] != 32:
        raise ValueError("Training budget differs from the frozen study")
    if training["optimizer_instances_created"] != 1 or training["model_instances_created"] != 1:
        raise ValueError("Model or optimizer was recreated during the run")
    if training["optimizer_reset_between_stages"] or training["final_optimizer_step_counters"] != [2400]:
        raise ValueError("Optimizer state continuity is not verified")
    for row in trace:
        lag = 64 if arm == "direct" else LAGS[(row["step"]-1)//400]
        if row["training_lag"] != lag or row["validation_lag"] != 64:
            raise ValueError("Training or validation lag differs from declared schedule")
        if row["record_exposures"] != row["step"]*32:
            raise ValueError("Recorded sample exposures differ from update count")
        if not all(np.isfinite(row[m]) and 0 <= row[m] <= 1 for m in ("val_ap", "val_query_ap")):
            raise ValueError("Invalid validation AP")
    best = max(trace, key=lambda row: row["val_ap"])
    if best["step"] != training["best_step"] or best["val_ap"] != training["best_val_ap"]:
        raise ValueError("Best checkpoint does not follow frozen earliest-maximum validation rule")
    if trace[-1]["val_ap"] != training["final_val_ap"] or trace[-1]["val_query_ap"] != training["final_val_query_ap"]:
        raise ValueError("Final validation metadata differs from step2400 trace")
    if len(training["stages"]) != 6 or sum(row["positive_exposures"] for row in training["stages"]) != training["positive_exposures"]:
        raise ValueError("Stage counts do not sum to total actual positive exposures")
    if training["record_exposures"] != 76800 or training["trace"][-1]["positive_exposures"] != training["positive_exposures"]:
        raise ValueError("Final exposure counts inconsistent")
    return best


def rank_metrics(ranks, weights):
    raw = np.array([rank(weights) for rank in ranks], dtype=np.float64)
    return np.r_[raw, raw[1]-raw[3]]


def paired_effects(values):
    """Input [..., model, arm, metric]; output path/model/interaction contrasts."""
    path = values[..., :, 1, :] - values[..., :, 0, :]
    architecture = values[..., 0, :, :] - values[..., 1, :, :]
    interaction = path[..., 0, :] - path[..., 1, :]
    return path, architecture, interaction


def bootstrap_arrays(caches, seeds, n_records, draws, bootstrap_seed, progress=False):
    rng = np.random.default_rng(bootstrap_seed)
    point = np.zeros((2, 2, len(METRICS)))
    ones = np.ones(n_records)
    for mi, model in enumerate(MODELS):
        for ai, arm in enumerate(ARMS):
            point[mi, ai] = np.mean([rank_metrics(caches[(model, arm, seed)], ones) for seed in seeds], axis=0)
    bootstrap = np.empty((draws, 2, 2, len(METRICS)), np.float64)
    start = time.perf_counter()
    for draw in range(draws):
        records = rng.multinomial(n_records, np.full(n_records, 1/n_records))
        repetitions = rng.multinomial(len(seeds), np.full(len(seeds), 1/len(seeds)))
        for mi, model in enumerate(MODELS):
            for ai, arm in enumerate(ARMS):
                totals = np.zeros(len(METRICS))
                for seed, weight in zip(seeds, repetitions):
                    if weight:
                        totals += weight * rank_metrics(caches[(model, arm, seed)], records)
                bootstrap[draw, mi, ai] = totals/len(seeds)
        if progress and (draw+1) % 250 == 0:
            print(json.dumps({"retention_bootstrap_draws": draw+1, "total": draws,
                              "seconds": round(time.perf_counter()-start, 1)}), flush=True)
    return point, bootstrap


def validate_manifest(manifest):
    if manifest["study"] != "retention-training-path-v1" or manifest["rate"] != .20:
        raise ValueError("Wrong retention study")
    if manifest["models"] != list(MODELS) or manifest["arms"] != list(ARMS) or manifest["final_seeds"] != list(range(5)):
        raise ValueError("Expected both registered models, paths, and five paired seeds")
    if [manifest[k] for k in ("n_train", "n_validation", "n_test", "steps", "batch")] != [4096, 4096, 16384, 2400, 32]:
        raise ValueError("Manifest data/training budget mismatch")
    if manifest["test_seed"] != 630001 or manifest["validation_seed"] != 620001:
        raise ValueError("Expected the frozen fresh validation/test namespace")
    if manifest["auxiliary_success"]["query_ap_greater_than"] != .90 or manifest["auxiliary_success"]["cue_shuffle_query_ap_drop_greater_than"] != .30:
        raise ValueError("Auxiliary success criterion changed")


def load_complete_study(source):
    manifest_path = source / "manifest.json"
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    fingerprint = digest(manifest_path)
    seeds = manifest["final_seeds"]
    entries, tune_entries, selections = [], [], {}
    missing = []
    for model in MODELS:
        selection_path = source / model / "selection.json"
        if not selection_path.is_file():
            missing.append(str(selection_path))
            continue
        selection = read_json(selection_path)
        if selection["manifest_sha256"] != fingerprint or not selection["shared_by_both_arms"]:
            raise ValueError("Learning-rate selection fingerprint/sharing mismatch")
        selections[model] = selection
        lr = selection["selected_lr"]
        for arm in ARMS:
            for candidate in manifest["learning_rates"]:
                folder = source / model / arm / f"tune_seed901_lr{candidate:g}"
                path = folder / "result.json"
                for artifact in ("result.json", "training_trace.json", "best.pt", "final.pt", "final_optimizer.pt"):
                    if not (folder/artifact).is_file():
                        missing.append(str(folder/artifact))
                tune_entries.append((model, arm, candidate, path))
            for seed in seeds:
                folder = source / model / arm / f"final_seed{seed}_lr{lr:g}"
                for name in ("result.json", "test_predictions.npz", "training_trace.json", "best.pt", "final.pt", "final_optimizer.pt"):
                    if not (folder/name).is_file():
                        missing.append(str(folder/name))
                entries.append((model, arm, seed, lr, folder))
    if missing:
        raise FileNotFoundError(f"Incomplete frozen 28-run study: {len(missing)} missing artifacts; first {missing[:8]}")
    return manifest, fingerprint, entries, tune_entries, selections


def run(args):
    source, destination = Path(args.source), Path(args.out)
    manifest, fingerprint, expected, expected_tune, selections = load_complete_study(source)
    seeds, n_records = manifest["final_seeds"], manifest["n_test"]
    caches, seed_rows, curves, tuning_curves = {}, [], [], []
    label_reference = None
    exposure_reference = {}
    max_metric_error = 0.
    tuning_ap = {}
    for model, arm, lr, path in expected_tune:
        result = read_json(path)
        if (result["model"], result["arm"], result["seed"], result["lr"], result["phase"]) != (model, arm, 901, lr, "tuning"):
            raise ValueError(f"Tuning metadata mismatch: {path}")
        if result["test_evaluated"]:
            raise ValueError("Tuning run evaluated the test set")
        validate_trace(result)
        saved_trace = read_json(path.parent / "training_trace.json")
        if not saved_trace["complete"] or saved_trace["trace"] != result["training"]["trace"]:
            raise ValueError("Tuning trace artifact differs from completion metadata")
        tuning_ap[(model, arm, lr)] = result["training"]["best_val_ap"]
        tuning_curves.extend({"model": model, "arm": arm, "seed": 901, "lr": lr, **row} for row in result["training"]["trace"])
    for model in MODELS:
        scores = {lr: np.mean([tuning_ap[(model, arm, lr)] for arm in ARMS]) for lr in manifest["learning_rates"]}
        chosen = max(scores, key=scores.get)
        if chosen != selections[model]["selected_lr"]:
            raise ValueError("Shared LR not selected by declared two-arm mean validation criterion")
    for model, arm, seed, lr, folder in expected:
        result = read_json(folder / "result.json")
        if (result["model"], result["arm"], result["seed"], result["lr"], result["phase"]) != (model, arm, seed, lr, "final"):
            raise ValueError(f"Final metadata mismatch: {folder}")
        if result["manifest_sha256"] != fingerprint:
            raise ValueError("Final run fingerprint mismatch")
        best = validate_trace(result)
        training = result["training"]
        saved_trace = read_json(folder / "training_trace.json")
        if not saved_trace["complete"] or saved_trace["trace"] != training["trace"]:
            raise ValueError("Final trace artifact differs from completion metadata")
        current_exposures = [(row["record_exposures"], row["positive_exposures"], row["query_exposures"])
                             for row in training["trace"]]
        if seed in exposure_reference and current_exposures != exposure_reference[seed]:
            raise ValueError("Record/positive/query exposures are not paired across all arms/models")
        exposure_reference.setdefault(seed, current_exposures)
        with np.load(folder / "test_predictions.npz", allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in ("y", "p", "query", "cue_shuffled_p", "endpoint", "cue", "bit")}
        if any(array.shape != (n_records,) for array in arrays.values()):
            raise ValueError("Each test record must have exactly one saved scalar endpoint")
        for key in ("p", "cue_shuffled_p"):
            if not np.isfinite(arrays[key]).all() or np.any((arrays[key] < 0) | (arrays[key] > 1)):
                raise ValueError("Invalid saved probabilities")
        if not np.isin(arrays["query"], [0, 1]).all() or not np.isin(arrays["bit"], [-1, 1]).all():
            raise ValueError("Invalid cue/query metadata")
        np.testing.assert_array_equal(arrays["y"], arrays["query"] * (arrays["bit"] > 0))
        np.testing.assert_array_equal(arrays["endpoint"] - arrays["cue"], np.full(n_records, 64))
        if np.any((arrays["endpoint"] < 96) | (arrays["endpoint"] > 127)):
            raise ValueError("Test endpoints outside frozen range")
        invariant = {key: arrays[key] for key in ("y", "query", "endpoint", "cue", "bit")}
        if label_reference is None:
            label_reference = {key: value.copy() for key, value in invariant.items()}
        else:
            for key in invariant:
                np.testing.assert_array_equal(invariant[key], label_reference[key])
        y, p, q, abp = (arrays[key][:, None] for key in ("y", "p", "query", "cue_shuffled_p"))
        ranks = (RankedClusterAP(y, p), RankedClusterAP(y, p, q > 0),
                 RankedClusterAP(y, abp), RankedClusterAP(y, abp, q > 0))
        computed = rank_metrics(ranks, np.ones(n_records))
        saved = result["metrics"]
        for name, value in zip(METRICS, computed):
            error = abs(float(value) - saved[name])
            max_metric_error = max(max_metric_error, error)
            if error > 1e-9:
                raise ValueError(f"Saved {name} disagrees with independent rank recomputation")
        success = computed[1] > .90 and computed[4] > .30
        if bool(saved["auxiliary_long_task_success"]) != bool(success):
            raise ValueError("Auxiliary success criterion is not the frozen .90/.30 rule")
        if saved["threshold"] != result["validation_threshold"]:
            raise ValueError("Final test threshold differs from validation choice")
        caches[(model, arm, seed)] = ranks
        row = {"model": model, "arm": arm, "seed": seed, "lr": lr,
               "parameters": result["parameters"], **saved,
               "best_step": training["best_step"], "best_stage": training["best_stage"],
               "best_training_lag": training["best_training_lag"],
               "best_val_ap": best["val_ap"], "best_val_query_ap": best["val_query_ap"],
               "final_val_ap": training["final_val_ap"], "final_val_query_ap": training["final_val_query_ap"],
               "training_seconds": training["seconds"], "record_exposures": training["record_exposures"],
               "positive_exposures": training["positive_exposures"], "query_exposures": training["query_exposures"],
               "unique_training_records_seen": training["unique_training_records_seen"]}
        seed_rows.append(row)
        curves.extend({"model": model, "arm": arm, "seed": seed, "lr": lr,
                       "selected_checkpoint": point["step"] == training["best_step"], **point}
                      for point in training["trace"])
    start = time.perf_counter()
    point, bootstrap = bootstrap_arrays(caches, seeds, n_records, args.draws, args.bootstrap_seed, progress=True)
    point_path, point_arch, point_interaction = paired_effects(point)
    boot_path, boot_arch, boot_interaction = paired_effects(bootstrap)
    cells = []
    summary_metrics = list(METRICS) + ["f1", "precision", "recall", "brier", "best_step", "best_val_ap",
                                      "best_val_query_ap", "final_val_ap", "final_val_query_ap",
                                      "positive_exposures", "query_exposures", "training_seconds"]
    for mi, model in enumerate(MODELS):
        for ai, arm in enumerate(ARMS):
            rows = [row for row in seed_rows if row["model"] == model and row["arm"] == arm]
            cell = {"model": model, "arm": arm, "seeds": len(rows), "parameters": rows[0]["parameters"], "shared_lr": rows[0]["lr"]}
            for metric in summary_metrics:
                values = np.array([row[metric] for row in rows], dtype=float)
                cell[metric + "_mean"], cell[metric + "_seed_sd"] = float(values.mean()), float(values.std(ddof=1))
            for qi, metric in enumerate(METRICS):
                interval = summarize_interval(bootstrap[:, mi, ai, qi], .95)
                cell[metric + "_descriptive_ci95"] = interval
            cell["auxiliary_success_seeds"] = sum(bool(row["auxiliary_long_task_success"]) for row in rows)
            cell["stable_success_all_five_seeds"] = cell["auxiliary_success_seeds"] == 5
            cells.append(cell)
    effects, comparisons, interactions = [], [], []
    for mi, model in enumerate(MODELS):
        for qi, metric in enumerate(METRICS):
            primary = metric == "query_ap"
            interval = summarize_interval(boot_path[:, mi, qi], .975 if primary else .95)
            effects.append({"model": model, "metric": metric, "contrast": "curriculum_minus_direct",
                            "estimate": float(point_path[mi, qi]), **interval,
                            "primary": primary, "multiplicity_family": "two model-specific query-AP path effects" if primary else None,
                            "positive_effect_interval_excludes_zero": bool(interval["lower"] > 0) if interval["undefined_draws"] == 0 else None})
    for ai, arm in enumerate(ARMS):
        for qi, metric in enumerate(("ap", "query_ap")):
            comparisons.append({"arm": arm, "metric": metric, "contrast": "gru_minus_mamba",
                                "estimate": float(point_arch[ai, qi]), **summarize_interval(boot_arch[:, ai, qi], .95),
                                "primary": False, "scope": "descriptive architecture contrast within fixed training path"})
    for qi, metric in enumerate(("ap", "query_ap")):
        interactions.append({"metric": metric, "contrast": "(curriculum-direct)_GRU minus (curriculum-direct)_Mamba",
                             "estimate": float(point_interaction[qi]), **summarize_interval(boot_interaction[:, qi], .95),
                             "primary": False})
    curve_cells = []
    for model, arm, step in itertools.product(MODELS, ARMS, STEPS):
        rows = [row for row in curves if (row["model"], row["arm"], row["step"]) == (model, arm, step)]
        row = {"model": model, "arm": arm, "step": step, "stage": rows[0]["stage"],
               "training_lag": rows[0]["training_lag"], "validation_lag": 64, "seeds": len(rows)}
        for metric in ("val_ap", "val_query_ap", "positive_exposures", "query_exposures", "train_loss"):
            values = np.array([r[metric] for r in rows], float)
            row[metric + "_mean"], row[metric + "_seed_sd"] = float(values.mean()), float(values.std(ddof=1))
        curve_cells.append(row)
    y, query = label_reference["y"], label_reference["query"]
    baseline = {"records": n_records, "positive_records": int(y.sum()), "query_records": int(query.sum()),
                "score_query_all_record_ap": float(average_precision_score(y, query)),
                "query_constant_score_ap": float(y[query > 0].mean()), "oracle_ap": float(average_precision_score(y, y))}
    report = {"study": manifest["study"], "manifest_sha256": fingerprint,
              "completed_final_runs": len(seed_rows), "completed_tuning_runs": len(expected_tune),
              "primary_metric": "long-lag query-conditioned AUPRC", "cells": cells,
              "path_effects": effects, "primary_path_effects": [r for r in effects if r["primary"]],
              "architecture_comparisons": comparisons, "path_by_architecture_interactions": interactions,
              "validation_curve_cells": curve_cells, "deterministic_baseline": baseline,
              "auxiliary_success_criterion": manifest["auxiliary_success"], "selected_learning_rates": selections,
              "checkpoint_reporting": {"test_predictions": "best long-validation-AP checkpoint only",
                                       "best_and_final_validation": "both reported per seed and cell, with all 24 long-validation points",
                                       "final_test_predictions": "not generated in the frozen training protocol; final validation is not final test"},
              "bootstrap": {"draws": args.draws, "seed": args.bootstrap_seed,
                            "method": "crossed paired resampling of 5 training replicates and 16384 independent test records",
                            "same_record_weights_across_models_arms_and_replicates": True,
                            "same_seed_weights_across_models_and_arms": True,
                            "primary_marginal_confidence": .975,
                            "primary_family": "2 query-AP curriculum-minus-direct effects, one per model",
                            "multiplicity": "Bonferroni nominal family coverage95%; percentile intervals remain approximate",
                            "descriptive_confidence": .95, "seconds": time.perf_counter()-start},
              "validation": {"max_saved_vs_recomputed_metric_abs_error": max_metric_error,
                             "identical_test_records_across_all_runs": True,
                             "positive_query_and_record_exposures_paired": True,
                             "continuous_optimizer_step2400_all_runs": True,
                             "all_24_long_validation_points_present": True,
                             "both_arms_share_selected_lr_within_model": True},
              "limitations": [
                  "Mechanism redesign after earlier results, using fresh test records; not part of the original v1 blind protocol.",
                  "Curriculum adds short-lag variants and has400 long-lag updates; direct has2400. Equal total updates/labels are not equal fixed-task input exposure.",
                  "Bootstrap conditions on selected shared learning rates, validation splits, checkpoint choices, model sizes and budgets; it does not repeat selection.",
                  "Five training replicates weakly resolve training-process tails; intervals are approximate.",
                  "Architecture rankings, full AP, shuffled diagnostics and interactions are descriptive rather than additional primary confirmations.",
                  "If neither path resolves the long-task floor, intrinsic memory-capacity claims remain unsupported.",
                  "Success concerns algorithmic one-bit retention, not tactile physics, future slip prediction or closed-loop recovery.",
                  "Saved final validation metrics do not substitute for ungenerated final-checkpoint test predictions.",
              ]}
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "analysis.json", report)
    for name, rows in [("per_seed_metrics.csv", seed_rows), ("cell_summary.csv", cells),
                       ("path_effects.csv", effects), ("architecture_comparisons.csv", comparisons),
                       ("path_architecture_interactions.csv", interactions),
                       ("validation_curves_per_seed.csv", curves), ("validation_curves_summary.csv", curve_cells),
                       ("tuning_validation_curves.csv", tuning_curves)]:
        write_csv(destination / name, rows)
    np.savez_compressed(destination / "bootstrap_draws.npz", cells=bootstrap, path_effects=boot_path,
                        architecture=boot_arch, interactions=boot_interaction,
                        models=np.array(MODELS), arms=np.array(ARMS), metrics=np.array(METRICS))
    print(json.dumps({"analysis": str(destination/"analysis.json"), "primary_path_effects": report["primary_path_effects"]}), flush=True)


def self_test():
    rng = np.random.default_rng(26416)
    max_error, compared = 0., 0
    for ties in (False, True):
        y = rng.integers(0, 2, size=(37, 1))
        q = np.maximum(y, rng.integers(0, 2, size=(37, 1)))
        p, abp = rng.random(y.shape), rng.random(y.shape)
        if ties:
            p, abp = np.round(p, 1), np.round(abp, 1)
        ranks = (RankedClusterAP(y, p), RankedClusterAP(y, p, q > 0),
                 RankedClusterAP(y, abp), RankedClusterAP(y, abp, q > 0))
        for _ in range(100):
            sampled = rng.integers(0, 37, size=37)
            weight = np.bincount(sampled, minlength=37)
            actual = rank_metrics(ranks, weight)
            expected = []
            for prediction, selection in ((p, np.ones_like(q, bool)), (p, q > 0),
                                           (abp, np.ones_like(q, bool)), (abp, q > 0)):
                yy, pp, mm = y[sampled], prediction[sampled], selection[sampled]
                expected.append(float(average_precision_score(yy[mm], pp[mm])))
            expected.append(expected[1]-expected[3])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
            max_error = max(max_error, float(np.max(np.abs(actual-expected))))
            compared += 1
    # Completely identical paired cells must have zero effect in every bootstrap draw.
    identical = {(model, arm, seed): ranks for model, arm, seed in itertools.product(MODELS, ARMS, range(5))}
    point, draws = bootstrap_arrays(identical, range(5), 37, 100, 773)
    for effects in (*paired_effects(point), *paired_effects(draws)):
        np.testing.assert_array_equal(effects, np.zeros_like(effects))
    # Algebra and orientation: curriculum-direct, then GRU-minus-Mamba.
    fake = np.arange(2*2*5, dtype=float).reshape(2, 2, 5)
    path, architecture, interaction = paired_effects(fake)
    np.testing.assert_array_equal(path, fake[:, 1]-fake[:, 0])
    np.testing.assert_array_equal(architecture, fake[0]-fake[1])
    np.testing.assert_array_equal(interaction, (fake[0, 1]-fake[0, 0])-(fake[1, 1]-fake[1, 0]))
    criterion_cases = [(0.91, .31, True), (.90, .31, False), (.91, .30, False), (.51, .01, False)]
    assert all((ap > .90 and drop > .30) == expected for ap, drop, expected in criterion_cases)
    report = {"status": "all CPU analysis tests passed; no model executed", "weighted_ap_cases": compared,
              "max_abs_error_vs_explicit_record_replication": max_error,
              "tied_scores_and_query_masks": True, "paired_null_exactly_zero": True,
              "path_architecture_interaction_orientation": True, "frozen_auxiliary_threshold_boundaries": True}
    write_json(Path(__file__).with_name("retention_analysis_self_test.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="work/results/retention_curriculum")
    parser.add_argument("--out", default="work/results/retention_curriculum_analysis")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261012)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else run(args)

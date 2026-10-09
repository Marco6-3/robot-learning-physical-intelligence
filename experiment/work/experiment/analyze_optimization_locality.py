"""Independent prediction-only analysis of the adaptive optimization/locality study.

No model is trained or selected here. Shared test-record and training-seed
bootstrap weights preserve pairing across every architecture and sampler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from analyze_synthetic import save_json, write_csv, summarize_interval


VARIANTS = ("gru", "gru_conv1", "gru_conv4", "mamba_conv4", "mamba_conv1")
SAMPLERS = ("uniform", "balanced")
SEEDS = tuple(range(10))
METRICS = ("ap", "query_ap", "cue_shuffled_ap", "cue_shuffled_query_ap")
CELLS = tuple((variant, sampler) for variant in VARIANTS for sampler in SAMPLERS)
N_TRAIN, N_TEST, TEST_SEED = 4096, 65536, 530001


class RankedScalarAP:
    """Exact record-weighted AP, summing only score groups containing positives.

    This is the same tied-score, non-interpolated AP as sklearn. It avoids
    allocating full-length positive cumulative sums for rare binary outcomes.
    Integer bootstrap weights are equivalent to physical record replication.
    """

    def __init__(self, y, p, selected=None):
        y, p = np.asarray(y), np.asarray(p)
        if y.ndim != 1 or p.shape != y.shape:
            raise ValueError("Expected equal scalar record vectors")
        if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
            raise ValueError("Nonbinary labels or nonfinite prediction")
        selected = np.ones(len(y), bool) if selected is None else np.asarray(selected, bool)
        if selected.shape != y.shape or not selected.any():
            raise ValueError("Invalid or empty record selection")
        records = np.flatnonzero(selected)
        local_order = np.argsort(-p[records], kind="stable")
        self.order = records[local_order].astype(np.int32)
        sorted_p = p[self.order]
        positive_positions = np.flatnonzero(y[self.order] == 1)
        self.positive_records = self.order[positive_positions]
        if not len(positive_positions):
            raise ValueError("AP is undefined when the selected labels contain no positives")
        tied_ends = np.searchsorted(-sorted_p, -sorted_p[positive_positions], side="right") - 1
        self.ends, self.starts = np.unique(tied_ends, return_index=True)
        self.n_records = len(y)

    def __call__(self, record_weights):
        weights = np.asarray(record_weights)
        if weights.shape != (self.n_records,):
            raise ValueError("Record weight length mismatch")
        positive_weights = weights[self.positive_records].astype(np.float64, copy=False)
        positive_group_weights = np.add.reduceat(positive_weights, self.starts)
        total = positive_group_weights.sum()
        if total == 0:
            return float("nan")
        cumulative_positives = np.cumsum(positive_group_weights)
        counts = np.cumsum(weights[self.order], dtype=np.float64)[self.ends]
        precision = np.divide(cumulative_positives, counts,
                              out=np.zeros_like(counts), where=counts > 0)
        return float(np.dot(precision, positive_group_weights) / total)


def scalar_metadata(n, seed, prevalence=.01, lag=2):
    """Rebuild independent scalar RNG streams without allocating full inputs."""
    streams = np.random.SeedSequence(seed).spawn(5)
    endpoint = np.random.default_rng(streams[1]).integers(96, 128, size=n)
    bit = np.random.default_rng(streams[2]).choice([-1., 1.], size=n).astype(np.float32)
    query = (np.random.default_rng(streams[3]).random(n) < 2 * prevalence).astype(np.float32)
    return {"endpoint": endpoint, "cue": endpoint - lag, "bit": bit, "query": query,
            "y": (query * (bit > 0)).astype(np.float32)}


def primary_contrasts(values):
    """Last two axes are [cell,metric]; preserve optional bootstrap axes."""
    at = lambda name, sampler: values[..., CELLS.index((name, sampler)), :]
    return np.stack([at("gru", "balanced") - at("gru", "uniform"),
                     at("gru_conv4", "uniform") - at("gru_conv1", "uniform"),
                     at("mamba_conv4", "uniform") - at("mamba_conv1", "uniform")], axis=-2)


def sampling_audit(seed, sampler, n=N_TRAIN):
    """Independently replay the fixed NumPy sampler without importing training."""
    y = scalar_metadata(n, 510000 + seed)["y"]
    positive, negative = np.flatnonzero(y > 0), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed + 1729)
    counts, visits = [], np.zeros(n, dtype=np.int64)
    h = hashlib.sha256()
    for _ in range(400):
        if sampler == "uniform":
            indices = rng.integers(0, n, size=32)
        elif sampler == "balanced":
            indices = np.concatenate([rng.choice(positive, 16), rng.choice(negative, 16)])
        else:
            raise ValueError(sampler)
        h.update(indices.astype(np.int64).tobytes())
        np.add.at(visits, indices, 1)
        counts.append(int(y[indices].sum()))
    return {"batch_positive_counts": counts, "batch_indices_sha256": h.hexdigest(),
            "visits_sha256": hashlib.sha256(visits.tobytes()).hexdigest(),
            "positive_presentations": sum(counts), "total_presentations": 400 * 32,
            "zero_positive_batches": sum(value == 0 for value in counts),
            "unique_positive_visited": int((visits[positive] > 0).sum()),
            "unique_negative_visited": int((visits[negative] > 0).sum())}


def load_inputs(source):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, expected_value in {"task": "sampling-and-locality-v3", "variants": list(VARIANTS),
                                "samplers": list(SAMPLERS), "seeds": list(SEEDS), "rate": .01,
                                "lag": 2, "n_train": N_TRAIN, "n_val": 8192, "n_test": N_TEST,
                                "steps": 400, "batch": 32, "train_data_seed_base": 510000,
                                "tune_data_seed": 510901, "val_data_seed": 520001,
                                "test_data_seed": TEST_SEED, "planned_tuning_runs": 20,
                                "planned_main_runs": 100}.items():
        if manifest[key] != expected_value:
            raise ValueError(f"Frozen manifest mismatch for {key}")
    expected = scalar_metadata(N_TEST, TEST_SEED)
    selected = expected["query"] > 0
    ones = np.ones(N_TEST, np.int32)
    caches, rows, validation = {}, [], {"max_saved_metric_abs_error": 0., "files": []}
    sampling = {(seed, sampler): sampling_audit(seed, sampler) for seed in SEEDS for sampler in SAMPLERS}
    initial_hashes = {}
    values = np.empty((len(CELLS), len(SEEDS), len(METRICS)))
    missing = []
    for variant, sampler in CELLS:
        for seed in SEEDS:
            stem = f"{variant}_{sampler}_seed{seed}"
            for suffix in (".json", "_predictions.npz", ".pt"):
                path = source / (stem + suffix)
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Incomplete fixed 100-run study: {len(missing)} missing artifacts; first {missing[:8]}")
    for ci, (variant, sampler) in enumerate(CELLS):
        for si, seed in enumerate(SEEDS):
            stem = f"{variant}_{sampler}_seed{seed}"
            jp, pp = source / (stem + ".json"), source / (stem + "_predictions.npz")
            result = json.loads(jp.read_text(encoding="utf-8"))
            if (result["variant"], result["sampler"], result["seed"]) != (variant, sampler, seed):
                raise ValueError(f"Result identity mismatch: {jp}")
            if result["n_train"] != N_TRAIN:
                raise ValueError(f"Unexpected train budget: {jp}")
            with np.load(pp, allow_pickle=False) as arrays:
                for key, correct in expected.items():
                    np.testing.assert_array_equal(arrays[key], correct, err_msg=f"{pp}: {key}")
                p, shuffled = arrays["p"].copy(), arrays["cue_shuffled_p"].copy()
            for scores in (p, shuffled):
                if scores.shape != (N_TEST,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                    raise ValueError(f"Invalid scalar probability vector: {pp}")
            ranked = tuple(RankedScalarAP(expected["y"], scores, selection)
                           for scores, selection in [(p, None), (p, selected),
                                                     (shuffled, None), (shuffled, selected)])
            recomputed = np.array([fn(ones) for fn in ranked])
            caches[(ci, si)] = ranked
            values[ci, si] = recomputed
            for mi, metric in enumerate(METRICS):
                error = abs(result["metrics"][metric] - recomputed[mi])
                validation["max_saved_metric_abs_error"] = max(validation["max_saved_metric_abs_error"], error)
                if error > 1e-7:
                    raise ValueError(f"Saved/recomputed {metric} mismatch: {jp}, error={error}")
            train_meta = scalar_metadata(N_TRAIN, 510000 + seed)
            npos = int(train_meta["y"].sum())
            training = result["training"]
            if training["train_positives"] != npos:
                raise ValueError(f"Recorded training-pool positive count mismatch: {jp}")
            np.testing.assert_allclose(training["train_prevalence"], npos / N_TRAIN, atol=0, rtol=0)
            np.testing.assert_allclose(training["balanced_loss_multiplier"], 2 * (1 - npos / N_TRAIN), atol=0, rtol=0)
            for key, expected_value in sampling[(seed, sampler)].items():
                if training[key] != expected_value:
                    raise ValueError(f"Independent sampling replay mismatch ({key}): {jp}")
            previous_initial = initial_hashes.setdefault((variant, seed), training["initial_state_sha256"])
            if previous_initial != training["initial_state_sha256"]:
                raise ValueError(f"Sampler pair does not share exact initial state: {variant}, seed{seed}")
            best = max(training["trace"], key=lambda item: item["val_ap"])
            if best["step"] != training["best_step"] or best["val_ap"] != training["best_val_ap"]:
                raise ValueError(f"Recorded best checkpoint does not match validation trace: {jp}")
            if [item["step"] for item in training["trace"]] != [100, 200, 300, 400]:
                raise ValueError(f"Unexpected checkpoint evaluation schedule: {jp}")
            if "pos_weight" in training:
                np.testing.assert_allclose(training["pos_weight"], (N_TRAIN - npos) / npos, rtol=1e-6)
            for field, correct in [("steps", 400), ("batch", 32)]:
                if field in training and training[field] != correct:
                    raise ValueError(f"Unexpected training {field}: {jp}")
            row = {"variant": variant, "sampler": sampler, "seed": seed,
                   "n_train": N_TRAIN, "parameters": result["parameters"],
                   "train_positive_records": npos, "train_prevalence": npos / N_TRAIN,
                   **dict(zip(METRICS, map(float, recomputed)))}
            row.update({f"training_{key}": value for key, value in training.items()
                        if value is None or isinstance(value, (str, int, float, bool))})
            rows.append(row)
            validation["files"].append({"result": jp.name,
                                         "result_sha256": hashlib.sha256(jp.read_bytes()).hexdigest(),
                                         "predictions": pp.name,
                                         "predictions_sha256": hashlib.sha256(pp.read_bytes()).hexdigest()})
    # Every within-family contrast is evaluated under the same selected LR.
    family_lrs = {}
    for family, variants in [("gru", VARIANTS[:3]), ("mamba", VARIANTS[3:])]:
        rates = {row["training_lr"] for row in rows if row["variant"] in variants}
        if len(rates) != 1:
            raise ValueError(f"Selected LR not shared within {family} family: {rates}")
        family_lrs[family] = next(iter(rates))
    tuning = []
    for variant, sampler in CELLS:
        for lr in (.001, .003):
            path = source / f"tune_{variant}_{sampler}_lr{lr:g}.json"
            entry = json.loads(path.read_text(encoding="utf-8"))
            if (entry["variant"], entry["sampler"], entry["seed"], entry["lr"]) != (variant, sampler, 901, lr):
                raise ValueError(f"Tuning result identity mismatch: {path}")
            tuning.append(entry)
    selection = json.loads((source / "family_lr_selection.json").read_text(encoding="utf-8"))
    for family, variants in [("gru", VARIANTS[:3]), ("mamba", VARIANTS[3:])]:
        scores = {str(lr): float(np.mean([t["training"]["best_val_ap"] for t in tuning
                                         if t["variant"] in variants and t["lr"] == lr])) for lr in (.001, .003)}
        if selection[family]["mean_validation_ap"] != scores:
            raise ValueError(f"Family tuning average mismatch: {family}")
        selected_lr = float(max(scores, key=scores.get))
        if selection[family]["selected_lr"] != selected_lr or family_lrs[family] != selected_lr:
            raise ValueError(f"Family LR does not follow declared validation rule: {family}")
    validation.update({"exact_test_metadata_reconstruction": True, "all_100_results_complete": True,
                       "test_labels_queries_endpoints_cues_bits_paired": True, "family_shared_lr": family_lrs,
                       "all_20_tuning_runs_and_family_selection_verified": True,
                       "all_sampling_counts_indices_and_visits_replayed": True,
                       "sampler_pairs_have_identical_initial_state_hashes": True,
                       "checkpoints_selected_only_by_recorded_validation_trace": True})
    return manifest, expected, caches, rows, values, validation


def make_interval(values, estimate, confidence):
    return {"estimate": float(estimate), **summarize_interval(values, confidence)}


def run(args):
    source, output = Path(args.source), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest, metadata, caches, rows, seed_values, validation = load_inputs(source)
    if args.draws < 1000:
        raise ValueError("Use at least 1000 draws; 5000 or more is recommended for the primary 98.333% tails")
    points = seed_values.mean(axis=1)
    rng = np.random.default_rng(args.bootstrap_seed)
    boot = np.empty((args.draws, len(CELLS), len(METRICS)))
    baseline_boot = np.empty((args.draws, 2))
    positive = metadata["y"] > 0
    query = metadata["query"] > 0
    start = time.perf_counter()
    for draw in range(args.draws):
        record_weights = rng.multinomial(N_TEST, np.full(N_TEST, 1 / N_TEST))
        seed_weights = rng.multinomial(len(SEEDS), np.full(len(SEEDS), 1 / len(SEEDS)))
        baseline_boot[draw] = [record_weights[positive].sum() / N_TEST,
                              record_weights[positive].sum() / record_weights[query].sum()]
        for ci in range(len(CELLS)):
            total = np.zeros(len(METRICS))
            for si, weight in enumerate(seed_weights):
                if weight:
                    total += weight * np.array([fn(record_weights) for fn in caches[(ci, si)]])
            boot[draw, ci] = total / len(SEEDS)
        if (draw + 1) % 100 == 0:
            print(json.dumps({"optimization_bootstrap_draws": draw + 1, "total": args.draws,
                              "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    conf = 1 - .05 / 3
    primary_point, primary_boot = primary_contrasts(points), primary_contrasts(boot)
    descriptions = ["GRU balanced - GRU uniform", "GRU conv4 uniform - GRU conv1 uniform",
                    "Mamba conv4 uniform - Mamba conv1 uniform"]
    primary = []
    for ci, name in enumerate(("C1", "C2", "C3")):
        for mi, metric in enumerate(METRICS[:2]):
            interval = make_interval(primary_boot[:, ci, mi], primary_point[ci, mi], conf if mi == 0 else .95)
            supported = interval["undefined_draws"] == 0 and interval["lower"] > args.margin
            primary.append({"contrast": name, "definition": descriptions[ci], "metric": metric,
                            **interval, "margin": args.margin, "primary": mi == 0,
                            "meaningful_positive_effect_supported": bool(supported),
                            "opposite_meaningful_effect_supported": bool(interval["undefined_draws"] == 0 and interval["upper"] < -args.margin)})
    cells = []
    for ci, (variant, sampler) in enumerate(CELLS):
        for mi, metric in enumerate(METRICS):
            cells.append({"variant": variant, "sampler": sampler, "metric": metric,
                          **make_interval(boot[:, ci, mi], points[ci, mi], .95),
                          "seed_min": float(seed_values[ci, :, mi].min()),
                          "seed_max": float(seed_values[ci, :, mi].max()),
                          "seed_median": float(np.median(seed_values[ci, :, mi])),
                          "seed_sd": float(seed_values[ci, :, mi].std(ddof=1)),
                          "seed_ap_ge_0_9_count": int((seed_values[ci, :, 0] >= .9).sum()),
                          "seed_ap_lt_0_7_count": int((seed_values[ci, :, 0] < .7).sum()),
                          "primary": False})
    secondary = []
    gaps, gap_points = {}, {}
    for sampler in SAMPLERS:
        gi, mi = CELLS.index(("gru", sampler)), CELLS.index(("mamba_conv4", sampler))
        gaps[sampler], gap_points[sampler] = boot[:, mi] - boot[:, gi], points[mi] - points[gi]
        for metric_index, metric in enumerate(METRICS[:2]):
            for confidence in (.90, .95):
                interval = make_interval(gaps[sampler][:, metric_index], gap_points[sampler][metric_index], confidence)
                secondary.append({"contrast": "Mamba conv4 - GRU", "sampler": sampler, "metric": metric,
                                  **interval, "equivalence_margin": args.margin,
                                  "equivalence_supported": bool(interval["undefined_draws"] == 0 and
                                                                interval["lower"] > -args.margin and interval["upper"] < args.margin),
                                  "primary": False})
    for mi, metric in enumerate(METRICS[:2]):
        secondary.append({"contrast": "(Mamba conv4 - GRU) uniform - (Mamba conv4 - GRU) balanced",
                          "metric": metric, **make_interval(gaps["uniform"][:, mi] - gaps["balanced"][:, mi],
                                                           gap_points["uniform"][mi] - gap_points["balanced"][mi], .95),
                          "primary": False})
    cue_diagnostics = []
    query_baseline = int(metadata["y"].sum()) / int(metadata["query"].sum())
    for ci, (variant, sampler) in enumerate(CELLS):
        for mi, metric in enumerate(METRICS[:2]):
            delta = boot[:, ci, mi] - boot[:, ci, mi + 2]
            shuffled_minus_baseline = boot[:, ci, mi + 2] - baseline_boot[:, 1]
            reference_interval = make_interval(shuffled_minus_baseline,
                                               points[ci, mi + 2] - query_baseline, .95)
            cue_diagnostics.append({"variant": variant, "sampler": sampler, "metric": metric,
                                    "original_minus_shuffled": make_interval(delta, points[ci, mi] - points[ci, mi + 2], .95),
                                    "shuffled_minus_query_reference": reference_interval,
                                    "shuffled_within_reference_plus_minus_0_05": bool(reference_interval["undefined_draws"] == 0 and
                                                                                     reference_interval["lower"] > -.05 and reference_interval["upper"] < .05),
                                    "interpretation": "query gate is preserved, so the relevant all-record reference is query-only AP, not overall prevalence",
                                    "primary": False})
    gru_balanced = CELLS.index(("gru", "balanced"))
    performance_interval = make_interval(boot[:, gru_balanced, 0], points[gru_balanced, 0], .95)
    c1 = next(row for row in primary if row["contrast"] == "C1" and row["metric"] == "ap")
    cue = next(row for row in cue_diagnostics if row["variant"] == "gru" and row["sampler"] == "balanced" and row["metric"] == "query_ap")
    rescue = {"primary_C1_meaningful_gain": c1["meaningful_positive_effect_supported"],
              "balanced_gru_ap_95_interval": performance_interval,
              "high_ap_threshold": args.rescue_ap,
              "high_ap_supported": bool(performance_interval["undefined_draws"] == 0 and performance_interval["lower"] > args.rescue_ap),
              "query_cue_shuffle_drop_95_interval": cue["original_minus_shuffled"],
              "query_shuffled_near_reference_auxiliary": cue["shuffled_within_reference_plus_minus_0_05"],
              "interpretation": "high performance and cue-shuffle checks are auxiliary diagnostics; only C1/C2/C3 AP belong to the multiplicity-adjusted family"}
    mc_stability = []
    for ci, name in enumerate(("C1", "C2", "C3")):
        intervals = [summarize_interval(part, conf) for part in np.array_split(primary_boot[:, ci, 0], 5)]
        mc_stability.append({"contrast": name, "five_disjoint_draw_blocks": intervals,
                             "lower_endpoint_range": [min(x["lower"] for x in intervals), max(x["lower"] for x in intervals)],
                             "upper_endpoint_range": [min(x["upper"] for x in intervals), max(x["upper"] for x in intervals)],
                             "interpretation": "Monte Carlo stability diagnostic only; no significance-based stopping"})
    baselines = []
    for split, n, seed in [("tuning", N_TRAIN, 510901), ("validation", 8192, 520001), ("test", N_TEST, TEST_SEED)] + [
            (f"train_seed{seed}", N_TRAIN, 510000 + seed) for seed in SEEDS]:
        item = scalar_metadata(n, seed)
        baselines.append({"split": split, "generator_seed": seed, "n_records": n,
                          "positive_records": int(item["y"].sum()), "query_records": int(item["query"].sum()),
                          "prevalence": float(item["y"].mean()),
                          "query_only_ap": int(item["y"].sum()) / int(item["query"].sum()), "oracle_ap": 1.})
    report = {"task": "adaptive-optimization-locality-mechanism-study",
              "source_manifest": manifest, "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "input_runs": len(rows), "cells": cells, "primary_contrasts": primary,
              "secondary_contrasts": secondary, "cue_diagnostics": cue_diagnostics,
              "rescue_diagnostics": rescue, "data_counts_and_baselines": baselines,
              "validation": validation,
              "bootstrap": {"draws": args.draws, "random_seed": args.bootstrap_seed,
                            "method": "paired crossed training-seed and independent-test-record percentile bootstrap",
                            "pairing": "same seed and record weights for all ten cells and all four metrics",
                            "ap": "exact tied-score non-interpolated AP, equal to physical record replication",
                            "primary_confidence": conf, "multiplicity_scope": "C1/C2/C3 AP only within this new adaptive study",
                            "seconds": time.perf_counter() - start, "tail_monte_carlo_diagnostic": mc_stability},
              "limitations": [
                  "This study was designed after previous outcomes were inspected; its new holdout and frozen design test specific follow-up mechanisms, not the original untouched hypotheses.",
                  "Only ten training seeds; percentile bootstrap tail coverage is approximate, especially if outcomes are bimodal.",
                  "Seed variation combines fresh training-pool and model-initialization variation; test data are shared and paired.",
                  "Intervals condition on family-selected learning rates and validation-selected checkpoints; selection itself is not rerun in the bootstrap.",
                  "Stratification changes positive presentation and repetition counts. Equal raw pools, updates and expected objectives do not mean equal positive exposure or identical Adam updates.",
                  "conv4-versus-conv1 conclusions concern the tested parameterizations and initializations; they are not a universal memory-capacity statement.",
                  "The task is a sparse independent-endpoint synthetic short-delay task; it cannot establish long-memory, real-tactile or closed-loop claims.",
                  "Auxiliary 95/90-percent intervals, cue diagnostics and high-AP thresholds are not additional multiplicity-corrected primary tests.",
              ]}
    save_json(output / "analysis.json", report)
    write_csv(output / "per_seed_metrics.csv", rows)
    write_csv(output / "cell_summary.csv", cells)
    write_csv(output / "primary_contrasts.csv", primary)
    write_csv(output / "secondary_contrasts.csv", secondary)
    write_csv(output / "data_counts_and_baselines.csv", baselines)
    np.savez_compressed(output / "bootstrap_draws.npz", cell_metrics=boot, primary_contrasts=primary_boot,
                        baselines=baseline_boot, seed_metrics=seed_values,
                        cell_names=np.array([f"{name}_{sampler}" for name, sampler in CELLS]),
                        metric_names=np.array(METRICS), seeds=np.array(SEEDS))
    print(json.dumps({"analysis": str(output / "analysis.json"), "primary_contrasts": primary,
                      "rescue_diagnostics": rescue}), flush=True)


def self_test():
    rng = np.random.default_rng(91021)
    maximum = 0.
    checks = 0
    for n, rate in [(13, .4), (81, .1), (4096, .01)]:
        y = (rng.random(n) < rate).astype(int)
        y[0] = 1
        for tied in (False, True):
            p = rng.random(n)
            if tied:
                p = np.round(p, 1)
            for conditional in (False, True):
                selected = rng.random(n) < .7 if conditional else np.ones(n, bool)
                selected[0] = True
                cached = RankedScalarAP(y, p, selected)
                for _ in range(25):
                    idx = rng.integers(n, size=n)
                    w = np.bincount(idx, minlength=n)
                    labels, scores = y[idx][selected[idx]], p[idx][selected[idx]]
                    actual = cached(w)
                    if not labels.sum():
                        assert np.isnan(actual)
                    else:
                        expected = average_precision_score(labels, scores)
                        maximum = max(maximum, abs(actual - expected))
                        np.testing.assert_allclose(actual, expected, atol=2e-15)
                    checks += 1
    # Every score tied: non-interpolated AP is exactly the weighted prevalence.
    y = np.array([1, 0, 1, 0, 0])
    weights = np.array([2, 4, 1, 0, 1])
    np.testing.assert_allclose(RankedScalarAP(y, np.ones(5))(weights), 3 / 8)
    try:
        RankedScalarAP(np.zeros(3), np.ones(3))
    except ValueError:
        pass
    else:
        raise AssertionError("All-negative AP input must be rejected")
    points = np.arange(40).reshape(10, 4)
    np.testing.assert_array_equal(primary_contrasts(points), np.array([[4] * 4, [8] * 4, [-8] * 4]))
    np.testing.assert_array_equal(primary_contrasts(np.stack([points, 2 * points])),
                                  np.stack([primary_contrasts(points), 2 * primary_contrasts(points)]))
    test = scalar_metadata(N_TEST, TEST_SEED)
    assert int(test["y"].sum()) == 671 and int(test["query"].sum()) == 1316
    replay = sampling_audit(0, "balanced")
    assert replay["positive_presentations"] == 6400 and replay["zero_positive_batches"] == 0
    assert replay["unique_positive_visited"] == 47
    print(json.dumps({"self_test": "passed", "replication_checks": checks,
                      "max_ap_error": maximum, "weighted_ties_and_zero_positive_draw": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="work/results/optimization_locality_v3")
    parser.add_argument("--out", default="work/results/optimization_locality_v3_analysis")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261012)
    parser.add_argument("--margin", type=float, default=.03)
    parser.add_argument("--rescue-ap", type=float, default=.90)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)

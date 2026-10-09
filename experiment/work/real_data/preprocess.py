"""Loss-minimizing trial export and label/timestamp audit; no learned transforms."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "deps"))
import h5py

LABELS = {
    0: "no slip: static held object OR object not held (ambiguous contact state)",
    1: "slip while arm static", 2: "object released", 3: "object grasped",
    4: "no slip while arm moving", 5: "slip while arm moving", 6: "other tactile event; excluded from binary targets",
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def export_arq():
    dest = ROOT / "processed"
    dest.mkdir(exist_ok=True)
    records = []
    counts = Counter()
    intervals = []
    for path in sorted((ROOT / "raw").glob("*.h5")):
        with h5py.File(path, "r") as f:
            groups = []
            f.visititems(lambda n, obj: groups.append(n) if isinstance(obj, h5py.Group) and "tactile_data_raw" in obj else None)
            for name in sorted(groups):
                group = f[name]
                source = group["tactile_data_raw"][:]
                x = source.transpose(2, 0, 1).reshape(-1, 54).astype(np.float32)
                # Raw readings are small integers; verify float32 export is lossless.
                assert np.array_equal(x.astype(np.float64), source.transpose(2, 0, 1).reshape(-1, 54))
                y = group["tactile_slips_label"][:].astype(np.int8)
                t = group["tactile_timestamps"][:].astype(np.float64) / 1000.0
                assert len(x) == len(y) == len(t)
                assert np.isfinite(x).all() and np.isfinite(t).all()
                assert set(np.unique(y)).issubset(LABELS)
                dt = np.diff(t)
                monotonic = bool((dt > 0).all())
                binary = np.isin(y, [1, 5]).astype(np.int8)
                valid = y != 6
                # Do not count an already-present slip at recording start as an onset.
                onset = np.zeros(len(y), dtype=bool)
                onset[1:] = (binary[1:] == 1) & (binary[:-1] == 0) & valid[1:] & valid[:-1]
                parts = name.split("/")
                obj, pose, experiment = parts[-3:]
                trial = "_".join(parts[-3:])
                out = dest / (trial + ".npz")
                np.savez_compressed(out, x=x, label=y, timestamp_s=t, slip=binary, valid=valid, onset=onset)
                c = Counter(map(int, y))
                counts.update(c)
                intervals.extend(dt.tolist())
                rec = {"trial_id": trial, "object": obj, "pose": pose, "experiment": int(experiment[3:]),
                       "source_file": path.name, "source_group": name, "path": str(out.relative_to(ROOT)),
                       "frames": len(y), "duration_s": float(t[-1]-t[0]), "slip_frames": int(binary.sum()),
                       "valid_frames": int(valid.sum()), "onset_count": int(onset.sum()),
                       "label_counts": {str(k): int(v) for k,v in sorted(c.items())},
                       "timestamp_strictly_increasing": monotonic,
                       "median_dt_s": float(np.median(dt)), "max_dt_s": float(np.max(dt)),
                       "nonpositive_intervals": int((dt <= 0).sum()), "gaps_above_20ms": int((dt > .02).sum()),
                       "sha256": sha256(out)}
                records.append(rec)
    if len(records) != 90:
        raise RuntimeError(f"Expected the documented 90 trials, found {len(records)}")
    split = {"description": "Deterministic trial split, fixed before model outcomes; never split overlapping windows across sets",
             "train": [r["trial_id"] for r in records if r["experiment"] <= 6],
             "validation": [r["trial_id"] for r in records if r["experiment"] in [7,8]],
             "test": [r["trial_id"] for r in records if r["experiment"] in [9,10]]}
    leave_object = []
    for obj in sorted({r["object"] for r in records}):
        leave_object.append({"held_out_object": obj,
                            "train": [r["trial_id"] for r in records if r["object"] != obj and r["experiment"] <= 8],
                            "validation": [r["trial_id"] for r in records if r["object"] != obj and r["experiment"] > 8],
                            "test": [r["trial_id"] for r in records if r["object"] == obj]})
    summary = {"source": "ARQ-CRISP/slip_detection_dataset_2021", "labels": LABELS,
               "channels": 54, "axis_order": "18 pins, within each pin [x shear, y shear, z normal]; native Hall readings, not calibrated Newtons",
               "documented_nominal_hz": 180, "timestamp_unit_source": "milliseconds", "timestamp_unit_export": "seconds",
               "total_trials": len(records), "total_frames": sum(r["frames"] for r in records),
               "total_duration_s": sum(r["duration_s"] for r in records), "label_counts": dict(counts),
               "onset_count": sum(r["onset_count"] for r in records),
               "dt_quantiles_s": np.quantile(intervals,[0,.01,.5,.99,1]).tolist(),
               "strictly_monotonic_all_trials": all(r["timestamp_strictly_increasing"] for r in records),
               "nonpositive_intervals": sum(r["nonpositive_intervals"] for r in records),
               "trials": records,
               "limitations": ["Only 3 objects and 90 independent trial units; frame count is not independent sample size.",
                               "Label 0 mixes no-contact and static stable contact; cannot reconstruct binary physical contact onset uniquely.",
                               "Slip annotations are manual. Exact temporal error and annotation protocol require original-paper verification.",
                               "tactile_changes_label is an input-derived heuristic and is deliberately never exported as ground truth.",
                               "Dataset supports observational slip detection and transition probes; it cannot establish closed-loop recovery or insertion success.",
                               "No temporal resampling, normalization, smoothing, or train-test fitted transform was applied.",
                               "No arbitrary class balancing: natural prevalence is retained; any balanced training must keep original evaluation prevalence."]}
    dump(ROOT / "arq2021_audit.json", summary)
    dump(ROOT / "splits.json", {"held_trial":split, "leave_one_object_out":leave_object})
    with (ROOT / "trials.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [k for k in records[0] if k != "label_counts"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in records)
    print(json.dumps({k:v for k,v in summary.items() if k not in ["trials","limitations","labels"]}), flush=True)

def audit_flexitac(cache_path=None):
    import pyarrow.parquet as pq
    old = Path(cache_path) if cache_path else Path(r"C:\Users\mingzhe Liu\Documents\Codex\2026-10-06\https-github-com-marco6-3-icra2028\work\leflexitac")
    info = json.loads((ROOT / "sources" / "flexitac_meta_info.json").read_text(encoding="utf-8"))
    api = json.loads((ROOT / "sources" / "flexitac_current_api.json").read_text(encoding="utf-8"))
    prior_revision = (old / "revision.txt").read_text(encoding="utf-8").strip()
    cached_files = []
    columns = set()
    total_rows = 0
    for path in sorted(old.glob("*.parquet")):
        schema = pq.read_schema(path)
        meta = pq.read_metadata(path)
        columns.update(schema.names)
        total_rows += meta.num_rows
        cached_files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "rows": meta.num_rows, "columns": schema.names})
    cached = np.load(old / "all_data.npz")
    t = cached["timestamp"]
    ep = cached["episode_index"]
    dt = np.diff(t)[np.diff(ep) == 0]
    result = {"dataset": "https://huggingface.co/datasets/Tna001/tactile_test_tube_pyflexitac", "license_current_card": api.get("cardData",{}).get("license"),
              "current_revision": api["sha"], "cached_revision": prior_revision, "revisions_match": api["sha"] == prior_revision,
              "metadata_episodes": info["total_episodes"], "metadata_frames": info["total_frames"], "metadata_hz":info["fps"],
              "metadata_features": list(info["features"]), "cached_columns_union": sorted(columns), "cached_total_rows": total_rows,
              "cached_episodes_count": len(np.unique(ep)), "cached_dt_quantiles_s": np.quantile(dt,[0,.5,1]).tolist(),
              "cached_files": cached_files,
              "independent_contact_or_slip_label_present_in_schema": False,
              "decision": "Not eligible as independent contact/slip onset or success validation without new independently sourced annotations.",
              "reason": ["Published features and actual cached parquet schemas contain tactile, action, state, timestamps and indexes only; no slip, contact, force, failure or success labels.",
                         "Thresholding the tactile input to create ground truth measures recovery of a chosen rule; it cannot validate an independent physical event.",
                         "Future joint-action prediction is possible but changes the event-detection question; does not substitute for slip/success labels.",
                         "30 Hz gives about 3 frames per 100 ms; metadata do not establish native tactile response latency.",
                         "The cached prior project is read-only; this audit creates new artifacts only in the present task."]}
    dump(ROOT / "flexitac_label_audit.json", result)
    print(json.dumps({k: result[k] for k in ["current_revision","cached_revision","revisions_match","cached_total_rows","cached_episodes_count","decision"]}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-flexitac", action="store_true", help="Export ARQ data only; preserve archived FlexiTac audit without requiring its separate cache")
    parser.add_argument("--flexitac-cache", type=Path, help="Directory containing the optional cached FlexiTac parquet files, revision.txt and all_data.npz")
    args = parser.parse_args()
    export_arq()
    if not args.skip_flexitac:
        audit_flexitac(args.flexitac_cache)

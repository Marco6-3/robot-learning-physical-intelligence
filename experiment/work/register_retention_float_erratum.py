"""Record the authorized validation-tolerance correction; no data rewriting."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
old = ROOT / "work/experiment/analyze_retention_curriculum_before_fp_tolerance.py"
new = ROOT / "work/experiment/analyze_retention_curriculum.py"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
old_text, new_text = old.read_text(encoding="utf-8"), new.read_text(encoding="utf-8")
assert new_text == old_text.replace('np.isfinite(row[m]) and 0 <= row[m] <= 1',
                                    'np.isfinite(row[m]) and -1e-12 <= row[m] <= 1+1e-12')
affected = []
for path in sorted((ROOT / "work/results/retention_curriculum").glob("**/result.json")):
    result = json.loads(path.read_text(encoding="utf-8"))
    for row in result["training"]["trace"]:
        for metric in ("val_ap", "val_query_ap"):
            value = row[metric]
            if not 0 <= value <= 1:
                affected.append({"result_path": str(path.relative_to(ROOT)), "step": row["step"],
                                 "metric": metric, "raw_value": value, "upper_overshoot": max(0, value-1),
                                 "lower_undershoot": max(0, -value)})
report = {"corrected_at_utc": datetime.now(timezone.utc).isoformat(),
          "authorization": "Root explicitly authorized only floating-point validity tolerance repair and archival of old/new hashes; no user reconfirmation required.",
          "original_analysis_script": str(old.relative_to(ROOT)), "original_sha256": digest(old),
          "corrected_analysis_script": str(new.relative_to(ROOT)), "corrected_sha256": digest(new),
          "change": "Only validation AP legal-range check changed from [0,1] to [-1e-12,1+1e-12].",
          "tolerance": 1e-12, "affected_entries_observed_at_correction": affected,
          "max_observed_upper_overshoot": max((row["upper_overshoot"] for row in affected), default=0),
          "max_observed_lower_undershoot": max((row["lower_undershoot"] for row in affected), default=0),
          "training_still_running_at_correction": True,
          "no_changes_to": ["raw data", "saved metric values", "predictions", "checkpoint selection", "learning-rate selection",
                            "bootstrap", "contrasts", "confidence levels", "training code", "models", "frozen training protocol", "original statistics registration"],
          "clipping_applied": False,
          "source_diff": list(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(),
                                                 fromfile=old.name, tofile=new.name, lineterm=""))}
(ROOT / "outputs/记忆训练路径_浮点校验勘误.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"old_sha256": report["original_sha256"], "new_sha256": report["corrected_sha256"],
                  "affected_entries": len(affected), "max_overshoot": report["max_observed_upper_overshoot"]}))

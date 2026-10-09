"""Read-only audit of frozen hashes and completed retention runs during training."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work" / "experiment"))
from analyze_retention_curriculum import validate_trace

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

seal = read(ROOT / "outputs" / "补充协议_记忆训练路径_登记.json")
stats = read(ROOT / "outputs" / "记忆训练路径_统计细化登记.json")
erratum_path = ROOT / "outputs" / "记忆训练路径_浮点校验勘误.json"
erratum = read(erratum_path) if erratum_path.exists() else None
analysis_sha = stats["analysis_script_sha256"]
if erratum is not None:
    assert erratum["original_sha256"] == analysis_sha
    original_copy = ROOT / erratum["original_analysis_script"]
    assert hashlib.sha256(original_copy.read_bytes()).hexdigest() == analysis_sha
    analysis_sha = erratum["corrected_sha256"]
expected = {
    ROOT / "outputs" / "补充协议_记忆训练路径.md": seal["protocol_sha256"],
    ROOT / "work/experiment/retention_curriculum.py": seal["script_sha256"],
    ROOT / "work/experiment/models.py": seal["models_sha256"],
    ROOT / "work/experiment/common.py": seal["common_sha256"],
    ROOT / "work/experiment/analyze_retention_curriculum.py": analysis_sha,
}
hashes = []
for path, expected_hash in expected.items():
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_hash:
        raise RuntimeError(f"Frozen hash changed: {path}")
    hashes.append({"path": str(path.relative_to(ROOT)), "sha256": actual, "matches_registration": True})
counts = {"tuning": 0, "final": 0}
checked = []
for path in sorted((ROOT / "work/results/retention_curriculum").glob("**/result.json")):
    result = read(path)
    validate_trace(result)
    counts[result["phase"]] += 1
    checked.append({"path": str(path.relative_to(ROOT)), "phase": result["phase"],
                    "model": result["model"], "arm": result["arm"], "seed": result["seed"],
                    "optimizer_step2400": True, "all24_validation_points": True})
report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "hashes": hashes,
          "counts": counts, "checked_runs": checked,
          "scope": "Frozen file hashes and existing completion metadata only; no model execution or interim hypothesis analysis."}
(ROOT / "work/retention_integrity_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"hashes_match": len(hashes), "completed": counts, "all_completed_traces_valid": True}))

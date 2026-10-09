"""Explicit factual timing correction; preserves the initial registration."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
WORK = ROOT / "work" / "experiment"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

original = OUT / "补充协议_单状态记忆_登记_初版.json"
seal = json.loads(original.read_text(encoding="utf-8"))
old_hash = seal["protocol_sha256"]
report_path = WORK / "cue_memory_precheck.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
report["status"] = "all generator checks passed; no model training performed by precheck"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
protocol = OUT / "补充协议_单状态记忆.md"
seal.update({
    "timing_correction_at_utc": datetime.now(timezone.utc).isoformat(),
    "initial_registration_sha256": digest(original),
    "initial_protocol_sha256": old_hash,
    "protocol_sha256": digest(protocol),
    "generator_and_run_script_sha256": digest(WORK / "cue_memory.py"),
    "precheck_sha256": digest(report_path),
    "results_visible_when_amendment_was_requested": "Root specified v2 before inspecting target-model rankings; root had seen only v1 MLP outputs at the design request.",
    "results_visible_before_file_freeze": "After the detailed v2 design request but before file completion, root inspected partial v1 short-lag GRU seed4 and Mamba seeds0/1 outputs.",
    "timing_correction": "The initial wording did not distinguish design-request timing from file-freeze timing. Initial protocol, registration and hash are preserved. V2 has not trained; no test, hyperparameter, dataset size or planned run count changed on seeing these values. This is a separately specified mechanism supplement, not part of the original v1 preregistration.",
    "training_design_changed": False,
    "planned_total_training_runs": 112,
    "training_launched": False,
})
(OUT / "补充协议_单状态记忆_登记.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2), encoding="utf-8")
protocol.with_suffix(".sha256").write_text(digest(protocol) + "  " + protocol.name + "\n", encoding="utf-8")
print(json.dumps(seal, ensure_ascii=False, indent=2))

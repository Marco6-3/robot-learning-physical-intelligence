"""Acquire public tactile data at immutable revisions; never modify old caches."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from datetime import datetime, timezone
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
SOURCES = ROOT / "sources"
REPO = "ARQ-CRISP/slip_detection_dataset_2021"
HF_REPO = "Tna001/tactile_test_tube_pyflexitac"
FILES = ["slipDataset_brush_tactile.h5", "slipDataset_screwDriver_tactile.h5", "slipDataset_SpoolSolder_tactile.h5"]
LIMIT = 350_000_000

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def get(url, **kwargs):
    response = requests.get(url, timeout=(15, 90), headers={"User-Agent": "Tactile-research-provenance-audit"}, **kwargs)
    response.raise_for_status()
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)
    ref_path = SOURCES / "github_revision.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
    else:
        ref = get(f"https://api.github.com/repos/{REPO}/commits/main").json()
        save_json(ref_path, ref)
    revision = ref["sha"]
    manifest = {"repository": f"https://github.com/{REPO}", "revision": revision,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(), "license": "BSD-3-Clause", "files": []}
    for path in ["README.md", "LICENSE", "data/README.md"]:
        target = SOURCES / ("arq2021_" + path.replace("/", "_"))
        url = f"https://raw.githubusercontent.com/{REPO}/{revision}/{path}"
        if not target.exists():
            target.write_bytes(get(url).content)
        manifest["files"].append({"path": str(target.relative_to(ROOT)), "url": url, "bytes": target.stat().st_size, "sha256": sha256(target)})
    data_records = []
    for name in FILES:
        url = f"https://raw.githubusercontent.com/{REPO}/{revision}/data/{name}"
        pointer_path = SOURCES / (name + ".lfs.txt")
        if not pointer_path.exists():
            pointer_path.write_bytes(get(url).content)
        pointer = pointer_path.read_text(encoding="utf-8")
        oid = re.search(r"oid sha256:([a-f0-9]{64})", pointer).group(1)
        size = int(re.search(r"size (\d+)", pointer).group(1))
        data_records.append({"filename": name, "bytes": size, "sha256_expected_lfs": oid,
                             "url": f"https://media.githubusercontent.com/media/{REPO}/{revision}/data/{name}"})
    if sum(item["bytes"] for item in data_records) > LIMIT:
        raise RuntimeError("The official data exceed the 350 MB acquisition budget")
    for rec in data_records:
        destination = RAW / rec["filename"]
        if args.download and not destination.exists():
            partial = destination.with_suffix(".h5.part")
            print(f"Downloading {rec['filename']}: {rec['bytes']} bytes", flush=True)
            response = get(rec["url"], stream=True)
            n = 0
            last_report = time.monotonic()
            with partial.open("wb") as output:
                for chunk in response.iter_content(2**20):
                    n += len(chunk)
                    if n > rec["bytes"]:
                        raise RuntimeError("Download exceeds pinned LFS size")
                    output.write(chunk)
                    if time.monotonic() - last_report > 5:
                        print(f"  {n / 1e6:.1f} / {rec['bytes'] / 1e6:.1f} MB", flush=True)
                        last_report = time.monotonic()
            if n != rec["bytes"] or sha256(partial) != rec["sha256_expected_lfs"]:
                raise RuntimeError("Pinned LFS size or SHA-256 verification failed")
            partial.replace(destination)
        if destination.exists():
            rec["sha256_actual"] = sha256(destination)
            rec["verified"] = destination.stat().st_size == rec["bytes"] and rec["sha256_actual"] == rec["sha256_expected_lfs"]
            if not rec["verified"]:
                raise RuntimeError("Existing data do not match pinned LFS digest")
        manifest["files"].append(rec)
        save_json(SOURCES / "source_manifest.json", manifest)
    hf_api_url = f"https://huggingface.co/api/datasets/{HF_REPO}"
    # Reproduction uses the archived revision. A new run without archived
    # metadata resolves current once and persists that choice.
    hf_path = SOURCES / "flexitac_current_api.json"
    hf = json.loads(hf_path.read_text(encoding="utf-8")) if hf_path.exists() else get(hf_api_url).json()
    save_json(hf_path, hf)
    for name in ["README.md", "meta/info.json"]:
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{hf['sha']}/{name}"
        target = SOURCES / ("flexitac_" + name.replace("/", "_"))
        target.write_bytes(get(url).content)
        manifest["files"].append({"path": str(target.relative_to(ROOT)), "url": url, "bytes": target.stat().st_size, "sha256": sha256(target)})
    manifest["flexitac_revision"] = hf["sha"]
    save_json(SOURCES / "source_manifest.json", manifest)
    print(json.dumps({"revision": revision, "data_bytes": sum(r["bytes"] for r in data_records), "downloaded": args.download, "flexitac_revision": hf["sha"]}), flush=True)

if __name__ == "__main__":
    main()

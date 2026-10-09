"""Build a checked, self-contained reproduction bundle after experiments finish.

Default invocation only lists the proposed copy. --build performs a new snapshot.
Never overwrites an existing bundle, never recursively copies the outputs bundle,
and never packages the downloaded raw HDF5 or locally installed dependencies.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"
TREES = ["experiment", "real_data", "frozen", "references", "results"]
SKIP_COMPONENTS = {"__pycache__", "deps", ".git", "raw"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".part", ".tmp"}

def below(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()

def inventory(destination):
    items = []
    for tree in TREES:
        base = WORK / tree
        if not base.is_dir():
            raise FileNotFoundError(base)
        for source in sorted(base.rglob("*")):
            rel = source.relative_to(base)
            if any(p in SKIP_COMPONENTS for p in rel.parts) or source.suffix in SKIP_SUFFIXES:
                continue
            if source.is_symlink():
                raise RuntimeError(f"Symlink must be reviewed explicitly: {source}")
            if source.is_file():
                items.append((source, Path("work") / tree / rel))
    # Protocol notes, audit scripts/results and logs at the work root are evidence.
    for source in sorted(WORK.iterdir()):
        if source.is_file() and source.suffix in {".md", ".log", ".py", ".json"}:
            items.append((source, Path("work") / source.name))
    items.append((OUTPUTS / "数据与复现说明.md", Path("README.md")))
    requirements = WORK / "requirements_reproduction.txt"
    if not requirements.is_file():
        requirements = ROOT / "requirements.txt"
    items.append((requirements, Path("requirements.txt")))
    # Only user deliverables, never the bundle, earlier bundles, or package zips.
    for source in sorted(OUTPUTS.rglob("*")):
        if below(source, destination):
            continue
        rel = source.relative_to(OUTPUTS)
        if any(p.startswith("复现实验") or p.startswith(".复现实验") for p in rel.parts):
            continue
        if source.is_symlink():
            raise RuntimeError(f"Symlink must be reviewed explicitly: {source}")
        if source.is_file() and source.suffix.lower() not in {".zip", ".7z", ".tar", ".gz"}:
            items.append((source, Path("outputs") / rel))
    relative = [str(target) for _,target in items]
    if len(relative) != len(set(relative)):
        raise RuntimeError("Duplicate bundle destination")
    return items

def stamp(items):
    return {str(src.relative_to(ROOT)): (src.stat().st_size, src.stat().st_mtime_ns) for src,_ in items}

def verify(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    bad = []
    for rec in manifest["files"]:
        path = bundle / rec["path"]
        if not below(path,bundle) or not path.is_file():
            bad.append({"path":rec["path"],"problem":"missing or outside bundle"})
        elif path.stat().st_size != rec["bytes"] or digest(path) != rec["sha256"]:
            bad.append({"path":rec["path"],"problem":"size or SHA256 mismatch"})
    listed = {r["path"] for r in manifest["files"]} | {"manifest.json", "SHA256SUMS.txt"}
    extras = [str(p.relative_to(bundle).as_posix()) for p in bundle.rglob("*") if p.is_file() and p.relative_to(bundle).as_posix() not in listed]
    checksums = (bundle / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    expected = [f"{r['sha256']}  {r['path']}" for r in manifest["files"]]
    expected.append(f"{digest(bundle/'manifest.json')}  manifest.json")
    if checksums != expected:
        bad.append({"path":"SHA256SUMS.txt","problem":"checksum inventory differs"})
    result = {"files_verified":len(manifest["files"]),"mismatches":bad,"unexpected_files":extras,"ok":not bad and not extras}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result["ok"]:
        raise RuntimeError("Bundle verification failed")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build",action="store_true")
    parser.add_argument("--verify",action="store_true")
    default_output=ROOT if (ROOT/"manifest.json").is_file() else OUTPUTS/"复现实验"
    parser.add_argument("--output",type=Path,default=default_output)
    args=parser.parse_args()
    destination=args.output.resolve()
    verify_self=args.verify and destination == ROOT.resolve() and (ROOT/"manifest.json").is_file()
    if not verify_self and (not below(destination,OUTPUTS) or destination == OUTPUTS.resolve()):
        raise RuntimeError("Bundle destination must be a child directory of this task's outputs")
    if args.verify:
        verify(destination)
        return
    items=inventory(destination)
    print(json.dumps({"destination":str(destination),"planned_files":len(items),
                      "planned_bytes":sum(src.stat().st_size for src,_ in items),
                      "build_requested":args.build,"excluded":["raw HDF5","deps","__pycache__","existing reproduction bundles","archive files"]},ensure_ascii=False,indent=2))
    if not args.build:
        return
    if destination.exists():
        raise RuntimeError("Destination already exists; choose a new output directory. Existing evidence is never overwritten.")
    now=datetime.now(timezone.utc)
    staging=(WORK/("package_staging_"+now.strftime("%Y%m%dT%H%M%S_%f"))).resolve()
    # Verify both absolute targets before the final recursive directory move.
    if not below(staging,WORK) or not below(destination,OUTPUTS):
        raise RuntimeError("Resolved staging/destination outside intended workspace")
    initial=stamp(items)
    staging.mkdir(parents=False,exist_ok=False)
    records=[]
    for source,relative in items:
        target=staging/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        before=(source.stat().st_size,source.stat().st_mtime_ns)
        shutil.copy2(source,target)
        after=(source.stat().st_size,source.stat().st_mtime_ns)
        if before != after:
            raise RuntimeError(f"Source changed during copy; wait for experiments to finish: {source}")
        checksum=digest(target)
        if checksum != digest(source):
            raise RuntimeError(f"Copy checksum failed: {source}")
        records.append({"path":relative.as_posix(),"bytes":target.stat().st_size,"sha256":checksum})
    if stamp(inventory(destination)) != initial:
        raise RuntimeError("Source inventory changed during packaging; wait for all work to finish. Staging retained for inspection.")
    records.sort(key=lambda r:r["path"])
    manifest={"created_utc":now.isoformat(),"purpose":"Reproduction snapshot; retained results are observations, not guaranteed rerun values",
              "layout":"Run commands from bundle root; work/ paths and nested outputs/ are preserved",
              "excluded":["raw HDF5 can be reacquired with pinned hashes","installed dependency binaries","Python bytecode","existing bundles"],
              "license_notice":"Third-party code/data retain source licenses copied under work/references and work/real_data/sources",
              "files":records}
    (staging/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=[f"{r['sha256']}  {r['path']}" for r in records]
    lines.append(f"{digest(staging/'manifest.json')}  manifest.json")
    (staging/"SHA256SUMS.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    verify(staging)
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Only the new, verified snapshot moves; original work and outputs remain intact.
    staging.rename(destination)
    print(json.dumps({"bundle_ready":str(destination),"files":len(records),"bytes":sum(r['bytes']for r in records)},ensure_ascii=False))

if __name__=="__main__":
    main()

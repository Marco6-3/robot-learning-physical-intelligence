"""Download pinned upstream Mamba sources and license for implementation audit."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
HEADERS = {"User-Agent": "research-reproducibility-audit"}


def fetch(url):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


commit = json.loads(fetch("https://api.github.com/repos/state-spaces/mamba/commits/main"))["sha"]
files = ["mamba_ssm/modules/mamba_simple.py", "mamba_ssm/ops/selective_scan_interface.py", "LICENSE"]
manifest = {"repository": "https://github.com/state-spaces/mamba", "commit": commit, "files": []}
for filename in files:
    url = f"https://raw.githubusercontent.com/state-spaces/mamba/{commit}/{filename}"
    content = fetch(url)
    destination = ROOT / ("mamba_LICENSE" if filename == "LICENSE" else Path(filename).name)
    destination.write_bytes(content)
    manifest["files"].append({"upstream_path": filename, "url": url, "local_path": destination.name, "sha256": hashlib.sha256(content).hexdigest()})
(ROOT / "mamba_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))

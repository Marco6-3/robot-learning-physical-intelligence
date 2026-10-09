from pathlib import Path
import datetime, json, platform, shutil
from common import sha256, save_json
import torch, numpy, sklearn

folder=Path('work/frozen');folder.mkdir(exist_ok=True)
sources=list(Path('work/experiment').glob('*.py'))+[Path('outputs/实验协议.md')]
items=[]
for p in sources:
    dst=folder/p.name
    if dst.exists(): raise RuntimeError(f'Frozen file already exists: {dst}')
    shutil.copy2(p,dst);items.append({'path':str(p),'sha256':sha256(p)})
save_json(folder/'freeze.json',{'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'client_date':'2026-10-09','sources':items,'environment':{'python':platform.python_version(),
    'torch':torch.__version__,'numpy':numpy.__version__,'sklearn':sklearn.__version__,
    'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform()}})
print('Protocol and execution source snapshot frozen.')

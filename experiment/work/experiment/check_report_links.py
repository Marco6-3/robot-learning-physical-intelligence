"""Check the final evidence report's local links and numeric source coverage."""
from pathlib import Path
import re
import json

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'outputs'/'实验结论.md'
text=path.read_text(encoding='utf-8')
assert '\ufffd' not in text and '??' not in text
assert 'TODO' not in text and '待填写' not in text
checked=[]
for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
    if target.startswith(('https://','http://','#')):
        continue
    target=target.strip('<>')
    resolved=(path.parent/target).resolve()
    assert resolved.is_file(),f'Broken report link: {target}'
    checked.append(target)
assert '582' in text and '430' in text and '152' in text
print(json.dumps({'utf8':True,'local_report_links_checked':checked,'training_count_text':582},ensure_ascii=False))

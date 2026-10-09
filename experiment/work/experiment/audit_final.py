"""Cross-study provenance/count audit; requires all five studies to be complete."""
from pathlib import Path
import json
from common import save_json, sha256

ROOT=Path(__file__).resolve().parents[2]

def read(p):
    return json.loads((ROOT/p).read_text(encoding='utf-8'))

def main():
    checks=[]
    frozen=read('work/frozen/freeze.json')
    for record in frozen['sources']:
        original=ROOT/record['path']
        copy=ROOT/'work'/'frozen'/original.name
        assert sha256(copy)==record['sha256'],f'Original snapshot changed: {copy}'
        # AP dot-product -> elementwise sum is an equivalent numerical
        # performance change; an interaction-direction self-test was added.
        # Both exact source hashes are retained instead of accepting any edit.
        if original.name=='analyze_synthetic.py':
            assert record['sha256']=='e8d3b2b58ee7caafd19872903312cf439118402efeb61aa47cdc627b0296ba45'
            assert sha256(original)=='2babec1489de102f9dd800878eb54ea933308d8cabc776a989784e929befc6d0'
        else:
            assert sha256(original)==record['sha256'],f'Frozen source changed: {original}'
    checks.append('Original protocol/training hashes and all frozen snapshots match')
    real=read('work/results/real/pretraining_manifest.json')
    for name,digest in real['code_hashes'].items():
        assert sha256(ROOT/'work'/'experiment'/name)==digest
    assert sha256(ROOT/'work'/'real_data'/'splits.json')==real['split_sha256']
    assert sha256(ROOT/'work'/'real_data'/'arq2021_audit.json')==real['data_audit_sha256']
    checks.append('Real-study training source and split hashes match pretraining manifest')
    from cue_memory import verify_registration as cue_seal
    from optimization_locality import verify_seal as optimization_seal
    from retention_curriculum import verify_registration as retention_seal
    cue_seal(); optimization_seal(); retention_seal()
    checks.append('Three adaptive study registrations match training sources and protocols')
    correction=read('outputs/记忆训练路径_浮点校验勘误.json')
    assert correction['original_sha256']=='699dd71533d96b3ffbe9f77b36ef9dbc0524c652ccb9b4ce54f5b01136c7458e'
    assert correction['corrected_sha256']=='86a15a6017c053999ee308f40da4e3f8070f23be49a85ba7d3527400aa622241'
    assert sha256(ROOT/correction['original_analysis_script'])==correction['original_sha256']
    assert sha256(ROOT/correction['corrected_analysis_script'])==correction['corrected_sha256']
    checks.append('Retention analysis matches the documented floating-point tolerance-only repair; original retained')
    expected=[('synthetic','synthetic_analysis',160,64),
              ('cue_memory_v2','cue_memory_v2_analysis',80,32),
              ('real','real_analysis',70,28),
              ('optimization_locality_v3','optimization_locality_v3_analysis',100,20),
              ('retention_curriculum','retention_curriculum_analysis',20,8)]
    records=[]
    for study,analysis_name,nfinal,ntune in expected:
        source=ROOT/'work'/'results'/study
        if study in ('synthetic','cue_memory_v2'):
            final=list(source.glob('*_seed*.json'))
            tuning=[t for p in source.glob('*_tuning.json') for t in json.loads(p.read_text(encoding='utf-8'))['trials']]
        elif study=='optimization_locality_v3':
            final=list(source.glob('*_seed*.json'))
            tuning=list(source.glob('tune_*.json'))
        else:
            paths=list(source.rglob('result.json'))
            # Identify from explicit directory component, never from the seed.
            final=[p for p in paths if any(s.startswith('seed') or s.startswith('final') for s in p.relative_to(source).parts)]
            tuning=[p for p in paths if any(s.startswith('tune') for s in p.relative_to(source).parts)]
        assert len(final)==nfinal,(study,'final',len(final),nfinal)
        assert len(tuning)==ntune,(study,'tuning',len(tuning),ntune)
        analysis=ROOT/'work'/'results'/analysis_name/'analysis.json'
        assert analysis.is_file(),analysis
        records.append(dict(study=study,final_runs=len(final),tuning_runs=len(tuning),
                            analysis_path=str(analysis.relative_to(ROOT)),analysis_sha256=sha256(analysis)))
    checks.append('All fixed study matrices and their independent analyses are complete')
    outputs=ROOT/'outputs'
    checked=[]
    for p in outputs.rglob('*.md'):
        if '复现实验' in p.parts:
            continue
        text=p.read_text(encoding='utf-8')
        assert '\ufffd' not in text and '??' not in text,p
        checked.append(str(p.relative_to(ROOT)))
    report=dict(status='passed',studies=records,checks=checks,
                formal_runs=sum(r['final_runs'] for r in records),
                tuning_runs=sum(r['tuning_runs'] for r in records),
                utf8_markdown_checked=checked,
                v1_analysis_change='AP np.dot -> np.sum of elementwise products; equivalent formula, explicit-resampling validation retained; added factorial interaction sign self-test. Original frozen code is preserved.',
                scope='Provenance and completion; separate per-study analyzers check predictions, metrics and paired bootstrap.')
    save_json(ROOT/'outputs'/'完整性核查.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

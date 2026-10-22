"""Audit INCLUDED records only. Never instantiate a network or run training."""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def audit(with_frozen_data=False):
    summary=json.loads((ROOT/'results/life_choice/summary.json').read_text())
    manifest=json.loads((ROOT/'data/life_choice/manifest.json').read_text())
    for split,entry in manifest['splits'].items():
        assert hashlib.sha256((ROOT/f'data/life_choice/{split}.npz').read_bytes()).hexdigest()==entry['file_sha256']
    arrays=None
    if with_frozen_data:
        import numpy as np
        with np.load(ROOT/'data/life_choice/test.npz',allow_pickle=False) as d:
            arrays={k:d[k].copy() for k in ('seeds','variants','utilities','mask')}
    reports=[]
    for s in summary['seeds']:
        seed=s['seed']
        rows=[json.loads(x) for x in (ROOT/f'results/life_choice/action_audit/seed{seed}-decisions.jsonl').read_text().splitlines()]
        assert len(rows)==2000
        counts={name:sum(r['variant']==name for r in rows) for name in ('daily','counterfactual','unknown')}
        assert counts=={'daily':1600,'counterfactual':300,'unknown':100}
        assert sum(r['execution']['arrived'] for r in rows)==s['action_execution']['arrived']==2000
        assert sum(r['execution']['completed'] for r in rows)==s['action_execution']['completed']==2000
        for row in rows:
            p=row['probabilities']
            assert len(p)==12 and all(math.isfinite(x) and 0<=x<=1 for x in p)
            assert abs(sum(p)-1)<1e-5
            assert row['selected_index']==max(range(12),key=lambda i:p[i])
        report={'seed':seed,'decision_rows':len(rows),'variants':counts,'action_records_verified':True}
        if arrays is not None:
            import numpy as np
            assert np.array_equal(arrays['seeds'],[r['episode_seed'] for r in rows])
            selected=np.array([r['selected_index'] for r in rows])
            assert arrays['mask'][np.arange(2000),selected].all()
            best=np.where(arrays['mask'],arrays['utilities'],-1e9).max(1)
            regret=best-arrays['utilities'][np.arange(2000),selected]
            correct=regret<=1e-5
            daily=arrays['variants']==0
            n=int(correct[daily].sum())
            assert n==s['daily']['successes']
            assert int((regret>1e-5).sum())==s['failure_count_positive_teacher_regret']
            report.update(daily_correct=n,daily_n=1600,frozen_utility_scoring_verified=True)
        else:report['frozen_utility_scoring_verified']=False
        reports.append(report)
    return {'mode':'existing-record audit, no model execution','numpy_scoring_requested':with_frozen_data,'reports':reports}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--with-frozen-data',action='store_true')
    print(json.dumps(audit(p.parse_args().with_frozen_data),indent=2))

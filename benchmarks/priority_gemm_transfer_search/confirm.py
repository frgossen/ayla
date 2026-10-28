"""Freeze one candidate per shape, then apply independent replica gates."""
import argparse
import json
from pathlib import Path
import statistics as st
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','analyze']);a=p.parse_args()
ps=json.loads((HERE/'priorities.json').read_text())
if a.action=='prepare':
    leaders=json.loads((HERE/'leaders.json').read_text());selected=[];jobs=[]
    for pr in ps:
        rows=[r for r in leaders if r['target']==pr['target'] and r['conservative_gain']>=1.007]
        if not rows:continue
        r=max(rows,key=lambda r:r['conservative_gain']);selected.append(r)
        for gpu in range(8):
            old=dict(kind='control',label='incumbent',configuration=pr['configuration'])
            new=dict(kind='candidate',label='fixed-candidate',configuration=r['configuration'],
                     donor=r['entry'].get('donor'),screen_label=r['entry']['label'])
            # Both orders occur on four independent GPUs. The bracketed control
            # establishes whether Torch changes during the candidate measurement.
            entries=[old,new,dict(old,label='incumbent-repeat')] if gpu%2==0 else [new,old,dict(old,label='incumbent-repeat')]
            jobs.append(dict(target=pr['target'],replica_gpu=gpu,entries=entries))
    (HERE/'fixed_candidates.json').write_text(json.dumps(selected,indent=2)+'\n')
    (HERE/'confirmation_jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print('Fixed candidates',len(selected),'jobs',len(jobs))
else:
    candidates=json.loads((HERE/'fixed_candidates.json').read_text());result=[]
    for c in candidates:
        replicas=[]
        for path in (HERE/'confirmation').glob('[0-9][0-9][0-9][0-9].json'):
            rows=json.loads(path.read_text())
            if not rows or rows[0]['target']!=c['target']:continue
            if len(rows)!=3 or any(r['status']!='ok' for r in rows):continue
            old=[r for r in rows if r['entry']['kind']=='control'];new=next(r for r in rows if r['entry']['kind']=='candidate')
            old_ms=st.median(r['ayla_ms'] for r in old);old_pct=st.median(r['torch_pct'] for r in old)
            old_torch=st.median(r['torch_ms'] for r in old)
            replicas.append(dict(gpu=new['gpu_uuid'],old_pct=old_pct,new_pct=new['torch_pct'],
                old_ms=old_ms,new_ms=new['ayla_ms'],gain=min(old_ms/new['ayla_ms'],new['torch_pct']/old_pct),
                normalized_gain=new['torch_pct']/old_pct,raw_gain=old_ms/new['ayla_ms'],
                torch_ratio=new['torch_ms']/old_torch,
                control_drift=max(r['torch_ms'] for r in old)/min(r['torch_ms'] for r in old),
                executable_changed=all(new['executable_fingerprint']!=r['executable_fingerprint'] for r in old),
                source_file=str(path.relative_to(HERE))))
        complete=len(replicas)==8 and len({r['gpu'] for r in replicas})==8
        med=lambda field:st.median(r[field] for r in replicas) if replicas else None
        positive=sum(r['gain']>1 for r in replicas)
        stable=sum(abs(r['torch_ratio']-1)<=0.05 and r['control_drift']<=1.05 for r in replicas)
        accepted=(complete and med('gain')>=1.01 and med('raw_gain')>=1.01 and
                  positive>=6 and stable>=6 and abs(med('torch_ratio')-1)<=0.02 and
                  all(r['executable_changed'] for r in replicas))
        pr=next(p for p in ps if p['target']==c['target'])
        result.append(dict(target=c['target'],rank=pr['rank'],op=pr['op'],dtype=pr['dtype'],
            m=pr['m'],n=pr['n'],k=pr['k'],status='updated' if accepted else 'retained',
            configuration=c['configuration'],donor=c['entry'].get('donor'),screen_label=c['entry']['label'],
            old_pct=med('old_pct'),candidate_pct=med('new_pct'),old_ms=med('old_ms'),candidate_ms=med('new_ms'),
            conservative_gain=med('gain'),raw_gain=med('raw_gain'),torch_reference_ratio=med('torch_ratio'),
            replicas_complete=complete,positive_replicas=positive,stable_replicas=stable,replicas=replicas))
    (HERE/'verified.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in result:print(r['rank'],r['target'],r['status'],r['old_pct'],r['candidate_pct'],r['conservative_gain'],r['positive_replicas'],r['stable_replicas'])

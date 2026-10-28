"""Choose a small, diverse set of measured donor geometries for weak shapes."""
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from ayla.gemm_jit import _seed, _DEFAULTS

priorities=json.loads((HERE/'priorities.json').read_text())
donors=json.loads((HERE/'donors.json').read_text())
baselines={r['target']:r for path in (HERE/'baseline').glob('[0-9][0-9][0-9][0-9].json')
           for r in json.loads(path.read_text()) if r['status']=='ok'}
assert len(baselines)==35, len(baselines)
SUPPORTED={'output_cta_count','output_tile_fast_axis','unroll_output_tiles',
 'output_m_tile','output_n_tile','warpgroup_n_tile','warpgroup_m_tile','warpgroup_fast_axis',
 'output_slices','output_store_buffers','output_tma_inner_size','output_store',
 'operand_k_tile','operand_stages','cyclic_operand_pipeline','output_tiles_per_cta',
 'output_n_tiles_per_cta','lhs_l2_cache_hint','rhs_l2_cache_hint','dst_l2_cache_hint',
 'lhs_l2_promotion','rhs_l2_promotion','ptx_optimization_level','reschedule_async',
 'swap_output_grid_axes','multicast','cluster_m_ctas','cluster_n_ctas',
 'transpose_output','pipeline_output_waves','rhs_tma_load_slices'}

def normalize(d):
    c=dict(d['configuration'])
    if any(value not in (None,False,1,'n') for key,value in c.items()
           if key not in SUPPORTED and key!='name'):
        return None
    c={key:value for key,value in c.items() if key in SUPPORTED}
    if 'output_n_tiles_per_cta' not in c and c.get('output_cta_count') is None:
        if c.get('output_tile_fast_axis')=='n':
            c['output_n_tiles_per_cta']=c.get('output_tiles_per_cta',1)
            c['output_tiles_per_cta']=1
    return {**_DEFAULTS,**c}

def geometry(c):
    return tuple(c.get(k) for k in ('output_m_tile','output_n_tile','warpgroup_n_tile',
       'operand_k_tile','operand_stages','output_tiles_per_cta','output_n_tiles_per_cta',
       'output_cta_count','unroll_output_tiles','cyclic_operand_pipeline',
       'output_slices','output_store_buffers','output_store','multicast','cluster_n_ctas','transpose_output'))

def distance(p,d):
    return 0.5*abs(math.log2(p['m']/d['m']))+2*abs(math.log2(p['n']/d['n']))+2*abs(math.log2(p['k']/d['k']))

for d in donors:d['normalized']=normalize(d)
jobs=[];audit=[]
for p in priorities:
    baseline=baselines[p['target']]
    # Include up to 100.5% to avoid classifying near-ties from one run as solved.
    if baseline['torch_pct']>100.5:
        continue
    cap=24 if p['rank']<=12 or p['rank']==19 else 18
    candidates=[];seen=set();geometries=set()
    incumbent=p['configuration']
    def add(c,label,donor=None,kind='transfer',ignore_transfer_limits=False):
        c={**_DEFAULTS,**c};key=json.dumps(c,sort_keys=True)
        if key==json.dumps(incumbent,sort_keys=True) or key in seen:return
        if c.get('unsafe_reuse_smem') or c.get('pipeline_output_waves'):return
        if c.get('transpose_output') and p['op']=='addmm':return
        try:
            seed=_seed(c,p['op'],p['m'],p['n'],p['k'],(1,p['m'],p['n'],p['k']),
                       exact=ignore_transfer_limits)
        except (AssertionError,KeyError,ValueError,TypeError):return
        if seed is None:return
        if not ignore_transfer_limits and geometry(c) in geometries:return
        seen.add(key);geometries.add(geometry(c))
        candidates.append(dict(kind=kind,label=label,configuration=c,donor=donor))
    pool=sorted([d for d in donors if d['normalized'] is not None],key=lambda d:
          (distance(p,d)+(0 if d['op']==p['op'] else 1)+(0 if d['dtype']=='bfloat16' else 0.5),
           -(d.get('donor_pct') or 100)))
    family=[d for d in pool if d['n']==p['n'] and d['k']==p['k'] and d['op']==p['op']]
    for d in family[:30]:
        if len(candidates)>=7:break
        add(d['normalized'],f"same-NK:{d['backend']}:{d['m']}x{d['n']}x{d['k']}:{d['dtype']}",d)
    # Transfer the successful 16384-M and newly installed 175360-M schedules
    # across the DINOv2 family; their geometry differs substantially.
    if p['m']>=16384 and p['k'] in (768,3072):
        for donor_rank in (4,5,7,8,9,10,24,30,32):
            d=next(x for x in priorities if x['rank']==donor_rank)
            if baselines[d['target']]['torch_pct']>=p['historical_pct'] or donor_rank in (4,7,9,10,32):
                add(d['configuration'],f"priority-family:r{donor_rank}",
                    {k:d[k] for k in ('target','m','n','k','op','dtype')})
            if len(candidates)>=13:break
    # Successful 8192-M wide output geometry also targets 16384-M shapes.
    for donor_rank in (12,14,26,1):
        if p['rank'] in (2,3,11,19,23):
            d=next(x for x in priorities if x['rank']==donor_rank)
            add(d['configuration'],f'priority-wide:r{donor_rank}',
                {k:d[k] for k in ('target','m','n','k','op','dtype')})
    # Reserve space for cross-dtype/cross-op donors and rolled-loop concepts.
    for d in pool:
        if len(candidates)>=cap-4:break
        add(d['normalized'],f"nearby:{d['backend']}:{d['m']}x{d['n']}x{d['k']}:{d['op']}:{d['dtype']}",d)
    if p['m']>=16384:
        for donor_rank in ((9,10) if p['k']<=1024 else (9,26)):
            d=next(x for x in priorities if x['rank']==donor_rank)
            c=dict(d['configuration']);c.update(unroll_output_tiles=False,pipeline_output_waves=False)
            add(c,f'rolled-transfer:r{donor_rank}',{k:d[k] for k in ('target','m','n','k')},'rolled_transfer')
    if p['rank'] in (2,3,11,19,23):
        # Cluster-2 operand reuse follows nearby successful measured clusters.
        for kt,stages in ((128,3),(64,4)):
            c=dict(incumbent);c.update(output_m_tile=128,output_n_tile=256,warpgroup_n_tile=128,
                operand_k_tile=kt,operand_stages=stages,output_cta_count=None,
                output_tiles_per_cta=1,output_n_tiles_per_cta=1,output_slices=2,
                multicast=True,cluster_n_ctas=2,cluster_m_ctas=1,unroll_output_tiles=True,
                lhs_l2_cache_hint='evict_first',rhs_l2_cache_hint='evict_last')
            add(c,f'cluster2-donor-concept:K{kt}:s{stages}',kind='cluster_concept',ignore_transfer_limits=True)
    for d in pool:
        if len(candidates)>=cap:break
        add(d['normalized'],f"diverse:{d['backend']}:{d['m']}x{d['n']}x{d['k']}:{d['dtype']}",d)
    entries=[]
    for i,c in enumerate(candidates[:cap]):
        if i%6==0:entries.append(dict(kind='control',label=f'incumbent-block{i//6}',configuration=incumbent))
        entries.append(c)
    entries.append(dict(kind='control',label='incumbent-end',configuration=incumbent))
    jobs.append(dict(target=p['target'],entries=entries))
    audit.append(dict(rank=p['rank'],target=p['target'],baseline_pct=baseline['torch_pct'],
                      candidates=len(candidates[:cap]),geometries=len({geometry(c['configuration']) for c in candidates[:cap]})))
(HERE/'transfer_jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
(HERE/'transfer_plan.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(targets=len(jobs),candidates=sum(a['candidates'] for a in audit),plan=audit),indent=2))

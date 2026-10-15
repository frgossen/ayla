"""Merge confirmed winners into the two existing branches; preserve other routes."""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--jit-table',action='store_true');a=p.parse_args()
root=a.root;sys.path.insert(0,str(root))
from ayla.gemm_heuristics import expand_gemm_heuristic,collapse_gemm_heuristic,select_gemm_schedules,_mapper
rows=[r for r in json.loads((HERE/'verified.json').read_text()) if r['status']=='updated']
assert rows
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(), 'Worktree has other edits'
backup=HERE/'integration'/root.name;backup.mkdir(parents=True,exist_ok=True)
path=root/'ayla/oss_gemm_heuristic.json';(backup/'heuristic_before.json').write_text(path.read_text())
before=expand_gemm_heuristic(json.loads(path.read_text()));after=copy.deepcopy(before)
keys={(r['op'],r['dtype'],1,r['m'],r['n'],r['k']) for r in rows}
old_exact={(e['operation'],e['dtype'],e['batch'],e['m'],e['n'],e['k']):e for e in before['exact']}
def key(e):return e['operation'],e['dtype'],e['batch'],e['m'],e['n'],e['k']
new_exact=[]
for r in rows:
    cfg=copy.deepcopy(r['configuration']);cfg['name']=f"confirmed_transfer_{r['op']}_m{r['m']}_n{r['n']}_k{r['k']}"
    e=dict(operation=r['op'],dtype=r['dtype'],batch=1,m=r['m'],n=r['n'],k=r['k'],
           seed=dict(configuration=cfg,source=dict(m=r['m'],n=r['n'],k=r['k']),
                     source_percentage_of_torch=r['candidate_pct']))
    new_exact.append(e)
    matches=[i for i,x in enumerate(after['exact']) if key(x)==key(e)]
    if matches:after['exact'][matches[0]]=e
    else:after['exact'].append(e)
after['source_shape_count']+=sum(key(e) not in old_exact for e in new_exact)

def split(feature,value,left,right):return dict(feature=feature,threshold=float(value),left=left,right=right)
def guards(es,fallback,features=('m','n','k')):
    if not features:
        assert len(es)==1
        return dict(leaf='confirmed_transfer:'+str(key(es[0])),seeds=[es[0]['seed']])
    feature,*rest=features;values=sorted({e[feature] for e in es})
    tree=fallback
    for value in reversed(values):
        inner=guards([e for e in es if e[feature]==value],fallback,tuple(rest))
        tree=split(feature,value-0.5,fallback,split(feature,value+0.5,inner,tree))
    return tree
for op in {r['op'] for r in rows}:
    es=[e for e in new_exact if e['operation']==op]
    for suffix in ('k128','k64'):
        group=f'{op}/bfloat16/{suffix}'
        fallback=after['groups'].get(group,dict(leaf=f'{op}_bfloat16_coverage_fallback',seeds=[dict(
            configuration=after['fallback'],source=dict(m=64,n=64,k=64),source_percentage_of_torch=None)]))
        after['groups'][group]=guards(es,fallback)
compact=collapse_gemm_heuristic(after);check=expand_gemm_heuristic(compact)
assert check['fallback']==before['fallback']
assert all(check['groups'][g]==v for g,v in before['groups'].items() if '/bfloat16/' not in g)
assert all(e==old_exact[key(e)] for e in check['exact'] if key(e) not in keys)
assert len({key(e) for e in check['exact']})==len(check['exact'])
path.write_text(json.dumps(compact,separators=(',',':'))+'\n');_mapper.cache_clear()
for e in new_exact:
    seed=select_gemm_schedules(e['m'],e['n'],e['k'],operation=e['operation'],dtype=e['dtype']).seeds[0]
    assert seed.builder_config is not None and seed.captured_config==e['seed']['configuration']
for k,e in old_exact.items():
    if k in keys:continue
    seed=select_gemm_schedules(e['m'],e['n'],e['k'],operation=e['operation'],dtype=e['dtype'],batch=e['batch']).seeds[0]
    assert seed.captured_config==e['seed']['configuration']
report=dict(winning_shapes=len(rows),preserved_original_exact_rows=len(old_exact)-len(keys&old_exact.keys()),
            fp16_groups_preserved=True,fallback_preserved=True,new_exact_rows=len(check['exact'])-len(before['exact']))
if a.jit_table:
    from ayla.gemm_jit import _DEFAULTS
    table=root/'ayla/gemm_schedule_table.py';text=table.read_text();(backup/'table_before.py').write_text(text)
    syntax=ast.parse(text)
    cn=next(n.value for n in syntax.body if isinstance(n,ast.AnnAssign) and n.target.id=='CONFIGS')
    rn=next(n.value for n in syntax.body if isinstance(n,ast.AnnAssign) and n.target.id=='SCHEDULES')
    configs=[{kw.arg:ast.literal_eval(kw.value) for kw in n.keywords} for n in cn.elts]
    new_configs=copy.deepcopy(configs);old_routes=ast.literal_eval(rn);expected=copy.deepcopy(old_routes);updates={}
    for r in rows:
        cfg=r['configuration'];matches=[i for i,c in enumerate(new_configs) if {**_DEFAULTS,**c}=={**_DEFAULTS,**cfg}]
        if matches:idx=matches[0]
        else:idx=len(new_configs);new_configs.append(cfg)
        g=(r['op'],r['dtype']);k=(1,r['m'],r['n'],r['k']);old=old_routes[g][k]
        expected[g][k]=(idx,*old);updates[g,k]=expected[g][k]
    lines=text.splitlines(keepends=True);starts=[];total=0
    for line in lines:starts.append(total);total+=len(line)
    patches=[]
    for g,group_node in zip(rn.keys,rn.values):
        group=ast.literal_eval(g)
        for k,v in zip(group_node.keys,group_node.values):
            shape=ast.literal_eval(k)
            if (group,shape) in updates:
                patches.append((starts[v.lineno-1]+v.col_offset,starts[v.end_lineno-1]+v.end_col_offset,repr(updates[group,shape])))
    pos=starts[cn.end_lineno-1]
    patches.append((pos,pos,''.join('    dict('+', '.join(f'{k}={v!r}' for k,v in sorted(c.items()))+'),\n' for c in new_configs[len(configs):])))
    for start,end,value in sorted(patches,reverse=True):text=text[:start]+value+text[end:]
    table.write_text(text)
    parsed=ast.parse(text);actual=ast.literal_eval(next(n.value for n in parsed.body if isinstance(n,ast.AnnAssign) and n.target.id=='SCHEDULES'))
    assert actual==expected
    for g,group in old_routes.items():
        for k,v in group.items():
            assert actual[g][k][1:]==v if (g,k) in updates else actual[g][k]==v
    report.update(jit_updated_routes=len(updates),jit_new_configs=len(new_configs)-len(configs),all_old_fallbacks_preserved=True)
(backup/'checks.json').write_text(json.dumps(report,indent=2)+'\n');print(root.name,json.dumps(report))

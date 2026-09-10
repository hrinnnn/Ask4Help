"""Select an exact common budget using complete successful expert suffixes."""
import argparse,json,shutil
from pathlib import Path

def reachable(rows):
    found={0:()}
    for row in rows:
        if not row['accepted']:continue
        length=row['expert_suffix_actions']
        for total,chosen in list(found.items()):
            target=total+length;candidate=chosen+(row['episode'],)
            if target not in found or candidate<found[target]:found[target]=candidate
    return found

def run(a):
    source=json.loads((a.source/'controller_state.json').read_text())['completed_roots']
    a.output.mkdir(parents=True,exist_ok=False);mapping={'fixed':'fixed','feedback':'sensitive'};summaries={};choices={}
    for arm,key in mapping.items():
        root=Path(source[key]);s=json.loads((root/'summary.json').read_text())
        audit=json.loads((root/'INDEPENDENT_COLLECTION_AUDIT.json').read_text());assert audit['status']=='PASS'
        assert s['accepted']==30
        summaries[arm]=s;choices[arm]=reachable(s['rows'])
        dest=a.output/arm;dest.mkdir()
        for name in ['summary.json','provenance.json','INDEPENDENT_COLLECTION_AUDIT.json','COLLECTION_COMPLETE.json']:
            shutil.copyfile(root/name,dest/name)
    budget=max(set(choices['fixed'])&set(choices['feedback']));assert budget>0
    result={'schema':'complete_suffix_budget_selection','rule':'maximum common reachable action budget; lexicographically earliest episode tuple breaks ties; no timing or test outcome selection','results':{}}
    for arm,s in summaries.items():
        selected=choices[arm][budget]
        result['results'][arm]={'expert_action_budget':budget,'selected_episodes':selected,
            'selected_ID':sum(s['rows'][i]['split']=='id' for i in selected),
            'selected_OOD':sum(s['rows'][i]['split']=='ood' for i in selected),
            'all_accepted_actions':sum(r['expert_suffix_actions'] for r in s['rows'] if r['accepted'])}
    (a.output/'matched_selection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

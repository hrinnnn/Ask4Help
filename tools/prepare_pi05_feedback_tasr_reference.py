"""Package only existing audited numeric TASR references for new evaluation."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--urdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
manifest={}
with ZipFile(args.output,'w',compression=ZIP_DEFLATED) as bundle:
    bundle.write(args.urdf,'panda_v2.urdf')
    for task in ['stackcube','airplane']:
        old=json.loads((args.source/f'tasr_{task}.json').read_text())
        normal=[r for r in old['rows'] if r['method']=='offline_oracle' and r['split']=='ood']
        nominal={r['seed'] for r in normal}
        limits=next(r['thresholds'] for r in old['rows'] if r['seed'] not in nominal)
        rows=[]
        for r in normal:
            names={}
            for key in ['arrays','replay_array']:
                name=f'{task}/{r["seed"]}_{key}.npz';bundle.write(r[key],name);names[key]=name
            rows.append({k:r[k] for k in ['seed','expert_start','expert_points','blocks','target_replay_max_error']}|names)
        manifest[task]={'protocol':old['protocol'],'limits':limits,'normal':rows,
                        'numerator_domain':'OOD only; all selected expert points remain in denominator',
                        'source_report':str(args.source/f'tasr_{task}.json')}
    bundle.writestr('manifest.json',json.dumps(manifest,indent=2))
print({task:len(data['normal']) for task,data in manifest.items()})

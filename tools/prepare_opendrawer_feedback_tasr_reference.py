"""Export the frozen historical Grasp-OOD reference, without new fitting."""
import argparse,io,json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from analyze_open_drawer_target_ratio import relative_pose


def run(args):
    source=args.workspace/'artifacts/open_drawer_tolerance_sweep_20260903/baseline_analysis.json'
    original=json.loads(source.read_text());protocol=original['protocol']
    wanted=set(protocol['reference_seeds']+protocol['check_seeds']);references=[]
    with ZipFile(args.output,'w',compression=ZIP_DEFLATED) as bundle:
        for row in original['rows']:
            if row['anchor']!=0 or row['seed'] not in wanted:continue
            directory=args.workspace/row['directory']
            timeline=json.loads((directory/'task_state_timeline.json').read_text())['rows']
            state=np.load(directory/'states.npy');assert len(state)==len(timeline)
            assert row['accepted'] and timeline[-1]['success']
            pose={'position':np.asarray([r['tcp_position'] for r in timeline]),
                  'quaternion':np.asarray([r['tcp_quaternion'] for r in timeline]),
                  'width':state[:,-2:].sum(1)[:,None]}
            relative=relative_pose(pose,timeline)
            name=f"seed_{row['seed']}.npz";buf=io.BytesIO()
            np.savez_compressed(buf,**relative,contact=np.asarray([r['object_grasped'] for r in timeline]))
            bundle.writestr(name,buf.getvalue())
            references.append({'seed':row['seed'],'blocks':row['blocks'],'array':name,
                               'is_reference':row['seed'] in protocol['reference_seeds'],
                               'old_phase_stats':row['phase_stats'],'source_directory':str(directory)})
        assert len(references)==len(wanted)
        metadata={'format':'frozen_opendrawer_grasp_TASR_reference_v1','source':str(source),
                  'protocol':protocol,'references':references,
                  'not_applicable_to':'Goal-OOD placement target; it needs its own target phases and reference bank'}
        bundle.writestr('manifest.json',json.dumps(metadata,indent=2))
    print(json.dumps({'references':len(protocol['reference_seeds']),'checks':len(protocol['check_seeds']),
                      'output':str(args.output),'bytes':args.output.stat().st_size}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())

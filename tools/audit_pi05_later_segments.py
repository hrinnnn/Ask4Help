"""Inspect complete expert suffix agreement. Diagnostic, never updates a gate."""
import argparse,json
from pathlib import Path
import numpy as np
from pi05_feedback_runtime import Pi05FeedbackRuntime

def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text())
    runtime=Pi05FeedbackRuntime('stackcube_legacy_ood',manifest['task_assets']['stackcube_legacy_ood'])
    runtime.torch.set_num_threads(4);runtime.load_model()
    cal=json.loads((a.calibration/'calibration.json').read_text())
    rows=json.loads((a.collection/'summary.json').read_text())['rows']
    reports=[]
    for episode in [1,3,7,17]:
        row=rows[episode];directory=Path(row['artifact_directory'])
        with np.load(directory/'trace.npz') as data:
            qpos=data['qpos'];main_rgb=data['main'];wrist=data['wrist'];actions=data['actions']
        env=runtime.build_env(row['split']);env.reset(seed=row['seed'])
        t=row['takeover'];blocks=[]
        for k in range(0,len(actions)-t-4,5):
            snap={'qpos':qpos[t+k],'main':main_rgb[t+k],'wrist':wrist[t+k]}
            targets=actions[t+k:t+k+5];predictions=[]
            for m in range(2):
                pred,_=runtime.predict(runtime.raw_snapshot(snap),280009+row['seed']*10+k*2+m)
                predictions.append(runtime.clip(pred,env)[:5])
            pred=np.asarray(predictions);sq=(pred-targets[None])**2
            blocks.append(dict(offset=k,overall_MSE=float(sq.mean()),arm_MSE=float(sq[:,:,:7].mean()),
                gripper_MSE=float(sq[:,:,7].mean()),gripper_sign_disagreement=float((np.sign(pred[:,:,7])!=np.sign(targets[None,:,7])).mean()),
                per_dimension_MSE=sq.mean(axis=(0,1)).tolist(),predictions=pred.tolist(),expert_actions=targets.tolist(),
                passes_old_threshold=bool(sq.mean()<=cal['error_reference'])))
        env.close()
        reports.append(dict(episode=episode,source=str(directory),takeover=t,old_cue=row['cue'],blocks=blocks))
        (a.output/'progress.json').write_text(json.dumps({'completed_cases':len(reports),'last_episode':episode}))
        print(json.dumps({'episode':episode,'blocks':[{k:v for k,v in b.items() if k not in ['predictions','expert_actions','per_dimension_MSE']} for b in blocks]}),flush=True)
    report=dict(scope='diagnostic on prior development expert states; not autonomous continuation or fresh validation',
                old_threshold=cal['error_reference'],runtime=runtime.provenance(),cases=reports)
    (a.output/'segment_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (a.output/'SEGMENT_AUDIT_COMPLETE.json').write_text(json.dumps({'cases':len(reports),'training':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['manifest','calibration','collection','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())

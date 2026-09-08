"""Real paired RGB/action/bridge smoke; never a success-rate experiment."""
import argparse
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from pi05_feedback_runtime import Pi05FeedbackRuntime


def main(args):
    started=time.time()
    args.output.mkdir(parents=True,exist_ok=False)
    def report(stage, **fields):
        row={'pid':os.getpid(),'stage':stage,'elapsed':time.time()-started,**fields}
        (args.output/'progress.json').write_text(json.dumps(row,indent=2))
        print(json.dumps(row),flush=True)
    report('imports')
    manifest=json.loads(args.manifest.read_text())
    runtime=Pi05FeedbackRuntime(args.task,manifest['task_assets'][args.task])
    runtime.torch.set_num_threads(4)
    report('model_load',provenance=runtime.provenance())
    runtime.load_model()
    report('model_loaded')
    rows=[]
    import cv2
    for split in ['id','ood']:
        report('reset',split=split)
        env=runtime.build_env(split)
        raw,info=env.reset(seed=args.seed)
        metadata=runtime.reset_metadata(env,split=split)
        snap=runtime.snapshot(env,raw)
        for camera in ['main','wrist']:
            ok,encoded=cv2.imencode('.png',snap[camera][:,:,::-1])
            assert ok
            (args.output/f'{split}_{camera}.png').write_bytes(encoded.tobytes())
        action1,model_action=runtime.predict(raw,args.seed*1000)
        action2,_=runtime.predict(raw,args.seed*1000)
        assert action1.shape==action2.shape and action1.shape[1]==8
        assert np.isfinite(action1).all()
        rng_diff=float(np.max(np.abs(action1-action2)))
        assert rng_diff <= 1e-6, rng_diff
        bridge=runtime.bridge(raw,model_action)
        assert bridge.ndim==1 and bridge.size>0 and np.isfinite(bridge).all()
        clipped=runtime.clip(action1[:5],env)
        trajectory=[snap['qpos']]
        for action in clipped:
            raw,_,terminated,truncated,info=env.step(runtime.torch.as_tensor(action,device=env.unwrapped.device).reshape(1,-1))
            trajectory.append(runtime.snapshot(env,raw)['qpos'])
            if bool(terminated) or bool(truncated):break
        buf=io.BytesIO()
        np.savez_compressed(buf,policy_actions=action1,bridge_feature=bridge,qpos=np.asarray(trajectory),reset_tcp=snap['tcp'])
        (args.output/f'{split}_arrays.npz').write_bytes(buf.getvalue())
        row={'split':split,'seed':args.seed,'reset':metadata,'action_shape':list(action1.shape),
             'bridge_shape':list(bridge.shape),'paired_prediction_max_difference':rng_diff,
             'executed_actions':len(trajectory)-1,'main_shape':list(snap['main'].shape),
             'wrist_shape':list(snap['wrist'].shape),'final_success_diagnostic':bool(info['success'])}
        rows.append(row);report('split_complete',result=row)
        env.close()
    final={'status':'RGB_ACTION_BRIDGE_SMOKE_COMPLETE','task':args.task,'rows':rows,
           'provenance':runtime.provenance(),'scope':'Five autonomous actions per split; not a rollout success-rate denominator or expert-collector validation.'}
    (args.output/'summary.json').write_text(json.dumps(final,indent=2))
    (args.output/'SMOKE_COMPLETE.json').write_text(json.dumps({'task':args.task,'rows':len(rows)}))
    report('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['stackcube_legacy_ood','airplane_yaw_ood'],required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=1730101)
    args=p.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.exists():
            (args.output/'SMOKE_FAILED.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error)}))
        raise

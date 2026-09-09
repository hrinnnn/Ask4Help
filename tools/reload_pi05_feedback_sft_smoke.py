"""Reload the two-step engineering checkpoint and check real task forward output."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from pi05_feedback_runtime import Pi05FeedbackRuntime


def run(a):
    m=json.loads(a.manifest.read_text());base=dict(m['task_assets'][a.task])
    runtime=Pi05FeedbackRuntime(a.task,{**base,'checkpoint':str(a.checkpoint)})
    runtime.inference_mode='eval';torch=runtime.torch;torch.set_num_threads(4)
    runtime.load_model()
    old_path=runtime.helper.resolve_sft_full_weights(Path(base['checkpoint']))
    new_path=runtime.helper.resolve_sft_full_weights(a.checkpoint)
    old=torch.load(old_path,map_location='cpu',mmap=True,weights_only=True)
    new=torch.load(new_path,map_location='cpu',mmap=True,weights_only=True)
    keys=[k for k in old if k.startswith(('action_in_proj.','action_out_proj.'))]
    assert keys and all(k in new for k in keys)
    changes={k:float((old[k].float()-new[k].float()).abs().max()) for k in keys}
    assert any(v>0 for v in changes.values()),'No action projection update observed'
    env=runtime.build_env('id');raw,_=env.reset(seed=1799900)
    actions,_=runtime.predict(raw,1799900000)
    assert actions.shape==(10,8) and np.isfinite(actions).all()
    a.output.mkdir(parents=True,exist_ok=False)
    snap=runtime.snapshot(env,raw)
    Image.fromarray(snap['main']).save(a.output/'reload_reset_main.png')
    Image.fromarray(snap['wrist']).save(a.output/'reload_reset_wrist.png')
    env.close()
    result={'status':'TWO_STEP_CHECKPOINT_RELOAD_FORWARD_PASS','checkpoint':str(a.checkpoint),
            'base_checkpoint':base['checkpoint'],'action_shape':list(actions.shape),
            'action_projection_max_changes':changes,'runtime':runtime.provenance(),
            'scope':'Engineering reload/forward only; no task SR or learning efficacy claim'}
    (a.output/'RELOAD_FORWARD_COMPLETE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['manifest','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--task',required=True);run(p.parse_args())

"""Prepare, but do not execute, a two-update native SFT smoke with exact source provenance."""
import argparse,json
from pathlib import Path


def run(a):
    config=json.loads(a.manifest.read_text())
    audit=json.loads((a.data/'NATIVE_BATCH_AUDIT.json').read_text())
    assert audit['status']=='NATIVE_TRANSFORMS_AND_BALANCED_BATCH_PASS'
    assert (a.data/'NATIVE_DATA_MASK_AUDIT.json').exists()
    exported=json.loads((a.data/f'{a.arm}_export_manifest.json').read_text())
    provenance=json.loads((Path(exported['source_collection'])/'provenance.json').read_text())
    task=provenance['runtime']['task'];assets=config['task_assets'][task]
    source=Path(provenance['runtime']['source_root'])
    ID=Path(assets.get('ID_dataset',assets.get('ID_dataset_candidate')))
    if not (ID/'meta/info.json').exists():ID=ID/'lerobot'
    checkpoint=Path(assets['checkpoint'])
    # Standalone restored .pt files need a separately registered read-only directory view.
    assert checkpoint.is_dir(),'Prepare a native checkpoint-directory view for standalone .pt assets first'
    assert (checkpoint/'actor/model_state_dict/full_weights.pt').exists() or (checkpoint/'model_state_dict/full_weights.pt').exists()
    assert assets['norm']==provenance['runtime']['norm']
    steps=config['comparison']['training_steps'][task]
    env={'CUDA_VISIBLE_DEVICES':a.gpu,'FEEDBACK_ASSIGNED_GPU':a.gpu,'FEEDBACK_ASSIGNED_CPUS':a.cpu,
         'ASK4HELP_RLINF_PLACEMENT':a.gpu+'-'+a.gpu,'JAX_PLATFORMS':'cpu',
         'PYTHONPATH':str(a.code/'tools')+':'+str(source)+':'+str(source/'RLinf'),
         'EMBODIED_PATH':str(source/'RLinf/examples/sft'),
         'FEEDBACK_ID_DATASET':str(ID),'FEEDBACK_EXPERT_DATASET':str(a.data/a.arm),
         'FEEDBACK_NORM_PATH':assets['norm'],'FEEDBACK_BASE_CHECKPOINT':str(checkpoint),
         'FEEDBACK_TASK_INSTRUCTION':provenance['runtime']['instruction'],
         'FEEDBACK_SFT_OUTPUT':str(a.scratch),'FEEDBACK_SFT_STEPS':str(steps),'FEEDBACK_SFT_SEED':'1787000',
         'RAY_TMPDIR':'/tmp/fbsmoke_ray_'+a.gpu,'TMPDIR':'/tmp/fbsmoke_tmp_'+a.gpu,
         'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'HF_DATASETS_CACHE':'/tmp/pi05_timing_feedback_ablation_v1/sft_data_cache',
         'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
    command=['taskset','-c',a.cpu,config['runtime_deployment']['python'],str(a.code/'tools/train_pi05_feedback_sft.py'),
             '--config-path',str(a.code/'configs'),'--config-name','pi05_feedback_sft',
             'runner.max_steps=2','runner.save_interval=2',f'actor.optim.total_training_steps={steps}']
    output={'status':'PREPARED_NOT_LAUNCHED','task':task,'arm':a.arm,'updates':2,'source_collection':exported['source_collection'],
            'retained_expert_anchors':exported['anchors'],'environment':env,'command':command,
            'checkpoint_expected':str(a.scratch/'run/checkpoints/global_step_2'),
            'scope':'Engineering smoke only, not formal SFT or an efficacy result.',
            'before_launch':['verify actual idle GPU and RAM/shm capacity','write source/protocol audit','respect pending formal training admission'],
            'after_run':['verify2 optimizer updates and temporal-mask metrics','reload finite model on task data','persist checkpoint and diagnostics to durable output']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2))
    print(json.dumps({'status':output['status'],'task':task,'anchors':exported['anchors'],'instruction':env['FEEDBACK_TASK_INSTRUCTION']}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['manifest','code','data','scratch','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=['fixed','feedback'],default='fixed')
    p.add_argument('--gpu',required=True);p.add_argument('--cpu',required=True);run(p.parse_args())

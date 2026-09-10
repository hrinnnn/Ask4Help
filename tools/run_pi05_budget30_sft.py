"""User-authorized matched-budget SFT and independent evaluation controller."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path

def run(a):
    p=json.loads((a.code/'configs/pipelines/pi05_budget30_sft_v1.json').read_text())
    assert p['authorized'] and p['training_authorized']
    root=Path(p['output_root']);root.mkdir(parents=True,exist_ok=True)
    logs=Path('/tmp/pi05_timing_feedback_ablation_v1/budget30_sft_logs');logs.mkdir(exist_ok=True)
    statepath=root/('prepare_state.json' if a.arm is None else a.arm+'_state.json')
    state={'controller_pid':os.getpid(),'mode':a.arm or 'prepare','stage':'preflight','jobs':[],'training_authorized':True}
    def save():statepath.write_text(json.dumps(state,indent=2))
    base_env={**os.environ,'PYTHONPATH':str(a.code/'tools')+':'+p['native_source']+':'+p['native_source']+'/RLinf',
              'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
              'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TMPDIR':'/tmp/pi05_timing_feedback_ablation_v1/tmp0'}
    def command(cmd,label,env=None):
        log=logs/(label+'.log');state['stage']=label
        with log.open('a') as f:q=subprocess.Popen(cmd,env=env or base_env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record={'pid':q.pid,'command':cmd,'log':str(log)};state['jobs'].append(record);save()
        rc=q.wait();record['returncode']=rc;save()
        if rc:raise RuntimeError(f'{label} failed; see {log}')
        shutil.copyfile(log,root/(label+'.log'))
    py=p['python'];tool=lambda n:str(a.code/'tools'/n)
    try:
        if a.arm is None:
            shutil.copyfile(a.code/'configs/pipelines/pi05_budget30_sft_v1.json',root/'manifest.json')
            view=Path(p['checkpoint_view']);link=view/'actor/model_state_dict/full_weights.pt';link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(p['base_weights'])
            assert link.resolve()==Path(p['base_weights']).resolve()
            manifest=json.loads((a.code/'configs/pipelines/pi05_timing_feedback_ablation_v1.json').read_text())
            manifest['task_assets'][p['task']].update(checkpoint=str(view),ID_dataset=p['ID_dataset'],norm=p['norm'])
            manifest['comparison']['training_steps'][p['task']]=p['training']['steps']
            (root/'training_manifest.json').write_text(json.dumps(manifest,indent=2))
            if not (root/'matched_collection/matched_selection.json').exists():
                command([py,tool('prepare_pi05_budget30_training_data.py'),'--source',p['source_collection'],'--output',str(root/'matched_collection')],'budget_selection')
            if not (root/'data/EXPORT_COMPLETE.json').exists():
                command([py,tool('export_pi05_feedback_sft_data.py'),'--paired-root',str(root/'matched_collection'),'--selection',str(root/'matched_collection/matched_selection.json'),'--output',str(root/'data')],'data_export')
            if not (root/'data/NATIVE_DATA_MASK_AUDIT.json').exists():
                command([py,tool('audit_pi05_feedback_sft_data.py'),'--root',str(root/'data'),'--source',p['native_source']],'data_mask_audit')
            if not (root/'data/NATIVE_BATCH_AUDIT.json').exists():
                command([py,tool('audit_pi05_feedback_native_batches.py'),'--source',p['native_source'],'--collection',str(root/'matched_collection'),'--export',str(root/'data'),'--manifest',str(root/'training_manifest.json')],'native_batch_audit')
            state['stage']='data_prepared_pending_visual';save();(root/'DATA_PREPARED.json').write_text(json.dumps(state));return
        assert (root/'DATA_PREPARED.json').exists() and (root/'VISUAL_DATA_AUDIT_COMPLETE.json').exists()
        arm=a.arm;slot=p['arms'][arm];gpu=slot['gpu'];cpu=slot['cpu']
        uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
        assert not [x for x in active.splitlines() if uuid in x and x.split(',')[0].strip()!='276925']
        env={**base_env,'CUDA_VISIBLE_DEVICES':gpu,'FEEDBACK_ASSIGNED_GPU':gpu,'FEEDBACK_ASSIGNED_CPUS':cpu,
             'ASK4HELP_RLINF_PLACEMENT':gpu+'-'+gpu,'EMBODIED_PATH':p['native_source']+'/RLinf/examples/sft',
             'FEEDBACK_ID_DATASET':p['ID_dataset'],'FEEDBACK_EXPERT_DATASET':str(root/'data'/arm),
             'FEEDBACK_NORM_PATH':p['norm'],'FEEDBACK_BASE_CHECKPOINT':p['checkpoint_view'],
             'FEEDBACK_TASK_INSTRUCTION':p['instruction'],'FEEDBACK_SFT_STEPS':str(p['training']['steps']),
             'FEEDBACK_SFT_SEED':str(p['training']['seed']),'HF_DATASETS_CACHE':'/tmp/pi05_timing_feedback_ablation_v1/sft_data_cache',
             'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json','TMPDIR':f'/tmp/pi05_budget30_sft_tmp_{gpu}',
             'RAY_TMPDIR':f'/tmp/pi05_budget30_sft_ray_{gpu}'}
        for key in ['TMPDIR','RAY_TMPDIR']:Path(env[key]).mkdir(exist_ok=True)
        for stage,steps in [('smoke',2),('formal',p['training']['steps'])]:
            stage_root=root/arm/stage;stage_root.mkdir(parents=True,exist_ok=True)
            archive=stage_root/'checkpoints';checkpoint=archive/f'global_step_{steps}'
            stage_env={**env,'FEEDBACK_SFT_OUTPUT':f'/dev/shm/pi05_budget30_{arm}_{stage}',
                       'FEEDBACK_CHECKPOINT_ARCHIVE':str(archive)}
            if not (checkpoint/'ARCHIVE_COMPLETE.json').exists():
                assert not Path(stage_env['FEEDBACK_SFT_OUTPUT']).exists(),'Inspect partial training and use explicit recovery, do not overwrite'
                cmd=['taskset','-c',cpu,py,tool('train_pi05_feedback_sft.py'),'--config-path',str(a.code/'configs'),'--config-name','pi05_feedback_sft',
                     f'runner.max_steps={steps}',f'runner.save_interval={2 if stage=="smoke" else p["training"]["save_every"]}',
                     f'actor.optim.total_training_steps={p["training"]["steps"]}']
                command(cmd,arm+'_'+stage+'_training',stage_env)
            if not (stage_root/'reload/RELOAD_FORWARD_COMPLETE.json').exists():
                command(['taskset','-c',cpu,py,tool('reload_pi05_feedback_sft_smoke.py'),'--manifest',str(root/'training_manifest.json'),
                    '--task',p['task'],'--checkpoint',str(checkpoint),'--output',str(stage_root/'reload')],arm+'_'+stage+'_reload',env)
        final=root/arm/'formal/checkpoints'/f'global_step_{p["training"]["steps"]}'
        for split in ['id','ood']:
            evalroot=root/arm/('eval_'+split);evalroot.mkdir(exist_ok=True)
            chunks=sorted(evalroot.glob('chunk_*'));resume=chunks[-1] if chunks else None
            n=len(json.loads((resume/'summary.json').read_text())['rows']) if resume else 0
            while n<100:
                output=evalroot/f'chunk_{n:04d}';assert not output.exists()
                cmd=['taskset','-c',cpu,py,tool('evaluate_pi05_feedback_policy.py'),'--manifest',str(root/'training_manifest.json'),
                     '--protocol',str(a.code/p['evaluation_protocol']),'--checkpoint',str(final),'--output',str(output),
                     '--task',p['task'],'--split',split,'--max-new-episodes','20']
                if resume:cmd+=['--resume-from',str(resume)]
                command(cmd,arm+'_'+split+f'_eval_{n:04d}',env)
                resume=output;n=len(json.loads((output/'summary.json').read_text())['rows'])
            command([py,tool('audit_pi05_feedback_policy_evaluation.py'),'--root',str(resume)],arm+'_'+split+'_final_audit',env)
            state[split+'_evaluation']=str(resume);save()
        state['stage']='training_and_evaluation_complete';save();(root/(arm+'_COMPLETE.json')).write_text(json.dumps(state,indent=2))
    except Exception as e:
        state.update(stage='needs_engineering_review',error=repr(e));save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--arm',choices=['fixed','feedback'])
    run(p.parse_args())

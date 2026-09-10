"""Restartable CPU extraction -> masked head update -> result summary controller."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np

def main(a):
 if (a.root/'INVALID_EVIDENCE_USER_CORRECTION.json').exists():
  raise SystemExit('Stopped by user: historical Stage2 evaluation is invalid; do not resume.')
 a.root.mkdir(parents=True,exist_ok=True);state=a.root/'pipeline_state.json';log=a.root/'controller.log'
 def mark(stage,**extra):
  state.write_text(json.dumps(dict(pipeline='tns_model_utility_v1',stage=stage,pid=os.getpid(),time=time.time(),**extra),indent=2))
 specpath=a.root/'independent_sample_spec.json'
 if not specpath.exists():
  spec=json.loads(a.spec.read_text());reserved=set(spec['ood_probe_seeds']);train=set(spec['all_selected_training_seeds']);assert not reserved.intersection(train)
  source=Path('/mnt/data/ask4help/results/xvla_stackcube_v1/temporal_mask_v2/stackcube_target_ood_timing_v1_retry2')
  records=[json.loads(x) for x in (source/'collection_pools/failure_recovery/training_episodes.jsonl').read_text().splitlines() if x.strip()]
  rec=[r for r in records if r['seed'] in reserved];missing=sorted(reserved-{r['seed'] for r in rec});assert len(rec)>=10
  rng=np.random.default_rng(310910)
  for r in rec:
   n=r['expert_action_steps'];offsets=sorted(set([0,n-1,*rng.choice(n,min(5,n),replace=False).tolist()]))
   path=source/'dataset_pools/failure_recovery/data/chunk-000'/f"episode_{r['dataset_episode_index']:06d}.parquet";assert path.exists()
   for i in offsets:spec['samples'].append(dict(group='ood_recovery_probe',seed=r['seed'],split='stage2_ood',episode=r['dataset_episode_index'],offset=i,path=str(path),episode_length=n,actual_takeover=r['expert_start_step'],role='heldout_recovery_probe'))
  spec['id_contiguous_episodes']=[0,1,2,3];spec['id_calibration_episodes']=[0,1,2,3]
  spec['tns_independent_contract']=dict(missing_recovery_probe_seeds=missing,recovery_probe_episodes=len(rec),ID_update_episodes=[0,1,2,3],ID_probe_episodes='108..127 randomly sampled96',native_input_size=224,device='cpu',threads=4,scope='frozen transformer features; final affine-head update only',candidate='50percentID/25percentnominal/25percentrecovery heldout loss decrease',candidate_frozen_before_new_cache_scoring=True)
  specpath.write_text(json.dumps(spec,indent=2));mark('SPEC_READY',recovery_probe_episodes=len(rec))
 if a.prepare_only:return
 cache=a.root/'independent_cache_cpu_v1';scores=a.root/'independent_scores_v1'
 stages=[('EXTRACTING',cache/'FEATURE_EXTRACTION_COMPLETE.json',[a.python,str(a.code/'run_tns_xvla_cache.py'),'--spec',str(specpath),'--output',str(cache),'--device','cpu','--threads','4']),('SCORING',scores/'CACHED_HEAD_COMPLETE.json',[a.python,str(a.code/'run_tns_cached_head_probe.py'),'--cache',str(cache),'--output',str(scores),'--include-gates','--independent'])]
 try:
  for stage,done,cmd in stages:
   if done.exists():continue
   with log.open('a') as stream:
    proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
    mark(stage,child_pid=proc.pid,command=cmd,log=str(log),completion=str(done));rc=proc.wait()
   if rc!=0 or not done.exists():raise RuntimeError(f'{stage}:exit={rc},marker={done.exists()}')
  result=json.loads((scores/'results.json').read_text());mark('READY_FOR_INDEPENDENT_AUDIT',results=str(scores/'results.json'),correlations=result['correlations'])
  (a.root/'COMPUTE_COMPLETE.json').write_text(json.dumps(dict(status='ready_for_audit_not_final_scientific_acceptance',results=str(scores/'results.json'))))
 except Exception as e:
  mark('FAILED',error=str(e));(a.root/'FAILED.json').write_text(json.dumps(dict(error=str(e))));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--python',default='/root/.venvs/xvla-h20/bin/python');p.add_argument('--prepare-only',action='store_true');main(p.parse_args())

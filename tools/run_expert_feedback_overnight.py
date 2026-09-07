"""Durable sequential controller for an explicitly frozen exploratory grid."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--code',type=Path,required=True);parser.add_argument('--grid',type=Path,required=True);parser.add_argument('--gpu',default='1');parser.add_argument('--cpu',default='4-7');args=parser.parse_args()
args.root.mkdir(parents=True,exist_ok=True);grid=json.loads(args.grid.read_text());statefile=args.root/'pipeline_state.json'
state=dict(controller_pid=os.getpid(),stage='development_collection',next_stage='independent_audit_and_shortlist',started=time.time(),completed=[],grid=str(args.grid),gpu=args.gpu,cpu=args.cpu)
def save():statefile.write_text(json.dumps(state,indent=2))
save()
for offset in grid['seed_offsets']:
 for v in grid['variants']:
  key=f'seed_{offset}/{v["name"]}';directory=args.root/key
  if (directory/'PILOT_COMPLETE.json').exists():state['completed'].append(key);continue
  if directory.exists() and any(directory.iterdir()):
   state.update(stage='partial_needs_engineering_review',current=key);save();sys.exit(2)
  directory.mkdir(parents=True,exist_ok=True)
  cmd=['taskset','-c',args.cpu,sys.executable,str(args.code/'tools/probe_expert_feedback_closedloop.py'),'--arm',v.get('arm','adaptive_prefix'),'--output',str(directory),'--episodes',str(grid.get('episodes',20)),'--seed-offset',str(offset),'--calibration',str(args.code/v['calibration']),'--pca','/mnt/data/ask4help/results/expert_feedback_pca_probe_v1/original_pca_bridge.npz','--feedback-rule',v.get('rule','original'),'--radius-multiplier',str(v.get('radius',1)),'--feedback-strength',str(v.get('strength',.5)),'--min-support-episodes',str(v.get('min_support',2))]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
  log=args.root/(key.replace('/','_')+'.log')
  with log.open('w') as f:
   child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   state.update(current=key,child_pid=child.pid,command=cmd,log=str(log));save();rc=child.wait()
  if rc or not (directory/'PILOT_COMPLETE.json').exists():state.update(stage='engineering_failed',returncode=rc);save();sys.exit(1)
  state['completed'].append(key);state['child_pid']=None;save()
state.update(stage='collection_complete',next_stage='independent_audit_and_shortlist',finished=time.time());save()
(args.root/'COLLECTION_COMPLETE.json').write_text(json.dumps(state,indent=2))
audit=[sys.executable,str(args.code/'tools/audit_expert_feedback_overnight.py'),'--root',str(args.root),'--grid',str(args.grid)]
rc=subprocess.call(audit)
state.update(stage='development_audited' if rc==0 else 'audit_failed',next_stage='freeze_shortlist_then_validation' if rc==0 else 'engineering_review');save()

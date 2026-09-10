"""Local real-cache final-head utility probe, with seed-disjoint cross-fitting.
No original checkpoint is modified. This is not full-model SFT or new policy SR.
"""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np

def main(args):
 out=args.output;out.mkdir(parents=True,exist_ok=True);t0=time.time()
 meta=json.loads((args.cache/'samples.json').read_text());d=np.load(args.cache/'head_probe_cache.npz')
 H=d['head_features'].astype(np.float32);Y=d['targets'].astype(np.float32);M=d['valid_mask'];W=d['head_weight'];b=d['head_bias']
 assert H.shape[:3]==(len(meta),2,10) and Y.shape==(len(meta),10,8)
 lengths=np.array([min(10,s['episode_length']-s['offset']) for s in meta]);assert np.array_equal(M.sum(1),lengths)
 assert all(M[i].sum()==1 for i,s in enumerate(meta) if s['offset']==s['episode_length']-1)
 H=np.concatenate([H,np.ones((*H.shape[:-1],1),dtype=np.float32)],axis=-1)
 base=np.vstack([W,b]);residual=H@base-Y[:,None,:,:]
 weights=M.astype(np.float32)/lengths[:,None]
 def loss(idx,delta):
  err=residual[idx]+H[idx]@delta
  per=(np.mean(err*err,axis=-1)*weights[idx,None,:]).sum(2).mean(1)*100
  # Uniform episode averaging in held-out probes prevents long episodes dominating.
  pools=defaultdict(list)
  for k,v in zip(idx,per):pools[meta[k]['episode_id']].append(float(v))
  return float(np.mean([np.mean(x) for x in pools.values()]))
 def gradient(idx,delta):
  x=H[idx];err=residual[idx]+x@delta
  werr=err*weights[idx,None,:,None]
  return (25/(len(idx)*2))*(x.reshape(-1,x.shape[-1]).T@werr.reshape(-1,8))
 methods=['immediate','post_grasp','post_lift','failure_recovery']+(['internal_pca','diffdagger'] if args.include_gates else [])
 seeds=sorted({s['seed'] for s in meta if s['group']=='expert_immediate'})
 assert len(seeds)==8
 idtrain=np.array([i for i,s in enumerate(meta) if s['group']=='id_calibration'])
 idcheck=np.array([i for i,s in enumerate(meta) if s['group']=='id_check'])
 nparams=base.shape
 zero=np.zeros(nparams,np.float32)
 # Verify the implemented masked gradient by a symmetric finite difference.
 ix=np.array([i for i,s in enumerate(meta) if s['group']=='expert_immediate'])[-12:]
 grad=gradient(ix,zero);direction=grad/np.linalg.norm(grad)
 def batchloss(delta):
  err=residual[ix]+H[ix]@delta
  return float((np.mean(err*err,axis=-1)*weights[ix,None,:]).sum(2).mean()*100)
 eps=1e-4;fd=(batchloss(eps*direction)-batchloss(-eps*direction))/(2*eps);analytic=float(np.sum(grad*direction));relative_error=abs(fd-analytic)/max(abs(analytic),1e-12)
 assert relative_error<.02,(fd,analytic,relative_error)
 runs=[]
 for fold in [0,1]:
  trainseeds=set(seeds[fold::2]);testseeds=set(seeds[1-fold::2]);assert trainseeds.isdisjoint(testseeds)
  probes={'nominal':np.array([i for i,s in enumerate(meta) if s['group']=='expert_immediate' and s['seed'] in testseeds]),'recovery':np.array([i for i,s in enumerate(meta) if s['group']=='expert_failure_recovery' and s['seed'] in testseeds]),'ID':idcheck}
  l0={k:loss(idx,zero) for k,idx in probes.items()}
  for method in methods:
   methodseeds=sorted({s['seed'] for s in meta if s['group']=='expert_'+method});mtrain=set(methodseeds[fold::2])
   train=np.array([i for i,s in enumerate(meta) if s['group']=='expert_'+method and s['seed'] in mtrain]);assert len(train)>0
   assert not mtrain.intersection(testseeds)
   for seed in [31001,31002,31003]:
    rng=np.random.default_rng(seed);delta=zero.copy();mom=zero.copy();variance=zero.copy();checkpoints=[]
    for step in range(1,101):
     # Fixed 1:1 source mixture, 16 ID and16 expert anchors per update.
     ix=np.r_[rng.choice(idtrain,16),rng.choice(train,16)]
     g=gradient(ix,delta)
     mom=.9*mom+.1*g;variance=.999*variance+.001*g*g
     delta-=args.lr*(mom/(1-.9**step))/(np.sqrt(variance/(1-.999**step))+1e-8)
     assert np.isfinite(delta).all()
     if step in [20,100]:
      after={k:loss(idx,delta) for k,idx in probes.items()}
      gain={k:l0[k]-after[k] for k in after};norm={k:gain[k]/l0[k] for k in gain}
      checkpoints.append(dict(step=step,before=l0,after=after,gain=gain,relative_gain=norm,TNS=.5*(norm['nominal']+norm['recovery']),TNS_retention=(.5*gain['ID']+.25*gain['nominal']+.25*gain['recovery'])/(.5*l0['ID']+.25*l0['nominal']+.25*l0['recovery']),ID_relative_gain=norm['ID']))
    runs.append(dict(method=method,fold=fold,seed=seed,train_reset_seeds=sorted(mtrain),probe_reset_seeds=sorted(testseeds),expert_anchors=len(train),tail_anchors=int(sum(lengths[train]<10)),checkpoints=checkpoints,delta_norm=float(np.linalg.norm(delta))))
    (out/'progress.json').write_text(json.dumps(dict(completed=len(runs),total=len(methods)*6,elapsed_seconds=time.time()-t0)))
    print('RUN',len(runs),method,fold,seed,'TNS',checkpoints[-1]['TNS'],flush=True)
 # Known SR is used only here, after all protocol-fixed updates and scoring.
 SR={'immediate':.52,'post_grasp':.10,'post_lift':.04,'failure_recovery':.30,'internal_pca':.74,'diffdagger':.45}
 Q={'immediate':.2225609756097561,'post_grasp':.3094512195121951,'post_lift':.4344512195121951,'failure_recovery':.19461382113821138,'internal_pca':.26497695852534564,'diffdagger':.1889400921658986}
 summary=[]
 for method in methods:
  rr=[r for r in runs if r['method']==method];c=[r['checkpoints'][-1] for r in rr]
  summary.append(dict(method=method,SR=SR[method],original_Q3=Q[method],TNS_retention_mean=float(np.mean([x['TNS_retention'] for x in c])),TNS_mean=float(np.mean([x['TNS'] for x in c])),TNS_std=float(np.std([x['TNS'] for x in c],ddof=1)),nominal_gain=float(np.mean([x['relative_gain']['nominal'] for x in c])),recovery_gain=float(np.mean([x['relative_gain']['recovery'] for x in c])),ID_gain=float(np.mean([x['relative_gain']['ID'] for x in c]))))
 rank=lambda x:np.array([1+sum(v<t for v in x)+(sum(v==t for v in x)-1)/2 for t in x])
 timings=[r for r in summary if r['method'] in ['immediate','post_grasp','post_lift','failure_recovery']]
 corr={k:float(np.corrcoef(rank([r[k] for r in timings]),rank([r['SR'] for r in timings]))[0,1]) for k in ['TNS_mean','TNS_retention_mean','nominal_gain','recovery_gain','ID_gain','original_Q3']}
 byrun=[]
 for fold in [0,1]:
  for seed in [31001,31002,31003]:
   rr=[r for r in runs if r['fold']==fold and r['seed']==seed and r['method'] in ['immediate','post_grasp','post_lift','failure_recovery']]
   byrun.append(dict(fold=fold,seed=seed,rho=float(np.corrcoef(rank([r['checkpoints'][-1]['TNS'] for r in rr]),rank([SR[r['method']] for r in rr]))[0,1])))
 result=dict(status='CACHED_FINAL_HEAD_PROBE_COMPLETE_NOT_FULL_VLA',protocol=dict(lr=args.lr,optimizer='Adam',beta=[.9,.999],eps=1e-8,steps=100,batch=32,ID_expert_ratio='1:1',train_seeds=[31001,31002,31003],folds=2,MC=2,head_shape=list(base.shape),loss='100 x MSE on real8 dimensions, mean valid targets per anchor, mean MC',score='mean of relative heldout nominal and recovery loss reductions',heldout_aggregation='equal episode weight',fixed_feature_limitation='No trunk/transformer updates; fixed noisy inputs; final affine-head response only'),input=dict(cache=str(args.cache.resolve()),samples=len(meta),tail=int(sum(lengths<10)),unique_original_episodes=len(set(s['episode_id'] for s in meta))),gradient_audit=dict(fd=fd,analytic=analytic,relative_error=relative_error),summary=summary,correlations=corr,per_fold_seed=byrun,runs=runs,limitations=['Only first8 episodes/method; not the complete original matched-budget training datasets.','Two seed folds and3 SGD-sampling repeats are not6 independent training datasets.','OOD probes are held-out resets from the same inspected collection; historicalSR already known.','Nominal/recovery probe mixture is fixed equal, not measured task-criticality or deployment occupancy.','Native cache bf16 versus FP32 affine reconstruction introduces small numerical differences.','ID_check episodes were part of original base training; measures ID retention, not new generalization.','No new closed-loop SR or full-model short-training result.'],elapsed_seconds=time.time()-t0)
 (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));(out/'CACHED_HEAD_COMPLETE.json').write_text(json.dumps(dict(runs=len(runs),finite=True,summary='results.json')))
 print(json.dumps(dict(summary=summary,correlations=corr,per_fold_seed=byrun,seconds=time.time()-t0),ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path('artifacts/expert_feedback_pca_20260907/free_action_v1'));p.add_argument('--output',type=Path,default=Path('artifacts/tns_model_utility_20260910/head_pilot_v1'));p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--include-gates',action='store_true');main(p.parse_args())

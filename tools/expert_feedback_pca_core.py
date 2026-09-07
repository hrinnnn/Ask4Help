"""Reference prototype: local threshold constraints from past expert feedback.

Input signals are frozen-model quantities. No OOD labels, task phases, optimal
takeover times or success labels are accepted by the algorithm.
"""
from dataclasses import dataclass,field
import numpy as np
from scipy.spatial.distance import cdist

def first_boundary_feedback(loss,valid,H=10,delta=5,q=1.,allow_censored_wait=False,lookahead=True):
 loss=np.asarray(loss,dtype=float);valid=np.asarray(valid,dtype=int);n=len(loss)
 if valid.shape!=(n,):raise ValueError('length mismatch')
 if np.any(valid>np.minimum(H,n-np.arange(n))):raise ValueError('invalid future-action mask')
 stop=next((i for i in range(n) if valid[i]<H or not np.isfinite(loss[i])),n)
 event=next((i for i in range(stop) if loss[i]>q),None)
 if event is None:
  idx=np.arange(max(0,stop-delta)) if allow_censored_wait else np.array([],dtype=int)
  return dict(event=None,observed_full_windows=stop,indices=idx,request=np.zeros(len(idx),dtype=bool),status='right_censored' if stop else 'no_full_window')
 idx=np.arange(event+1);request=(event-idx<delta) if lookahead else idx==event
 return dict(event=event,observed_full_windows=stop,indices=idx,request=request,status='observed_boundary')

@dataclass
class ExpertFeedbackPCA:
 tau0:float
 center:np.ndarray
 scale:float
 radius:float
 q_loss:float
 H:int=10
 delta:int=5
 comparator:str='gt'
 memory:list=field(default_factory=list)
 memory_episodes:set=field(default_factory=set)

 @classmethod
 def calibrate(cls,features,episodes,loss,valid,tau0,H=10,delta=5,quantile=.95,comparator='gt',calibration_mode='point'):
  x=np.asarray(features,dtype=float);ep=np.asarray(episodes);loss=np.asarray(loss,dtype=float);valid=np.asarray(valid)
  if not(np.isfinite(x).all() and np.isfinite(loss).all()):raise ValueError('nonfinite calibration')
  if len(np.unique(ep))<2:raise ValueError('need multiple ID episodes')
  center=x.mean(0);scale=float(np.sqrt(np.mean(np.sum((x-center)**2,axis=1))))
  if scale<=0:raise ValueError('degenerate support scale')
  z=(x-center)/scale;nearest=[]
  for first in range(0,len(z),128):
   d=cdist(z[first:first+128],z);d[ep[first:first+128,None]==ep[None,:]]=np.inf;nearest.extend(d.min(1))
  radius=float(np.quantile(nearest,quantile,method='higher'))
  full=valid==H
  values=loss[full] if calibration_mode=='point' else np.array([max(loss[(ep==e)&full]) for e in np.unique(ep) if np.any((ep==e)&full)])
  if not len(values):raise ValueError('no full calibration windows')
  q=float(np.quantile(values,quantile,method='higher'))
  return cls(float(tau0),center,scale,radius,q,H,delta,comparator)

 def normalize(self,x):return (np.asarray(x,dtype=float)-self.center)/self.scale
 def stopped(self,score,threshold):return bool(score>threshold if self.comparator=='gt' else score>=threshold)

 def query(self,feature,score):
  if not np.isfinite(score) or not np.isfinite(feature).all():raise ValueError('nonfinite query')
  z=self.normalize(feature);neighbors=[r for r in self.memory if np.linalg.norm(r['z']-z)<=self.radius]
  threshold=self.tau0;reason='no_support';lo=-np.inf;hi=np.inf
  if neighbors:
   wait=[r['score'] for r in neighbors if not r['request']];request=[r['score'] for r in neighbors if r['request']]
   if wait:lo=max(wait)
   if request:hi=min(request)
   if self.comparator=='gt':hi=np.nextafter(hi,-np.inf) if np.isfinite(hi) else hi
   else:lo=np.nextafter(lo,np.inf) if np.isfinite(lo) else lo
   if lo<=hi:threshold=float(np.clip(self.tau0,lo,hi));reason='supported'
   else:reason='conflict'
  return dict(threshold=threshold,stop=self.stopped(score,threshold),baseline_stop=self.stopped(score,self.tau0),reason=reason,neighbors=len(neighbors),
   wait_neighbors=sum(not r['request'] for r in neighbors),request_neighbors=sum(r['request'] for r in neighbors))

 def add_episode(self,episode_id,features,scores,loss,valid,allow_censored_wait=False,lookahead=True):
  if episode_id in self.memory_episodes:raise ValueError('episode already committed')
  if not(len(features)==len(scores)==len(loss)==len(valid)):raise ValueError('episode lengths differ')
  if not np.isfinite(features).all() or not np.isfinite(scores).all():raise ValueError('nonfinite features/scores')
  feedback=first_boundary_feedback(loss,valid,self.H,self.delta,self.q_loss,allow_censored_wait,lookahead)
  z=self.normalize(features)
  for i,y in zip(feedback['indices'],feedback['request']):self.memory.append(dict(episode=episode_id,offset=int(i),z=z[i],score=float(scores[i]),request=bool(y)))
  self.memory_episodes.add(episode_id)
  return dict(event=feedback['event'],status=feedback['status'],observed_full_windows=feedback['observed_full_windows'],
   wait_labels=int(np.sum(~feedback['request'])),request_labels=int(np.sum(feedback['request'])))

 def add_takeover_prefix_credit(self,episode_id,prefix_features,prefix_scores,prefix_steps,expert_loss,expert_valid,
                                regression=None,regression_bound=None,progress_supported=False):
  """Experimental adapter: attribute feedback only to real policy-query states.

  Regression is an ID-calibrated descriptor supplied by the progress adapter,
  never a manual 'too-late' label. This optional rule requires new on-policy
  validation; add_episode implements the supplied document's baseline.
  """
  if episode_id in self.memory_episodes:raise ValueError('episode already committed')
  x=np.asarray(prefix_features);r=np.asarray(prefix_scores);t=np.asarray(prefix_steps)
  if len(x)==0 or not(len(x)==len(r)==len(t)) or np.any(np.diff(t)<=0):raise ValueError('invalid actual policy prefix')
  if not np.isfinite(x).all() or not np.isfinite(r).all():raise ValueError('nonfinite policy prefix')
  feedback=first_boundary_feedback(expert_loss,expert_valid,self.H,self.delta,self.q_loss)
  onset=feedback['event'];corrective=progress_supported and regression is not None and regression_bound is not None and regression>regression_bound
  if corrective:
   eligible=np.flatnonzero(t<=t[-1]-self.delta);index=int(eligible[-1]) if len(eligible) else len(t)-1
   request=True;reason='corrective_prefix_move_one_observed_query_earlier' if len(eligible) else 'corrective_at_reset_no_earlier_state'
  elif onset is not None and onset<self.delta:index=len(t)-1;request=True;reason='current_supervision_gap_no_claim_of_lateness'
  elif onset is not None and onset>=self.delta:index=len(t)-1;request=False;reason='observed_low_loss_prefix_candidate_delay'
  else:
   self.memory_episodes.add(episode_id)
   return dict(reason='no_localizable_feedback',event=onset,credit_step=None)
  self.memory.append(dict(episode=episode_id,offset=int(t[index]),z=self.normalize(x[index]),score=float(r[index]),request=request))
  self.memory_episodes.add(episode_id)
  return dict(reason=reason,event=onset,credit_step=int(t[index]),request=request)

"""Exploratory bounded log-threshold correction from episode-balanced feedback."""
import numpy as np
from expert_feedback_pca_core import ExpertFeedbackPCA,first_boundary_feedback

class FlexibleFeedbackPCA(ExpertFeedbackPCA):
 def configure(self,rule='soft',radius_multiplier=2.,strength=.5,min_episodes=2,censored_wait=True):
  self.rule=rule;self.base_radius=self.radius;self.radius*=radius_multiplier;self.strength=strength;self.min_episodes=min_episodes;self.censored_wait=censored_wait
  return self

 def query(self,feature,score):
  if self.rule=='hard':return super().query(feature,score)
  z=self.normalize(feature);by_episode={}
  for r in self.memory:
   distance=np.linalg.norm(r['z']-z)
   if distance<=self.radius:
    weight=np.exp(-.5*(distance/self.radius)**2);bucket=by_episode.setdefault(r['episode'],[]);bucket.append((weight,1. if r['request'] else -1.))
  count=len(by_episode);threshold=self.tau0;vote=0.;reason='no_support' if count==0 else 'insufficient_episode_support'
  if count>=self.min_episodes:
   values=[sum(w*y for w,y in b)/sum(w for w,y in b) for b in by_episode.values()]
   vote=float(np.mean(values));confidence=count/(count+2.)
   if abs(vote)>=.25:threshold=float(self.tau0*np.exp(-self.strength*confidence*vote));reason='supported'
   else:reason='conflict'
  return dict(threshold=threshold,stop=self.stopped(score,threshold),baseline_stop=self.stopped(score,self.tau0),reason=reason,neighbors=sum(len(v) for v in by_episode.values()),support_episodes=count,vote=vote,
   wait_neighbors=sum(y<0 for b in by_episode.values() for w,y in b),request_neighbors=sum(y>0 for b in by_episode.values() for w,y in b))

 def add_takeover_prefix_credit(self,episode_id,prefix_features,prefix_scores,prefix_steps,expert_loss,expert_valid,regression=None,regression_bound=None,progress_supported=False):
  f=first_boundary_feedback(expert_loss,expert_valid,self.H,self.delta,self.q_loss)
  corrective=progress_supported and regression is not None and regression_bound is not None and regression>regression_bound
  if self.censored_wait and not corrective and f['event'] is None and f['observed_full_windows']>=self.delta:
   if episode_id in self.memory_episodes:raise ValueError('duplicate episode')
   self.memory.append(dict(episode=episode_id,offset=int(prefix_steps[-1]),z=self.normalize(prefix_features[-1]),score=float(prefix_scores[-1]),request=False));self.memory_episodes.add(episode_id)
   return dict(reason='censored_observed_low_error_block_wait',event=None,credit_step=int(prefix_steps[-1]),request=False)
  return super().add_takeover_prefix_credit(episode_id,prefix_features,prefix_scores,prefix_steps,expert_loss,expert_valid,regression,regression_bound,progress_supported)

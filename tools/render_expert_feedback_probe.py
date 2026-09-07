"""Make real signal figures; figures are diagnostics on archived expert paths."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('artifacts/expert_feedback_pca_20260907')

def main():
 a=json.loads((ROOT/'feedback_probe_results.json').read_text());v=a['variants']['point_document'];progress=json.loads((ROOT/'support_progress_diagnostics.json').read_text())['progress'];tau=a['original_pca']['threshold'];q=a['calibrations']['point']['q_loss']
 changed=[e for e in v['episodes'] if e['old_first_on_expert_states']!=e['new_first_on_expert_states']]
 selected=changed[:2];fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
 for row,e in enumerate(selected):
  signal=next(s for s in a['episode_signals'] if s['episode']==e['episode']);ds=[d for d in v['decisions'] if d['episode']==e['episode']];t=[d['offset'] for d in ds]
  axes[row,0].plot(t,[d['score'] for d in ds],'o-',label='Original PCA residual');axes[row,0].axhline(tau,color='black',ls='--',label='Fixed threshold')
  axes[row,0].step(t,[d['threshold'] for d in ds],where='post',color='#c46b25',label='Past-feedback local threshold');axes[row,0].set(title=e['episode'],xlabel='Expert-path offset',ylabel='PCA score');axes[row,0].legend(fontsize=8)
  values=np.array(signal['loss'])/q;full=np.array(signal['valid'])==10;values[~full]=np.nan
  axes[row,1].plot(np.arange(signal['length']),values);axes[row,1].axhline(1,color='black',ls='--');axes[row,1].set(title='Frozen native loss: complete target windows only',xlabel='Expert-path offset',ylabel='Loss / ID q95')
  if not full.all():axes[row,1].axvspan(int(np.flatnonzero(~full)[0]),signal['length']-1,color='grey',alpha=.12,label='Tail: not timing feedback');axes[row,1].legend(fontsize=8)
 fig.suptitle('Actual frozen-model signals; not autonomous closed-loop takeover results',fontsize=11);fig.savefig(ROOT/'actual_threshold_changes.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
 groups=['expert_immediate','expert_post_grasp','expert_post_lift','expert_failure_recovery','expert_internal_pca','expert_diffdagger']
 for group,color in [('expert_immediate','#3588b9'),('expert_failure_recovery','#c96743')]:
  examples=[r for r in progress['rows'] if r['group']==group]
  for e in examples:axes[0].plot(np.arange(len(e['matched_phase'])),e['matched_phase'],color=color,alpha=.35)
  axes[0].plot([],[],color=color,label=group.removeprefix('expert_'))
 axes[0].set(xlabel='Expert-path offset',ylabel='Nearest-ID phase',title='ID-only progress proxy (may jump between nearby states)');axes[0].legend()
 values=[progress['corrected_above_counts'][g] for g in groups];supported=[progress['regression_with_reference_support_counts'][g] for g in groups]
 axes[1].bar(np.arange(6)-.17,values,width=.34,color='#b97942',label='Regression flag');axes[1].bar(np.arange(6)+.17,supported,width=.34,color='#4a8c91',label='+ ID reference support');axes[1].legend(fontsize=8)
 axes[1].set_xticks(np.arange(6),[g.removeprefix('expert_') for g in groups],rotation=25,ha='right');axes[1].set(ylim=(0,8),ylabel='Flagged / 8 episodes',title='Initial regression exceeds ID pseudo-takeover q95')
 fig.savefig(ROOT/'id_progress_feedback.png',dpi=150);plt.close(fig)
 print('REAL_SIGNAL_FIGURES_COMPLETE')

if __name__=='__main__':main()

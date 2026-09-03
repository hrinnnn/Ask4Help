#!/usr/bin/env python3
"""Read-only, CPU-only export of existing utility summaries and evidence counts."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT5090='/data/zhaozhixuan/Ask4Help-airplane-5090/results/'
ROOTH20='/mnt/data/ask4help/results/'
COHORTS={
 'drawer_probe_live':('5090','/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_grasp_timing_sweep_v1_direct_oracle_adaptive_retry1/ood20_probe'),
 'fixedgrid':('5090',ROOT5090+'xvla_fixedgrid_taskpolicy_knee_v1/stage_b_evaluation_v1'),
 'ycb_object':('5090',ROOT5090+'object_variation_pick_single_ycb_v1/diagnostic_low_threshold_005_v1_retry3/final_evaluation'),
 'xvla_stack_gates':('5090',ROOT5090+'xvla_stackcube_v1/temporal_mask_v2/four_group_dagger_bridge_pca_v3/eval_ckpt2500_100id100ood_h150_v1'),
 'xvla_plane_gates':('5090',ROOT5090+'xvla_airplane_v1/ood_dagger_id_ood_alternating_centered_oracle_v3/eval_ckpt5000_100id100ood_h150_v1'),
 'stack_stage2_timing':('h20',ROOTH20+'xvla_stackcube_v1/temporal_mask_v2/stackcube_target_ood_timing_v1_retry2/final_evaluation_retry2'),
 'stack_stage2_gates':('h20',ROOTH20+'xvla_stackcube_v1/temporal_mask_v2/stage2_data_selection_pca_diff_v1_repair1_matched_budget_434_retry2/final_evaluation'),
 'pi05_plane_v1':('h20',ROOTH20+'pick_single_ycb_airplane/four_group_dagger_v1/formal_v3/eval_step2500_100id100ood_horizon250'),
 'pi05_plane_v2':('h20',ROOTH20+'pick_single_ycb_airplane/four_group_dagger_v1/formal_v3/eval_training_v2_step2500_100id100ood_horizon250'),
 'openvla_plane':('h20',ROOTH20+'pick_single_ycb_airplane/openvla_original_lora_r32_v1/gated_dagger_v1/formal_v1/eval_step10000_50ood_v1'),
 'pi05_stack_knn':('h20',ROOTH20+'stackcube_gated_dagger/full_v3/temporal_mask_original_id_norm_5k_retry1/bridge_knn/eval_ood_step2000_100_live'),
 'pi05_stack_offline':('h20',ROOTH20+'stackcube_gated_dagger/full_v3/temporal_mask_original_id_norm_5k_retry1/offline_oracle/eval_ood_step2000_100_live'),
 'pi05_stack_diff':('h20',ROOTH20+'stackcube_gated_dagger/full_v5/diffdagger_successful_experts_100_from7000_retry2/eval_ood_step2000_100_live'),
 'pi05_stack_recovery':('h20',ROOTH20+'stackcube_gated_dagger/full_v5/failure_recovery_successful_experts_100_from7000_retry1/eval_ood_step2000_100_live'),
}
HOSTS={'5090':('12001','zhaozhixuan@111.198.58.150'),'h20':('1012','root@39.101.70.188')}
REMOTE=r'''
import json,sys
from pathlib import Path
out={}
for cohort,root in requests.items():
 p=Path(root);files=[]
 for f in sorted(p.rglob('summary.json')) if p.exists() else []:
  a=json.load(open(f));rows=a.get('rows',a.get('results',[]));n=a.get('episodes',a.get('num_episodes'))
  if isinstance(n,list):n=len(n)
  check={'declared_episodes':n,'row_count':len(rows) if isinstance(rows,list) else None}
  if rows and isinstance(rows,list) and isinstance(rows[0],dict):
   check['row_keys']=list(rows[0])
   for k in ['success','strict_success','ever_grasped','grasped_once']:
    if all(k in r for r in rows):check[k+'_recount']=sum(bool(r[k]) for r in rows)
  check['row_denominator_matches']=n==len(rows) if isinstance(n,int) and isinstance(rows,list) else None
  # Count local episode artifacts, not a second claim copied from summary.
  for directory,pattern in [('videos','*.mp4'),('actions','*.npy'),('raw_actions','*.npy')]:
   q=f.parent/directory
   if q.exists():check[directory+'_files']=len(list(q.glob(pattern)))
  files.append({'path':str(f),'relative_path':str(f.relative_to(p)),'summary':a,'audit':check})
 out[cohort]={'root':root,'exists':p.exists(),'files':files}
 print(cohort,len(files),file=sys.stderr,flush=True)
print(json.dumps(out,allow_nan=True))
'''

def main(output,only=None):
 output.mkdir(parents=True,exist_ok=True)
 for host in HOSTS:
  requests={k:p for k,(h,p) in COHORTS.items() if h==host and (only is None or k==only)}
  if not requests:continue
  port,target=HOSTS[host]
  proc=subprocess.run(['ssh','-o','ConnectTimeout=12','-o','BatchMode=yes','-p',port,target,'python3 -'],
   input=('requests='+repr(requests)+'\n'+REMOTE).encode(),stdout=subprocess.PIPE,check=True)
  data=json.loads(proc.stdout)
  destination=output/f'utility_evidence_{host}.json'
  saved=json.loads(destination.read_text()) if destination.exists() else {};saved.update(data)
  destination.write_text(json.dumps(saved,indent=2))
  for cohort,item in data.items():
   print(cohort,'summaries',len(item['files']),'episode summaries',sum(isinstance(v['audit']['declared_episodes'],int) for v in item['files']),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cohort');a=p.parse_args();main(a.output,a.cohort)

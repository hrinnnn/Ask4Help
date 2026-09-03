#!/usr/bin/env python3
"""Document old source schema gaps without inventing object/contact traces."""
import argparse,json,subprocess
from pathlib import Path
from fetch_cross_asset_utility_evidence import HOSTS,ROOT5090,ROOTH20

REMOTE=r'''
import json,sys
from pathlib import Path
out={}
for cohort,methods in requests.items():
 rows=[]
 for method,collection in methods.items():
  p=Path(collection);raw=p/'raw_archive';record={'method':method,'collection':collection,'exists':p.exists()}
  record['raw_subdirectories']=[x.name for x in raw.iterdir() if x.is_dir()] if raw.exists() else []
  record['task_states_directory_present']=(raw/'task_states').exists()
  for name in ['summary.json','collection_provenance.json']:
   f=p/name
   if f.exists():record[name]=json.load(open(f))
  for name in ['episodes.jsonl','attempts.jsonl','training_episodes.jsonl']:
   f=p/name
   if f.exists():
    rs=[json.loads(s) for s in f.read_text().splitlines() if s.strip()];record[name]={'count':len(rs),'first_row_keys':list(rs[0]) if rs else [],'has_task_states_field':any('task_states' in r for r in rs)}
    if name=='training_episodes.jsonl':record['collected_expert_points']=sum(r.get('expert_action_steps',0) for r in rs)
  record['metric_status']='MISSING_DYNAMIC_OBJECT_AND_CONTACT_EVIDENCE' if not record['task_states_directory_present'] else 'NEEDS_NUMERIC_SOURCE_RECONCILIATION'
  rows.append(record)
 out[cohort]=rows
if ycb:
 p=Path(ycb);comparison=json.load(open(p/'diagnostic_low_threshold_005_v1_retry3/comparison.json'));out['ycb_budget']=comparison['matched_budget'];out['ycb_diagnostic_threshold']=comparison['diff_threshold_override']
 samples=[]
 for method,collection in requests['ycb_object'].items():
  q=Path(collection);accepted=[json.loads(s) for s in (q/'attempts.jsonl').read_text().splitlines() if s.strip()];accepted=[r for r in accepted if r.get('accepted')]
  chosen=set(comparison['matched_budget']['selected_source_episode_indices'][method])
  for i,r in enumerate(accepted):
   if i in chosen and r['split']=='ood':
    videos=list((q/'accepted_suffix_videos').glob(f'*seed_{r["seed"]:06d}.mp4'))
    samples.append(dict(method=method,source_episode_index=i,seed=r['seed'],expert_start=r['expert_start_step'],expert_points=r['expert_action_steps'],video=str(videos[0]) if videos else None,metric_status='UNSCORED_MISSING_DYNAMIC_STATE'))
    break
 out['ycb_selected_examples']=samples
print(json.dumps(out,allow_nan=True))
'''

def main(root):
 s=ROOT5090+'xvla_stackcube_v1/temporal_mask_v2/four_group_dagger_bridge_pca_v3/collections/'
 a=ROOT5090+'xvla_airplane_v1/ood_dagger_id_ood_alternating_centered_oracle_v3/collections/'
 y=ROOT5090+'object_variation_pick_single_ycb_v1/'
 requests5090={'xvla_stack_gates':{m:s+m for m in ['offline_oracle','vlm_bridge_pca','diffdagger','failure_recovery']},
  'xvla_plane_gates':{m:a+m for m in ['offline_oracle','vlm_pool_pca','diffdagger','failure_recovery']},
  'ycb_object':{'bridge_pca':y+'collections_v1/bridge_pca_retry1','diffdagger':y+'collections_diagnostic_v1/diffdagger_low_threshold_005_retry1',
   'failure_recovery':y+'collections_v1/failure_recovery','offline_oracle':y+'collections_v1/offline_oracle'}}
 p=ROOTH20+'pick_single_ycb_airplane/four_group_dagger_v1/formal_v3/'
 v=ROOTH20+'pick_single_ycb_airplane/openvla_original_lora_r32_v1/gated_dagger_v1/formal_v1/collections/'
 requestsH20={'pi05_plane':{m:p+folder for m,folder in [('bridge_pca','bridge_pca_collection'),('offline_oracle','offline_oracle_collection'),('diffdagger','diffdagger_collection_gatecal_retry1'),('failure_recovery','failure_recovery_collection')]},
  'openvla_plane':{'offline_oracle':v+'offline_oracle','pca':v+'siglip_pca'},
  'pi05_stack':{'knn':ROOTH20+'stackcube_gated_dagger/full_v1/bridge_knn_successful_experts_100',
   'diffdagger':ROOTH20+'stackcube_gated_dagger/full_v5/diffdagger_successful_experts_100_from7000_retry2/collection',
   'failure_recovery':ROOTH20+'stackcube_gated_dagger/full_v5/failure_recovery_successful_experts_100_from7000_retry1'}}
 result={}
 for host,requests in [('5090',requests5090),('h20',requestsH20)]:
  port,target=HOSTS[host]
  script='requests='+repr(requests)+'\nycb='+repr(y if host=='5090' else None)+'\n'+REMOTE
  p=subprocess.run(['ssh','-o','ConnectTimeout=12','-o','BatchMode=yes','-p',port,target,'python3 -'],input=script.encode(),stdout=subprocess.PIPE,check=True)
  result[host]=json.loads(p.stdout)
 (root/'inputs/source_schema_inventory.json').write_text(json.dumps(result,indent=2))
 print('SOURCE_INVENTORY_COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)

"""ID-only calibration for the explicitly separate gripper-commitment candidate."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pi05_timing_feedback import gripper_commitment_opening


def main(args):
    cal=json.loads((args.calibration/'calibration.json').read_text())
    demos=json.loads((args.calibration/'demo_progress.json').read_text())
    data_root=Path(cal['source_dataset']);scores=[];per_episode=[]
    for demo in demos:
        episode=demo['episode']
        data=pq.read_table(data_root/f'data/chunk-000/episode_{episode:06d}.parquet',columns=['state','actions']).to_pylist()
        actions=np.asarray([r['actions'] for r in data]);states=np.asarray([r['state'] for r in data]);width=states[:,-2:].sum(1)
        local=[]
        for k in range(5,len(data)-5):
            score=gripper_commitment_opening(actions[k-5:k,-1],actions[k:k+5,-1],width[k],width[k+5])
            local.append(score if demo['errors'][k]>cal['error_reference'] else 0.)
        scores.extend(local);per_episode.append({'episode':episode,'joint_scores_m':local})
    empirical=float(np.quantile(scores,.95,method='higher'))
    result={'rule':'gripper_commitment_v2_candidate','opening_reference_m':max(.01,empirical),
            'empirical_ID_q95_m':empirical,'minimum_opening_m':.01,'minimum_rationale':'one pre-existing gripper-normalization unit (10 mm), frozen before new validation',
            'ID_calibration':str(args.calibration),'ID_episode_count':len(demos),'source_dataset':str(data_root),
            'reference_history':'nominal ID expert closing commands and observed opening, jointly gated by existing free-policy discrepancy reference',
            'scores':per_episode,'claim':'candidate measurement repair, not a validated performance gain'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='scores'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--calibration',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())

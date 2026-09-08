"""Common whole-suffix budget for fixed, original hard, and soft feedback."""
import argparse,json
from pathlib import Path
from score_pi05_feedback_tasr import budget_selection


def run(a):
    report={}
    for task in ['grasp','goal']:
        soft=json.loads((a.soft_root/task/'pilot_TASR.json').read_text())
        hard=json.loads((a.hard_root/task/'pilot_TASR.json').read_text())
        assert soft.get('frozen_limits',soft.get('limits'))==hard.get('frozen_limits',hard.get('limits'))
        assert soft['rows']['fixed']==hard['rows']['fixed']
        arms={'fixed':soft['rows']['fixed'],'hard_feedback':hard['rows']['feedback'],'soft_feedback':soft['rows']['feedback']}
        budget,selected=budget_selection(arms);results={}
        for name,rows in arms.items():
            chosen=[r for r in rows if r['episode'] in selected[name]]
            assert sum(r['expert_points'] for r in chosen)==budget
            assert all(r['compatible_points'] is not None for r in chosen)
            counts={str(f):sum(r['compatible_points'][str(f)] for r in chosen) for f in [1,2,3]}
            results[name]={'budget':budget,'selected_episodes':selected[name],
                'OOD_points':sum(r['expert_points'] for r in chosen if r['split']=='ood'),
                'compatible_points':counts,'TASR':{f:n/budget if budget else None for f,n in counts.items()}}
        report[task]={'matched_three_arm_results':results,
            'hard_timing':json.loads((a.hard_root/task/'paired_timing_report.json').read_text())['summary'],
            'soft_timing':json.loads((a.soft_root/task/'paired_timing_report.json').read_text())['summary']}
    result={'status':'THREE_ARM_DIAGNOSTIC_COMPLETE','results':report,
            'scope':'Same new seeds and40 raw episodes per arm; same fixed controls. No post-SFT SR claim.'}
    (a.hard_root/'three_arm_TASR.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--soft-root',type=Path,required=True);p.add_argument('--hard-root',type=Path,required=True)
    run(p.parse_args())

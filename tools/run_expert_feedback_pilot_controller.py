"""Durable sequential controller for the approved small collection pilot."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/data/ask4help/results/expert_feedback_closedloop_pilot_v1/pilot384')
CODE = Path('/tmp/expert_feedback_pca_probe_v1/code/pilot')

def main(args):
    ROOT=args.root
    ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    state = dict(controller_pid=os.getpid(), started=started, completed=[],
                 collector_protocol='pilot384_v1', seed_offset=args.seed_offset,
                 current_stage='launch', next_stage='independent_audit')
    def save():
        (ROOT/'pipeline_state.json').write_text(json.dumps(state, indent=2))
    save()
    for arm in args.arms:
        output = ROOT/arm
        if (output/'PILOT_COMPLETE.json').exists():
            summary = json.loads((output/'summary.json').read_text())
            assert summary['episodes'] == 20 and len(summary['rows']) == 20
            state['completed'].append(arm)
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f'Partial artifacts preserved; explicit retry root required: {output}')
        command = [sys.executable, str(CODE/'probe_expert_feedback_closedloop.py'),
                   '--arm', arm, '--output', str(output), '--episodes', '20', '--seed-offset', str(args.seed_offset),
                   '--calibration', '/tmp/expert_feedback_pca_probe_v1/gate_calibration_free5.json',
                   '--pca', '/mnt/data/ask4help/results/expert_feedback_pca_probe_v1/original_pca_bridge.npz']
        log_path = ROOT/f'{arm}.log'
        with log_path.open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            state.update(current_stage=arm, child_pid=child.pid, command=command, log=str(log_path))
            save()
            rc = child.wait()
        if rc:
            state.update(current_stage='PILOT_FAILED', returncode=rc)
            save()
            (ROOT/'PILOT_FAILED.json').write_text(json.dumps(state, indent=2))
            return rc
        summary = json.loads((output/'summary.json').read_text())
        assert summary['episodes'] == 20 and len(summary['rows']) == 20
        assert sum(r['expert_actions'] for r in summary['rows']) == summary['expert_actions']
        state['completed'].append(arm)
        save()
    state.update(current_stage='collection_complete_pending_independent_audit', child_pid=None,
                 elapsed=time.time()-started)
    save()
    (ROOT/'COLLECTION_COMPLETE.json').write_text(json.dumps(state, indent=2))
    return 0

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--seed-offset',type=int,default=0)
    parser.add_argument('--arms',nargs='+',choices=['fixed','adaptive_current','adaptive_prefix'],default=['fixed','adaptive_current','adaptive_prefix'])
    raise SystemExit(main(parser.parse_args()))

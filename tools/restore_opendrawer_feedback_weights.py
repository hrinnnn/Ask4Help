"""Durable scoped SCP of the existing OpenDrawer checkpoint, not a model download."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main(args):
    cfg=json.loads(args.manifest.read_text())['opendrawer_restore']
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.destination_file or cfg['target_root']+'/full_weights.pt'
    source_path=args.source_file or cfg['source_weights']
    expected=args.expected_bytes or cfg['expected_weight_bytes']
    source='scp://zhaozhixuan@111.198.58.150:12001/'+source_path
    destination='scp://root@39.101.70.188:1012/'+target
    command=['scp','-3','-o','BatchMode=yes','-o','ConnectTimeout=10',source,destination]
    def remote_size():
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','1012','root@39.101.70.188',
                               'stat','-c','%s',target],text=True,capture_output=True,timeout=20)
        return int(result.stdout.strip()) if result.returncode==0 else None
    initial=remote_size()
    if initial is not None:raise RuntimeError(f'Target exists ({initial} bytes); preserve and inspect before retry')
    state={'controller_pid':os.getpid(),'stage':'asset_transfer','source':source_path,
           'destination':target,'expected_bytes':expected,'started':time.time()}
    def save():(args.output/'transfer_state.json').write_text(json.dumps(state,indent=2))
    with (args.output/'scp.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['child_pid']=child.pid;save()
        while child.poll() is None:
            time.sleep(30)
            try:state['observed_destination_bytes']=remote_size()
            except subprocess.TimeoutExpired:state['last_size_probe']='observation_timeout_not_transfer_failure'
            state['elapsed']=time.time()-state['started'];save()
    state['returncode']=child.returncode;state['observed_destination_bytes']=remote_size()
    valid=child.returncode==0 and state['observed_destination_bytes']==expected
    state['stage']='asset_copy_complete_contents_not_yet_used' if valid else 'transfer_requires_inspection'
    state['finished']=time.time();save()
    if valid:(args.output/('ASSET_TRANSFER_COMPLETE.json' if args.source_file else 'WEIGHTS_TRANSFER_COMPLETE.json')).write_text(json.dumps(state,indent=2))
    return 0 if valid else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-file');parser.add_argument('--destination-file');parser.add_argument('--expected-bytes',type=int)
    args=parser.parse_args();raise SystemExit(main(args))

"""Persistent transfer-to-extraction transition for the original ID remainder."""
import argparse,json,subprocess,sys,os,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--remote-code',required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
source='/data/zhaozhixuan/Ask4Help-open-drawer/results/pi05_feedback_remaining_ID_20260909/remaining_ID.tar'
inputs='/mnt/data/ask4help/datasets/pi05_opendrawer_feedback_calibration_20260909'
state={'pid':os.getpid(),'stage':'transfer_remaining_original_ID','started':time.time(),'source':source,'inputs':inputs}
def save():(args.output/'controller_state.json').write_text(json.dumps(state,indent=2))
save()
cmd=[sys.executable,str(Path(__file__).with_name('restore_opendrawer_feedback_weights.py')),'--manifest',str(args.manifest),'--output',str(args.output/'transfer'),'--source-file',source,'--destination-file',inputs+'/remaining_ID.tar','--expected-bytes','4606218240']
rc=subprocess.call(cmd)
if rc:state['stage']='transfer_failed_requires_review';state['returncode']=rc;save();raise SystemExit(rc)
state['stage']='extract_and_verify_original_ID';save()
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','1012','root@39.101.70.188','/root/Ask4Help-online-awbc/RLinf/.venv/bin/python',args.remote_code+'/tools/finalize_opendrawer_id_restore.py','--inputs',inputs]
rc=subprocess.call(cmd);state.update(stage='full_ID_restore_complete' if rc==0 else 'extraction_requires_review',returncode=rc,finished=time.time());save();raise SystemExit(rc)

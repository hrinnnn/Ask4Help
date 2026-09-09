"""Hand controller ownership over after its current child finishes; never stop the collector."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def main(a):
    statefile=a.root/'controller_state.json'
    old=json.loads(statefile.read_text());pid=old['controller_pid']
    cmdline=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().replace('\0',' ')
    assert 'run_pi05_feedback_small30.py' in cmdline and str(a.root) in cmdline
    os.kill(pid,signal.SIGSTOP)
    old=json.loads(statefile.read_text());child=old['jobs'][-1]['pid']
    record={'stage':'waiting_current_child','migration_pid':os.getpid(),'old_controller':pid,'preserved_child':child,'training':False}
    path=a.root/'parallel_migration_state.json'
    def save():path.write_text(json.dumps(record,indent=2))
    save()
    while True:
        status=Path('/proc')/str(child)/'status'
        if not status.exists():break
        line=next(x for x in status.read_text().splitlines() if x.startswith('State:'))
        if 'Z' in line:break
        time.sleep(10)
    # The old controller is paused at wait(); its data child has finished.
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    for _ in range(30):
        p=Path('/proc')/str(pid)/'status'
        if not p.exists() or 'State:\tZ' in p.read_text():break
        time.sleep(1)
    else:raise RuntimeError('old controller still alive; do not create a second owner')
    cmd=[sys.executable,str(a.code/'tools/run_pi05_feedback_small30.py'),'--code',str(a.code),
         '--plan',str(a.plan),'--root',str(a.root),'--logs',str(a.logs),'--task',a.task,
         '--external-sensitive-root',str(a.parallel_root)]
    with (a.logs/'parallel_main_controller.log').open('a') as f:
        p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record.update(stage='new_controller_started',controller_pid=p.pid,command=cmd);save()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['code','plan','root','logs','parallel-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--task',required=True);main(p.parse_args())

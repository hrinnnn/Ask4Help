"""Independent video-container inventory for completed night collections."""
import json,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path('/mnt/data/ask4help/results/expert_feedback_overnight_v2')
stages={'development':240,'directional':80,'validation':240,'bounded_development':120,'bounded_validation':180,'longstream':180}
tasks=[]
for stage,n in stages.items():
 rows=list((root/stage).glob('seed_*/*/episode_*/result.json'));assert len(rows)==n
 for path in rows:
  row=json.loads(path.read_text());tasks.append((path.parent/'video.mp4',row['total_actions']))
def inspect(item):
 path,n=item;assert path.stat().st_size>0
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames','-of','json',str(path)]))['streams'][0]
 assert (d['width'],d['height'],int(d['nb_frames']))==(768,384,n),(path,d,n)
 return n
with ThreadPoolExecutor(max_workers=4) as pool:frames=list(pool.map(inspect,tasks))
result=dict(status='VIDEO_INVENTORY_PASS',videos=len(frames),total_recorded_frames=sum(frames),stages=stages,scope='ffprobe dimensions and declared frame counts match action lengths; representative RGB manually inspected, not full semantic labeling of every frame')
(root/'VIDEO_INVENTORY_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

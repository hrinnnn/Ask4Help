"""Show actual source frames with old versus OOD-gated target labels."""
import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path('artifacts/pi05_tasr_ood_gap_20260906');OLD=Path('artifacts/pi05_table_tasr_20260903')
COLORS={'excluded_id':'#aeb5bd','non_target':'#aeb5bd','target_compatible':'#279b70','target_mismatch':'#d2574d'}

def main():
 out=ROOT/'videos';out.mkdir(exist_ok=True);sources=ROOT/'source_videos';sources.mkdir(exist_ok=True);audits=[]
 font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',17);small=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',13)
 for task in ['stackcube','airplane','object']:
  old=json.loads((OLD/f'tasr_{task}.json').read_text());new=json.loads((ROOT/f'tasr_{task}.json').read_text())
  for split in ['id','ood']:
   method='diffdagger' if task=='object' and split=='id' else 'offline_oracle'
   candidates=[r for r in old['rows'] if r['selected'] and r['split']==split and r['method']==method and r['compatible_points']['3']>0]
   base=min(candidates,key=lambda r:r['source_episode_index']);row=next(r for r in new['rows'] if r['method']==method and r['source_episode_index']==base['source_episode_index'])
   remote=base['video'] or base['source_actions'].replace('/actions/','/videos/').replace('.npy','.mp4')
   host,port=('root@39.101.70.188','1012') if task!='object' else ('zhaozhixuan@111.198.58.150','12001')
   source=sources/f'{task}_{method}_{split}.mp4'
   if not source.exists():subprocess.run(['scp','-q','-P',port,'-o','ConnectTimeout=12','-o','BatchMode=yes',host+':'+remote,str(source)],check=True)
   raw=np.load(base['arrays']);count=len(raw['actions']);n=base['expert_points'];start=base['expert_start']
   decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(source),'-vf','scale=720:360:force_original_aspect_ratio=decrease,pad=720:360:(ow-iw)/2:(oh-ih)/2','-f','rawvideo','-pix_fmt','rgb24','-'])
   frames=np.frombuffer(decoded,dtype=np.uint8).reshape(-1,360,720,3)
   suffix='replayed_suffix_videos' in remote
   assert len(frames)==(n if suffix else count),(remote,len(frames),n,count)
   dest=out/f'{task}_{method}_{split}_old_vs_ood_gap.mp4'
   enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','720x650','-r','10','-i','-','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
   for i in range(n):
    im=Image.new('RGB',(720,650),'white');d=ImageDraw.Draw(im);im.paste(Image.fromarray(frames[i if suffix else start+i]),(0,70))
    d.text((12,10),f'{task} | {method} | {split.upper()} | seed {base["seed"]}',font=font,fill='black')
    d.text((12,39),f'Expert point {i}/{n} | original source frame {i if suffix else start+i}',font=small,fill='black')
    for y,title,r in [(448,'OLD',base),(530,'OOD GAP',row)]:
     d.text((12,y),f'{title}: episode score={r["Q"]["3"]:.4f} | {r["labels"]["3"][i]}',font=small,fill='black')
     for j,label in enumerate(r['labels']['3']):d.rectangle((int(12+696*j/n),y+24,int(12+696*(j+1)/n)+1,y+44),fill=COLORS[label])
     x=12+696*i/n;d.line((x,y+20,x,y+48),fill='black',width=2)
    d.text((12,611),'ID: numerator 0; ALL original expert points stay in denominator.',font=small,fill='black')
    enc.stdin.write(im.tobytes())
    if i==base['blocks']['close'][0]:im.save(dest.with_suffix('.jpg'))
   enc.stdin.close();assert enc.wait()==0
   probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(dest)]))
   assert int(probe['streams'][0]['nb_frames'])==n
   audits.append(dict(task=task,method=method,split=split,seed=base['seed'],frames=n,source=remote,source_semantics='pre-action; records[:-1]',output=str(dest.resolve())))
   print('VIDEO',dest,flush=True)
 (ROOT/'video_audit.json').write_text(json.dumps(audits,indent=2))
 with (ROOT/'结果与目标片段核对.md').open('a') as f:
  f.write('\n## ID/OOD同帧标记核对视频\n\n')
  for a in audits:f.write(f"- [{a['task']} {a['method']} {a['split']}]({a['output']})\n")

if __name__=='__main__':main()

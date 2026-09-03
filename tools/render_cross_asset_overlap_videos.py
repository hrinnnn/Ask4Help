#!/usr/bin/env python3
"""Render existing RGB with frozen x1/x2/x3 membership and cohort utility."""
import argparse,copy,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from render_open_drawer_ed_audit import frames

COLORS={'non_target':'#a9b1ba','target_compatible':'#23966d','target_mismatch':'#d55749','policy_prefix':'#417db0'}

def encode_video(out,source,base,variants,group,task,fps):
 raw=np.load(base['arrays']);video,meta=frames(source,len(raw['actions'])+1)
 assert len(video)==len(raw['actions'])
 meta['state_step_offset']=0;meta['source_semantics']='records[:-1], pre-action observation'
 start=base['expert_start'];n=base['expert_anchors'];first=max(0,start-10);last=start+n
 font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',18);small=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',13)
 enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','800x850','-r',str(fps),'-i','-',
  '-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
 for t in range(first,last):
  i=t-start;canvas=Image.new('RGB',(800,850),'white');d=ImageDraw.Draw(canvas)
  d.text((14,10),f"{task} | {group['condition']} | seed {base['seed']} | step {t}",font=font,fill='black')
  d.text((14,39),f"Group OOD utility={group['OOD_SR']:.4f} | {group['evaluation']}",font=small,fill='black')
  canvas.paste(Image.fromarray(video[t]).resize((800,480)),(0,65))
  for k,(f,r) in enumerate(variants.items()):
   y=558+k*79;lab='policy_prefix' if i<0 else r['labels'][i]
   d.text((14,y),f"x{f} | group Q={group['Q'][str(f)]:.4f} | this trajectory Q={r['Q']:.4f} | {lab}",font=small,fill=COLORS[lab])
   for j,label in enumerate(r['labels']):
    x0=15+765*j/n;x1=15+765*(j+1)/n;d.rectangle((int(x0),y+27,int(x1)+1,y+47),fill=COLORS[label])
   x=15+765*max(0,i)/n;d.line((x,y+21,x,y+52),fill='black',width=2)
   phase=next((p for p,(a,b) in r['blocks'].items() if a<=i<b),None)
   if phase:d.text((14,y+54),f"{phase}: error={r['distance'][i]:.4f} / radius={r['thresholds'][phase]:.4f}",font=small,fill='black')
  d.text((14,803),'Green=target-compatible; red=mismatch (not necessarily recovery); grey=non-target.',font=small,fill='black')
  d.text((14,825),'Offline, post-hoc sensitivity. Same matching/contact/data; only radius changes.',font=small,fill='black')
  enc.stdin.write(canvas.tobytes())
  if i in [0,n//3,2*n//3,n-1]:canvas.save(out.with_name(out.stem+f'_offset{i}.jpg'),quality=89)
 enc.stdin.close();assert enc.wait()==0
 return dict(output=str(out.resolve()),frames=last-first,fps=fps,source=meta,first_step=first,last_step_exclusive=last,seed=base['seed'])

def stage2_videos(root):
 out=root/'videos';out.mkdir(exist_ok=True);sources=root/'source_videos';sources.mkdir(exist_ok=True)
 analysis=json.loads((root/'stage2_overlap.json').read_text());comparison=json.loads((root/'comparison.json').read_text());audit=[]
 timing=[r for r in analysis['rows'] if r['cohort']=='stack_stage2_timing' and r['selected']]
 shared=set.intersection(*[{r['seed'] for r in timing if r['method']==m} for m in ['immediate','post_grasp','post_lift','failure_recovery']]);paired=min(shared)
 for group in comparison['rows']:
  if group['cohort'] not in ['stack_stage2_timing','stack_stage2_gates']:continue
  pool=[r for r in analysis['rows'] if r['cohort']==group['cohort'] and r['method']==group['condition'] and r['selected'] and r['split']=='stage2_ood']
  row=next(r for r in pool if r['seed']==paired) if group['cohort']=='stack_stage2_timing' else min(pool,key=lambda r:r['seed'])
  source=sources/f"{row['cohort']}_{row['method']}_{row['seed']}.mp4"
  if not source.exists():subprocess.run(['scp','-q','-P','1012','-o','ConnectTimeout=12','-o','BatchMode=yes','root@39.101.70.188:'+row['video'],str(source)],check=True)
  variants={}
  for f in [1,2,3]:
   v=copy.deepcopy(row);v['Q']=row['Q'][str(f)];v['labels']=row['labels_by_factor'][str(f)];v['thresholds']={p:t*f for p,t in row['thresholds'].items()};variants[f]=v
  record=encode_video(out/f"{row['cohort']}_{row['method']}_x1_x2_x3.mp4",source,variants[1],variants,group,'StackCube Stage2',5)
  record.update(cohort=row['cohort'],method=row['method']);audit.append(record);print('STAGE2_VIDEO_COMPLETE',row['method'],flush=True)
 for cohort in ['stack_stage2_timing','stack_stage2_gates']:
  clips=[r for r in audit if r['cohort']==cohort];target=out/f'{cohort}_all_x1_x2_x3.mp4'
  listing=''.join("file 'file:"+r['output']+"'\n" for r in clips)
  subprocess.run(['ffmpeg','-v','error','-y','-protocol_whitelist','file,pipe','-f','concat','-safe','0','-i','pipe:0','-c','copy','-movflags','+faststart',str(target)],input=listing.encode(),check=True)
 (root/'stage2_video_audit.json').write_text(json.dumps(audit,indent=2))

def main(root):
 out=root/'videos';out.mkdir(exist_ok=True);audit=[]
 comparison=json.loads((root/'comparison.json').read_text())
 variants={f:json.loads((root/f'fixedgrid_x{f}/analysis.json').read_text()) for f in [1,2,3]}
 for task in ['stackcube','airplane']:
  result=variants[1][task]
  shared=set.intersection(*[{r['seed'] for r in result['rows'] if r['step']==int(s) and r['selected']} for s in result['summary']]);seed=min(shared)
  for step in map(int,result['summary']):
   rows={f:next(r for r in v[task]['rows'] if r['step']==step and r['seed']==seed) for f,v in variants.items()}
   source=Path('artifacts/stackcube_airplane_target_ratio_20260831/source_videos')/f'{task}_t{step}_seed_{seed}.mp4'
   group=next(r for r in comparison['rows'] if r['cohort']=='fixedgrid_'+task and r['takeover_step']==step)
   record=encode_video(out/f'{task}_t{step}_x1_x2_x3.mp4',source,rows[1],rows,group,task,5 if task=='stackcube' else 10)
   record['task']=task;audit.append(record);print('VIDEO_COMPLETE',task,step,flush=True)
 for task in ['stackcube','airplane']:
  clips=[r for r in audit if r['task']==task];target=out/f'{task}_all_timings_x1_x2_x3.mp4'
  listing=''.join("file 'file:"+r['output']+"'\n" for r in clips)
  subprocess.run(['ffmpeg','-v','error','-y','-protocol_whitelist','file,pipe','-f','concat','-safe','0','-i','pipe:0','-c','copy','-movflags','+faststart',str(target)],input=listing.encode(),check=True)
 (root/'video_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage2',action='store_true');a=p.parse_args();(stage2_videos if a.stage2 else main)(a.root)

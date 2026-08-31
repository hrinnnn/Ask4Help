#!/usr/bin/env python3
"""Per-task colored timing examples, with source pre-action frame alignment."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from analyze_open_drawer_suffix_recovery import pose_channels
from render_open_drawer_ed_audit import download,frames

COLORS={'non_target':'#b8bec5','target_compatible':'#24986c','target_mismatch':'#d95648','policy_prefix':'#427fb0'}


def compile_videos(root,audit):
    outputs=[]
    for task in ['stackcube','airplane']:
        clips=[v for v in audit if v['task']==task]
        listing=''.join("file 'file:"+str(Path(v['output']).resolve())+"'\n" for v in clips)
        target=root/'videos'/f'{task}_all_timings.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-protocol_whitelist','file,pipe','-f','concat','-safe','0','-i','pipe:0',
                        '-c','copy','-movflags','+faststart',str(target)],input=listing.encode(),check=True)
        outputs.append(dict(task=task,output=str(target),frames=sum(v['frames'] for v in clips),clips=[v['output'] for v in clips]))
    (root/'compilations.json').write_text(json.dumps(outputs,indent=2))


def main(root):
    data=json.loads((root/'analysis.json').read_text());out=root/'videos';out.mkdir(exist_ok=True);sources=root/'source_videos';sources.mkdir(exist_ok=True)
    font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',18);small=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',13)
    audit=[]
    for task,result in data.items():
        shared=set.intersection(*[{r['seed'] for r in result['rows'] if r['step']==int(s) and r['selected']} for s in result['summary']])
        seed=min(shared)
        for step in map(int,result['summary']):
            row=next(r for r in result['rows'] if r['step']==step and r['seed']==seed)
            source=sources/f'{task}_t{step}_seed_{seed}.mp4';download(row['video'],source)
            raw=np.load(row['arrays']);n_source=len(raw['actions']);video,meta=frames(source,n_source+1)
            assert len(video)==n_source
            # Verified _build_frames enumerates records[:-1]: frame t is state t,
            # before action t; N frames are NOT an indication of post-action frames.
            meta['state_step_offset']=0;meta['source_semantics']='records[:-1], before-action frames'
            pose=dict(np.load(root/task/f'pose_step_{step}_seed_{seed}.npz'))
            reference=dict(np.load(root/task/f"pose_step_0_seed_{row['reference_seed']}.npz"))
            component=pose_channels(pose,reference)
            fps=5 if task=='stackcube' else 10
            target=out/f'{task}_t{step}_seed_{seed}.mp4'
            encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x760','-r',str(fps),'-i','-',
                '-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
            first=max(0,row['expert_start']-10);last=row['expert_start']+row['expert_anchors']
            for absolute in range(first,last):
                i=absolute-row['expert_start'];canvas=Image.new('RGB',(640,760),'white');draw=ImageDraw.Draw(canvas)
                canvas.paste(Image.fromarray(video[absolute]).resize((640,400)),(0,70))
                label='policy_prefix' if i<0 else row['labels'][i]
                draw.rectangle((5,72,635,468),outline=COLORS[label],width=5)
                draw.text((12,10),f'{task} | takeover {step} | seed {seed}',font=font,fill='black')
                draw.text((12,40),f"Q={row['Q']*100:.1f}% ({row['compatible_anchors']}/{row['expert_anchors']}) | playback {'0.5x' if fps==5 else '1x'}",font=small,fill='black')
                phase=next((p for p,(a,b) in row['blocks'].items() if a<=i<b),'outside_target') if i>=0 else 'policy_prefix'
                draw.text((12,483),f'Absolute step {absolute} | expert offset {i} | {phase}',font=small,fill='black')
                draw.text((12,510),label,font=font,fill=COLORS[label])
                if i>=0 and phase in row['thresholds']:
                    j=row['reference_mapping'][i];p,angle,gap=component[i,j]
                    draw.text((12,540),f"score {row['distance'][i]:.5f} / limit {row['thresholds'][phase]:.5f} | ref {row['reference_seed']}:{j}",font=small,fill='black')
                    draw.text((12,564),f"p {p*100:.2f}cm R {np.rad2deg(angle):.2f}deg gap {gap*1000:.2f}mm held {int(pose['contact'][i])}/{int(reference['contact'][j])}",font=small,fill='black')
                for k,lab in enumerate(row['labels']):
                    x0=15+600*k/row['expert_anchors'];x1=15+600*(k+1)/row['expert_anchors']
                    draw.rectangle((int(x0),600,int(x1)+1,632),fill=COLORS[lab])
                cursor=15+600*max(0,i)/row['expert_anchors'];draw.line((cursor,593,cursor,640),fill='black',width=2)
                draw.text((12,660),'Green counted | Red mismatch | Grey non-target | Blue policy',font=small,fill='black')
                draw.text((12,686),'Target: approach/alignment + stable close. No lift/transport.',font=small,fill='black')
                draw.text((12,712),'Frame: object centre + observed stable-grasp axes; offline.',font=small,fill='black')
                draw.text((12,738),'Diagnostic membership, not SR. Source frames verified pre-action.',font=small,fill='black')
                encoder.stdin.write(canvas.tobytes())
                close=row['blocks']['close'][0];end=row['blocks']['close'][1]
                if i in [0,max(0,close//2),close,end-1,end,min(end+5,row['expert_anchors']-1)]:
                    canvas.save(out/f'{task}_t{step}_seed_{seed}_offset_{i}.jpg',quality=90)
            encoder.stdin.close();assert encoder.wait()==0
            audit.append(dict(task=task,step=step,seed=seed,output=str(target),frames=last-first,fps=fps,source=meta,first_absolute_step=first))
            print('VIDEO_COMPLETE',task,step,seed,flush=True)
    (root/'video_audit.json').write_text(json.dumps(audit,indent=2))
    compile_videos(root,audit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)

#!/usr/bin/env python3
"""Visualize exactly which recorded supervision anchors enter the numerator."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from render_open_drawer_ed_audit import download,frames
from analyze_open_drawer_piecewise_ed import load_episode
from analyze_open_drawer_target_ratio import relative_pose
from analyze_open_drawer_suffix_recovery import pose_channels

COLORS={'non_target':'#b8bec5','target_compatible':'#24986c','target_mismatch':'#d95648'}


def main(root,previous,suffix,seeds=None):
    more=bool(seeds)
    result=json.loads((root/'analysis.json').read_text());out=root/('videos_more' if more else 'videos');out.mkdir(exist_ok=True)
    source=root/'source_videos';source.mkdir(exist_ok=True)
    font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',19)
    small=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',14)
    specs=[('reference_rotation_grasp_only',[78324]),('same_t220_low_vs_high_recovery',[78829,78817])] if not seeds else [
        (f"target_t{next(r['anchor'] for r in result['rows'] if r['seed']==seed)}_seed_{seed}",[seed]) for seed in seeds]
    lookup={r['seed']:r for r in result['rows']};pose_cache={}
    def context(seed):
        if seed not in pose_cache:
            item=load_episode(Path(lookup[seed]['directory']),lookup[seed])
            pose_cache[seed]=(relative_pose(item['pose'],item['rows']),item['rows'])
        return pose_cache[seed]
    audits=[]
    for name,seeds in specs:
        tracks=[]
        for seed in seeds:
            row=next(r for r in result['rows'] if r['seed']==seed)
            candidates=[previous/'source_videos'/f'query_{seed}.mp4',suffix/'source_videos'/f'query_{seed}.mp4']
            path=next((p for p in candidates if p.exists()),source/f'query_{seed}.mp4')
            download(row['video'],path)
            nrows=len(np.load(Path(row['directory'])/'states.npy'))
            video,audit=frames(path,nrows)
            qp,qr=context(seed);rp,rr=context(row['reference_seed'])
            components=pose_channels(qp,rp)
            tracks.append((row,video,audit,components,qr,rr))
        width=640*len(tracks);height=740
        target=out/f'{name}.mp4'
        encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{width}x{height}','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
        duration=max(tr[0]['expert_anchors'] for tr in tracks)
        for i in range(duration):
            canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
            for col,(r,video,audit,components,qr,rr) in enumerate(tracks):
                x=col*640;n=r['expert_anchors'];t=r['takeover']+min(i,n-1)
                frame=Image.fromarray(video[max(0,t-audit['state_step_offset'])]).resize((640,400))
                if i>=n:frame=Image.blend(frame,Image.new('RGB',frame.size,'white'),.5)
                canvas.paste(frame,(x,65))
                draw.text((x+12,10),f"t={r['anchor']} | seed {r['seed']} | Q={100*r['Q']:.1f}%",font=font,fill='black')
                draw.text((x+12,38),f"Counted: {r['compatible_anchors']} / {n} expert anchors",font=small,fill='black')
                label=r['labels'][i] if i<n else 'non_target'
                draw.rectangle((x+5,67,x+635,463),outline=COLORS[label],width=5)
                phase=next((p for p,(a,b) in r['blocks'].items() if a<=t<b),'outside_target') if i<n else 'finished'
                draw.text((x+12,478),f'Expert offset {i} | abs step {t} | {phase}',font=small,fill='black')
                draw.text((x+12,505),label if i<n else 'Finished; final observation held',font=font,fill=COLORS[label])
                if i<n and phase in r['thresholds']:
                    j=r['reference_mapping'][i];d=r['distance'][i];tau=r['thresholds'][phase]
                    position,angle,width_error=components[t,j]
                    draw.text((x+12,535),f"score {d:.3f} / limit {tau:.3f} | ref {r['reference_seed']}:{j}",font=small,fill='black')
                    draw.text((x+12,560),f"p {position*100:.2f}cm  R {np.rad2deg(angle):.2f}deg  gap {width_error*1000:.2f}mm  held {int(qr[t]['object_grasped'])}/{int(rr[j]['object_grasped'])}",font=small,fill='black')
                else:
                    draw.text((x+12,535),'Not evaluated for target similarity: outside target phase.',font=small,fill='black')
                for k,lab in enumerate(r['labels']):
                    left=x+15+600*k/n;right=x+15+600*(k+1)/n
                    draw.rectangle((int(left),595,int(right)+1,629),fill=COLORS[lab])
                cursor=x+15+600*min(i,n-1)/n;draw.line((cursor,588,cursor,637),fill='black',width=2)
                draw.text((x+12,654),'Green: counted | Red: target mismatch | Grey: not target',font=small,fill='black')
                draw.text((x+12,680),'Lift / carry / place remain in denominator, never numerator.',font=small,fill='black')
                draw.text((x+12,708),'Task-stage + object-relative pose/gripper diagnostic; not SR.',font=small,fill='black')
            encoder.stdin.write(canvas.tobytes())
            selected=[0,10,20,30,40,79,100,122,124,140] if len(tracks)==1 else [0,20,32,44,60]
            if i in selected:canvas.save(out/f'{name}_offset_{i}.jpg',quality=91)
        encoder.stdin.close();assert encoder.wait()==0
        audits.append(dict(output=str(target),frames=duration,seeds=seeds,source_audits=[t[2] for t in tracks]))
        print('TARGET_VIDEO_COMPLETE',target,flush=True)
    (root/('video_audit_more.json' if more else 'video_audit.json')).write_text(json.dumps(audits,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--suffix',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+')
    a=p.parse_args();main(a.root,a.previous,a.suffix,a.seeds)

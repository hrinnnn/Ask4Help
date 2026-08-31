#!/usr/bin/env python3
"""Real-clock success/failure video with each displayed rollout's own curves."""
import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_open_drawer_piecewise_ed import load_episode
from analyze_xvla_erd_pose import _quat_multiply, _quat_inverse, _quat_rotvec
from render_open_drawer_ed_audit import frames, download


def main(root):
    result=json.loads((root/'analysis.json').read_text())
    mode=result['modes']['open_end_monotone_dp']
    expert_meta={r['seed']:r for r in map(json.loads,(root/'inputs/formal/anchor_0/accepted_experts.jsonl').read_text().splitlines())}
    policy_meta={r['seed']:r for r in json.loads((root/'inputs/policy20/summary.json').read_text())['rows']}
    output=root/'success_failure_comparison';output.mkdir(exist_ok=True)

    def episode(seed):
        if seed in expert_meta:
            meta=expert_meta[seed]
            directory=root/'inputs/formal/anchor_0/accepted'/f"episode_{meta['episode_index']:06d}"
        else:
            meta=policy_meta[seed]
            directory=root/'inputs/policy20/episodes'/f"episode_{meta['episode_index']:06d}"
        return load_episode(directory,meta)

    tracks=[]
    for seed,color,label in [(78324,'#2376b7','Successful expert'),(78706,'#d64d32','Failed policy')]:
        row=next(r for r in mode['rows'] if r['seed']==seed)
        item=episode(seed);reference=episode(row['reference_seed'])
        mapping=np.array(row['aligned_reference_steps'])
        q=item['pose'];r=reference['pose']
        pos=np.linalg.norm(q['position']-r['position'][mapping],axis=1)*100
        angle=np.linalg.norm(_quat_rotvec(_quat_multiply(q['quaternion'],_quat_inverse(r['quaternion'][mapping]))),axis=1)*180/np.pi
        gap=q['width'].ravel()*1000
        src=root/'source_videos'/f'query_{seed}.mp4';download(row['video'],src)
        video,video_meta=frames(src,len(q['position']))
        tracks.append(dict(seed=seed,color=color,label=label,row=row,item=item,
            curves=[np.array(row['D_over_tau']),pos,angle,gap],frames=video,video_audit=video_meta))

    fig,axes=plt.subplots(2,2,figsize=(12.8,5.6),dpi=100)
    fig.subplots_adjust(left=.06,right=.98,top=.94,bottom=.1,hspace=.5,wspace=.2)
    labels=['Existing D / q92.5 threshold (diagnostic)','TCP position residual (cm)',
            'TCP orientation residual (degrees)','Actual gripper opening (mm)']
    for k,ax in enumerate(axes.flat):
        for tr in tracks:
            ax.plot(tr['curves'][k],color=tr['color'],lw=1.8,label=f"{tr['label']} {tr['seed']}")
        if k==0:ax.axhline(1,color='black',ls='--',lw=.8,label='Old threshold; NOT validated')
        ax.set(xlim=(0,400),xlabel='Environment step (real clock)',title=labels[k])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.15)
        ax.legend(fontsize=7,loc='upper left')
    stream=io.BytesIO();fig.savefig(stream,format='png')
    chart=Image.open(stream).convert('RGB')
    boxes=[ax.get_position().bounds for ax in axes.flat]
    plt.close(fig)
    font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',19)
    small=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',14)
    path=output/'success_78324_vs_failure_78706.mp4'
    encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x1120','-r','10','-i','-','-an','-c:v','libx264','-preset','fast',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    keyframes=[]
    for t in range(1,401):
        canvas=Image.new('RGB',(1280,1120),'white');draw=ImageDraw.Draw(canvas)
        for index,tr in enumerate(tracks):
            row=tr['row'];end=row['steps'];step=min(t,end)
            vi=max(0,step-tr['video_audit']['state_step_offset'])
            frame=Image.fromarray(tr['frames'][vi]).resize((640,400))
            if t>end:
                frame=Image.blend(frame,Image.new('RGB',frame.size,'white'),.45)
            canvas.paste(frame,(index*640,70))
            draw.text((15+index*640,10),f"{tr['label']} | seed {tr['seed']}",font=font,fill=tr['color'])
            status=f"step {t}/{end}" if t<=end else f"SUCCESS ENDED at {end}; final frame held"
            draw.text((15+index*640,39),status,font=small,fill='black')
            event=row['events']['ever_drawer_opened']
            grasp=row['events']['ever_grasped']
            text=f"drawer-open step: {event}; grasp-event: {grasp}"
            draw.text((15+index*640,477),text,font=small,fill=tr['color'])
        canvas.paste(chart,(0,510))
        for x,y,w,h in boxes:
            xx=int((x+w*t/400)*1280)
            draw.line((xx,510+int((1-y-h)*560),xx,510+int((1-y)*560)),fill='#333333',width=2)
        draw.text((15,1075),'Both videos use real steps (10 steps/s). Expert curves STOP when its rollout ends.',font=small,fill='black')
        draw.text((15,1097),'Same OOD condition, different resets. Residuals use each rollout\'s own frozen reference; no self-scoring.',font=small,fill='black')
        encoder.stdin.write(canvas.tobytes())
        if t in [60,100,120,160,300]:
            canvas.save(output/f'comparison_step_{t}.jpg',quality=92)
            keyframes.append(canvas.resize((640,560)))
    encoder.stdin.close();assert encoder.wait()==0
    sheet=Image.new('RGB',(640*len(keyframes),560),'white')
    for i,im in enumerate(keyframes):sheet.paste(im,(i*640,0))
    sheet.save(output/'contact_sheet.jpg',quality=90)
    subprocess.run(['ffmpeg','-v','error','-y','-ss','5','-i',str(path),'-t','11',
                    '-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',
                    str(output/'grasp_window_steps_51_160.mp4')],check=True)
    meta=dict(mode='open_end_monotone_dp; unchanged existing scores',
              playback='real clock; source first frame is state step 1; expert video held after completion, curves never extended',
              trajectories=[dict(seed=tr['seed'],reference_seed=tr['row']['reference_seed'],
                  events=tr['row']['events'],video=tr['video_audit'],curves=[c.tolist() for c in tr['curves']]) for tr in tracks],
              output=str(path))
    (output/'comparison_metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
    print(path,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)

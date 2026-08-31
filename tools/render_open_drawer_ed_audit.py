#!/usr/bin/env python3
"""Render existing videos against saved diagnostic scores; no simulation."""
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


def download(remote, destination):
    if not destination.exists():
        subprocess.run(['scp','-q','-P','12001','-o','ConnectTimeout=12','-o','BatchMode=yes',
                        'zhaozhixuan@111.198.58.150:'+remote,str(destination)],check=True)


def frames(path, expected):
    result = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                             'stream=nb_frames,r_frame_rate,width,height','-of','json',str(path)],
                            capture_output=True,check=True)
    meta=json.loads(result.stdout)['streams'][0]
    data=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-vf',
        'scale=512:320:force_original_aspect_ratio=decrease,pad=512:320:(ow-iw)/2:(oh-ih)/2',
        '-pix_fmt','rgb24','-f','rawvideo','-'],capture_output=True,check=True).stdout
    array=np.frombuffer(data,dtype=np.uint8).reshape(-1,320,512,3)
    assert len(array) in [expected, expected-1], (str(path),len(array),expected)
    # Collectors render after each action; N frames mean index 0 is state step 1.
    offset=int(len(array)==expected-1)
    return array,dict(path=str(path),frame_count=len(array),state_rows=expected,state_step_offset=offset,source=meta)


def main(root, mode_name):
    analysis=json.loads((root/'analysis.json').read_text())
    mode=analysis['modes'][mode_name]
    sources=root/'source_videos'; sources.mkdir(exist_ok=True)
    outputs=root/('videos_'+mode_name);outputs.mkdir(exist_ok=True)
    experts={r['seed']:r for r in map(json.loads,(root/'inputs/formal/anchor_0/accepted_experts.jsonl').read_text().splitlines())}
    font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',16)
    small=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',13)
    audits=[]
    for seed in [78706,78707,78705,78324]:
        row=next(r for r in mode['rows'] if r['seed']==seed)
        peer=experts[row['reference_seed']]
        qp=sources/f'query_{seed}.mp4';rp=sources/f"reference_{peer['seed']}.mp4"
        download(row['video'],qp);download(peer['video'],rp)
        qf,qa=frames(qp,row['steps']+1)
        rf,ra=frames(rp,row['reference_actions']+1)
        fig,ax=plt.subplots(figsize=(10.24,3.1),dpi=100)
        fig.subplots_adjust(left=.075,right=.975,bottom=.17,top=.92)
        for er in mode['rows']:
            if er['group']=='expert_calibration':
                ax.plot(er['D_over_tau'],color='#2676b8',alpha=.3,lw=1)
        ax.plot(row['D_over_tau'],color='#d65b39',lw=2,label=f"Query {seed}")
        ax.plot([],[],color='#2676b8',label='6 calibration expert curves (own clock)')
        ax.axhline(1,color='black',ls='--',lw=1,label='q92.5 threshold')
        if row['Tref'] is not None:
            ax.axvline(row['Tref'],color='#d65b39',ls='--',label=f"Tref={row['Tref']}")
        ax.set(xlim=(0,row['steps']),ylim=(0,max(1.6,max(row['D_over_tau'])*1.1)),xlabel='Environment step',ylabel='D / threshold')
        ax.legend(fontsize=7,loc='upper right')
        stream=io.BytesIO();fig.savefig(stream,format='png');plt.close(fig)
        chart=Image.open(stream).convert('RGB')
        target=outputs/f'audit_seed_{seed}.mp4'
        proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1024x800','-r','10','-i','-','-an','-c:v','libx264','-preset','fast',
            '-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
        key_steps=sorted(set([0,row['events']['ever_drawer_opened'] or 0,
                              row['Tref'] if row['Tref'] is not None else 160,row['steps']]))
        thumbs=[]
        for t in range(row['steps']+1):
            j=row['aligned_reference_steps'][t]
            canvas=Image.new('RGB',(1024,800),'white')
            canvas.paste(Image.fromarray(qf[max(0,t-qa['state_step_offset'])]),(0,72))
            canvas.paste(Image.fromarray(rf[max(0,j-ra['state_step_offset'])]),(512,72))
            canvas.paste(chart,(0,420))
            d=ImageDraw.Draw(canvas)
            label='OOD autonomous' if row['group']=='policy_ood' else 'HELD-OUT SUCCESSFUL EXPERT'
            d.text((14,12),f'{label}: seed {seed}',font=font,fill='black')
            d.text((525,12),f"Reference expert: seed {peer['seed']}",font=font,fill='black')
            d.text((14,39),f'Query step {t} / {row["steps"]}',font=font,fill='black')
            d.text((525,39),f'Aligned reference step {j} (time-warped)',font=small,fill='black')
            d.text((15,396),f"D/tau={row['D_over_tau'][t]:.3f}   Tref={row['Tref']}   Tconfirm={row['Tconfirm']}",font=font,fill='#b23d25')
            x=int(1024*(.075+.9*t/row['steps']))
            d.line((x,445,x,677),fill='#bd2c1b',width=2)
            d.text((15,742),'DIAGNOSTIC: expert-only calibration; no validated optimal timing or downstream SR.',font=small,fill='black')
            d.text((15,766),f'{mode_name}; Blue curves stop at completion; reference panel is time-warped.',font=small,fill='black')
            proc.stdin.write(canvas.tobytes())
            if t in key_steps:
                canvas.save(outputs/f'seed_{seed}_step_{t}.jpg',quality=88)
                thumbs.append(canvas.resize((512,400)))
        proc.stdin.close();assert proc.wait()==0
        sheet=Image.new('RGB',(512*len(thumbs),400),'white')
        for i,thumb in enumerate(thumbs):sheet.paste(thumb,(512*i,0))
        sheet.save(outputs/f'contact_seed_{seed}.jpg',quality=90)
        audits.append(dict(seed=seed,query=qa,reference=ra,output=str(target),selected_steps=key_steps))
        print('VIDEO_COMPLETE',seed,flush=True)
    (root/('video_audit_'+mode_name+'.json')).write_text(json.dumps(audits,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',default='open_end_monotone_dp')
    args=parser.parse_args()
    main(args.root,args.mode)

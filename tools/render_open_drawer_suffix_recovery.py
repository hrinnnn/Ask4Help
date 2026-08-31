#!/usr/bin/env python3
"""Video audit for suffix deformation; source videos and scores stay unchanged."""
import argparse
import io
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from analyze_open_drawer_piecewise_ed import load_episode,read_jsonl
from analyze_open_drawer_suffix_recovery import trim_success,slice_episode,rotations
from render_open_drawer_ed_audit import download,frames


def cumulative(pose):
    R=rotations(pose['quaternion'])
    return np.column_stack([
        np.r_[0,np.cumsum(np.linalg.norm(np.diff(pose['position'],axis=0),axis=1))*100],
        np.r_[0,np.cumsum((R[1:]*R[:-1].inv()).magnitude())*180/np.pi],
        np.r_[0,np.cumsum(abs(np.diff(pose['width'].ravel())))*1000]])


def main(root,previous,selected):
    result=json.loads((root/'analysis.json').read_text())
    formal=previous/'inputs/formal'
    expert_meta={r['seed']:r for r in read_jsonl(formal/'anchor_0/accepted_experts.jsonl')}
    videos=root/'videos';videos.mkdir(exist_ok=True)
    sources=root/'source_videos';sources.mkdir(exist_ok=True)
    font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',18)
    small=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',13)
    audit_path=root/'video_audit.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else []
    audit=[r for r in audit if r['seed'] not in selected]
    for seed in selected:
        row=next(r for r in result['attempts'] if r['seed']==seed)
        meta=next(r for r in read_jsonl(formal/f"anchor_{row['anchor']}"/'raw_attempts.jsonl') if r['seed']==seed)
        item=load_episode(Path(row['directory']),meta)
        rm=expert_meta[row['reference_seed']]
        ref=load_episode(formal/'anchor_0/accepted'/f"episode_{rm['episode_index']:06d}",rm)
        qp=sources/f'query_{seed}.mp4';rp=sources/f"reference_{rm['seed']}.mp4"
        download(row['video'],qp);download(rm['video'],rp)
        qf,qa=frames(qp,len(item['rows']));rf,ra=frames(rp,len(ref['rows']))
        query=slice_episode(trim_success(item),row['takeover'])
        ref=trim_success(ref)
        mapping=np.array(row['reference_mapping']);j0=mapping[0]
        qcum=cumulative(query['pose']);rcum=cumulative(ref['pose']);rcum=rcum-rcum[j0]
        ref_aligned=rcum[mapping]
        n=len(mapping)
        fig,axes=plt.subplots(2,2,figsize=(12.8,5.4),dpi=100)
        fig.subplots_adjust(left=.06,right=.98,bottom=.1,top=.94,wspace=.21,hspace=.5)
        axes[0,0].plot(row['position_cm_timeline'],color='#c65032',label='position error (cm)')
        axes[0,0].plot(row['orientation_deg_timeline'],color='#7959b8',label='orientation error (deg)')
        axes[0,0].set(title='Aligned residuals (different physical units)')
        axes[0,1].plot(query['pose']['width'].ravel()*1000,color='#c65032',label='takeover expert')
        axes[0,1].plot(ref['pose']['width'].ravel()[mapping]*1000,color='#2676b8',label='nominal expert, aligned')
        axes[0,1].set(title='Gripper opening (mm)')
        axes[1,0].plot(qcum[:,1],color='#c65032',label='takeover rotation')
        axes[1,0].plot(ref_aligned[:,1],color='#2676b8',label='nominal rotation')
        axes[1,0].set(title='Cumulative rotation (deg); full turns retained')
        axes[1,1].plot(qcum[:,2],color='#c65032',label='takeover gripper travel')
        axes[1,1].plot(ref_aligned[:,2],color='#2676b8',label='nominal gripper travel')
        axes[1,1].set(title='Cumulative gripper travel (mm); cycles retained')
        for ax in axes.flat:
            ax.set(xlim=(0,max(1,n-1)),xlabel='Steps after takeover')
            ax.set_ylim(bottom=0);ax.grid(alpha=.15);ax.legend(fontsize=7)
        stream=io.BytesIO();fig.savefig(stream,format='png')
        chart=Image.open(stream).convert('RGB');boxes=[ax.get_position().bounds for ax in axes.flat];plt.close(fig)
        target=videos/f'suffix_t{row["anchor"]}_seed_{seed}.mp4'
        encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x1120',
            '-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
        prefix=min(20,row['takeover']);shots=[]
        for offset in range(-prefix,n):
            relative=max(0,offset);absolute=row['takeover']+offset;j=int(mapping[relative])
            canvas=Image.new('RGB',(1280,1120),'white');draw=ImageDraw.Draw(canvas)
            canvas.paste(Image.fromarray(qf[max(0,absolute-qa['state_step_offset'])]).resize((640,400)),(0,70))
            canvas.paste(Image.fromarray(rf[max(0,j-ra['state_step_offset'])]).resize((640,400)),(640,70))
            draw.text((15,10),f"Takeover t={row['anchor']} | seed {seed}",font=font,fill='#c65032')
            draw.text((655,10),f"Nominal OOD expert | seed {rm['seed']}",font=font,fill='#2676b8')
            phase='POLICY PREFIX (not scored)' if offset<0 else 'EXPERT CONTINUATION'
            draw.text((15,40),f'{phase} | absolute step {absolute}',font=small,fill='black')
            draw.text((655,40),f'Reference step {j}, time-warped; different reset',font=small,fill='black')
            status='completed successfully' if row['success'] else 'INCOMPLETE: low D does NOT mean good takeover'
            draw.text((15,478),f"Dshape={row['D_shape']:.2f} Dmotion={row['D_motion']:.2f} Dmax={row['D_combined']:.2f} | {status}",font=small,fill='black')
            canvas.paste(chart,(0,510))
            for x,y,w,h in boxes:
                xx=int((x+w*relative/max(1,n-1))*1280)
                draw.line((xx,510+int((1-y-h)*540),xx,510+int((1-y)*540)),fill='#333333',width=2)
            draw.text((15,1065),f"Excess vs remaining nominal: path {row['extra_path_cm']:.1f} cm | rotation {row['extra_rotation_deg']:.1f} deg | gripper {row['extra_gripper_travel_mm']:.1f} mm",font=small,fill='black')
            draw.text((15,1090),'Post-takeover diagnostic, not an online trigger or a downstream SR prediction. Inspect Oracle behavior.',font=small,fill='black')
            encoder.stdin.write(canvas.tobytes())
            if offset in [-prefix,0,n//4,n//2,n-1]:
                canvas.save(videos/f'seed_{seed}_offset_{offset}.jpg',quality=88)
                shots.append(canvas.resize((640,560)))
        encoder.stdin.close();assert encoder.wait()==0
        sheet=Image.new('RGB',(640*len(shots),560),'white')
        for i,shot in enumerate(shots):sheet.paste(shot,(640*i,0))
        sheet.save(videos/f'contact_seed_{seed}.jpg',quality=90)
        audit.append(dict(seed=seed,anchor=row['anchor'],query_video=qa,reference_video=ra,
                          output=str(target),frames=n+prefix,offset_range=[-prefix,n-1]))
        print('SUFFIX_VIDEO_COMPLETE',seed,flush=True)
    (root/'video_audit.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[78529,78526,78703,78817,78829,78402,78833])
    a=p.parse_args();main(a.root,a.previous,a.seeds)

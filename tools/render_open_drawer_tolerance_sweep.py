#!/usr/bin/env python3
"""Same source frame / correspondence, three tolerance levels side by side."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from render_open_drawer_ed_audit import download,frames
from sweep_open_drawer_overlap_tolerance import name

COLORS={'non_target':'#b8bec5','target_compatible':'#24986c','target_mismatch':'#d95648'}


def main(root,source,suffix):
    factors=[1.,2.,3.]
    variants=[json.loads((root/name(f)/'analysis.json').read_text()) for f in factors]
    out=root/'videos';out.mkdir(exist_ok=True)
    font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',16)
    small=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',12)
    audits=[]
    for seed in [78600,78703,78810,78817]:
        rows=[next(r for r in v['rows'] if r['seed']==seed) for v in variants];base=rows[0]
        candidates=[source/'source_videos'/f'query_{seed}.mp4',suffix/'source_videos'/f'query_{seed}.mp4']
        path=next((p for p in candidates if p.exists()),out/f'source_{seed}.mp4')
        download(base['video'],path)
        nrows=len(np.load(Path(base['directory'])/'states.npy'));video,meta=frames(path,nrows)
        assert len(video)==nrows-1
        # Source collectors use _build_frames(records[:-1]): frame t is BEFORE action t.
        meta['state_step_offset']=0;meta['source_semantics']='pre-action observation frame t = state t'
        destination=out/f"t{base['anchor']}_seed_{seed}_x1_x2_x3.mp4"
        enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1440x550','-r','10','-i','-',
            '-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],stdin=subprocess.PIPE)
        for i in range(base['expert_anchors']):
            t=base['takeover']+i;canvas=Image.new('RGB',(1440,550),'white');draw=ImageDraw.Draw(canvas)
            im=Image.fromarray(video[t]).resize((480,300))
            for col,(factor,row) in enumerate(zip(factors,rows)):
                x=480*col;canvas.paste(im,(x,60));color=COLORS[row['labels'][i]]
                draw.text((x+10,8),f"Tolerance x{factor:g} | t={row['anchor']}",font=font,fill='black')
                draw.text((x+10,34),f"Q={row['Q']:.4f}  {row['compatible_anchors']}/{row['expert_anchors']}",font=small,fill='black')
                draw.rectangle((x+3,62,x+477,358),outline=color,width=4)
                phase=next((p for p,(a,b) in row['blocks'].items() if a<=t<b),None)
                draw.text((x+10,372),f"seed {seed} | step {t} | {phase or 'outside target'}",font=small,fill='black')
                draw.text((x+10,398),row['labels'][i],font=font,fill=color)
                if phase:
                    draw.text((x+10,428),f"error {row['distance'][i]:.4f} | limit {row['thresholds'][phase]:.4f}",font=small,fill='black')
                else:draw.text((x+10,428),'Not target: stays grey at every tolerance',font=small,fill='black')
                for j,label in enumerate(row['labels']):
                    x0=x+10+460*j/row['expert_anchors'];x1=x+10+460*(j+1)/row['expert_anchors']
                    draw.rectangle((int(x0),466,int(x1)+1,492),fill=COLORS[label])
                cursor=x+10+460*i/row['expert_anchors'];draw.line((cursor,461,cursor,498),fill='black',width=2)
                draw.text((x+10,519),'Same frames, target, reference & contact rules',font=small,fill='black')
            enc.stdin.write(canvas.tobytes())
            if i in [10,20,30,40]:canvas.save(out/f'seed_{seed}_offset_{i}.jpg',quality=91)
        enc.stdin.close();assert enc.wait()==0
        audits.append(dict(seed=seed,anchor=base['anchor'],factors=factors,frames=base['expert_anchors'],output=str(destination),source=meta))
        print('SWEEP_VIDEO_COMPLETE',seed,flush=True)
    (root/'video_audit.json').write_text(json.dumps(audits,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--suffix',type=Path,required=True)
    a=p.parse_args();main(a.root,a.source,a.suffix)

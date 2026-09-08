"""Extract unchanged reset/contact/terminal RGB frames for manual evidence review."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pi05_feedback_artifacts import completed_rows,episode_path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
summary=json.loads((args.root/'summary.json').read_text()) if (args.root/'summary.json').exists() else {}
episodes=[episode_path(args.root,row) for row in completed_rows(args.root)] if summary.get('rows') and 'episode' in summary['rows'][0] else sorted(args.root.glob('episode_*'))
for episode in episodes:
    data=np.load(episode/'trace.npz');indices=np.flatnonzero(data['grasped'])
    picks={'reset':0,'contact':int(indices[0]) if len(indices) else len(data['main'])//2,'terminal':len(data['main'])-1}
    for name,i in picks.items():Image.fromarray(data['main'][i]).save(episode/f'{name}_main.png')
    print(episode.name,picks)

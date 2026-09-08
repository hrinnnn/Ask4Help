"""Extract unchanged reset/contact/terminal RGB frames for manual evidence review."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
for episode in sorted(args.root.glob('episode_*')):
    data=np.load(episode/'trace.npz');indices=np.flatnonzero(data['grasped'])
    picks={'reset':0,'contact':int(indices[0]) if len(indices) else len(data['main'])//2,'terminal':len(data['main'])-1}
    for name,i in picks.items():Image.fromarray(data['main'][i]).save(episode/f'{name}_main.png')
    print(episode.name,picks)

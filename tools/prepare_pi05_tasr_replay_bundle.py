#!/usr/bin/env python3
import argparse,gzip,json
from pathlib import Path
import numpy as np

def main(root):
 rows=[]
 for group in json.loads((root/'training_assets/manifest.json').read_text()):
  for row in json.loads(Path(group['episodes_file']).read_text()):
   arrays=np.load(row['arrays']);row.update(qpos=arrays['qpos'].tolist(),actions=arrays['actions'].tolist());rows.append(row)
 sources={task:Path('RLinf/rlinf/envs/maniskill/'+file).read_text() for task,file in [('object','pick_single_ycb_object_variation.py'),('airplane','pick_single_ycb_airplane_variants.py'),('stackcube','stack_cube_variants.py')]}
 payload=dict(rows=rows,sources=sources,provenance='Exact previously selected pi05 training suffixes, immutable source metadata in training_assets/manifest.json')
 (root/'replay_bundle.json.gz').write_bytes(gzip.compress(json.dumps(payload).encode()))
 print('BUNDLE',len(rows),(root/'replay_bundle.json.gz').stat().st_size,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)

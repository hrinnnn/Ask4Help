"""Small read-only export of original proprioception aligned with frozen forward."""
import io,json,subprocess
from pathlib import Path
import numpy as np

ROOT=Path('artifacts/expert_feedback_pca_20260907')
REMOTE=r'''
import io,sys,json,numpy as np,pyarrow.parquet as pq
from pathlib import Path
samples=json.load(open('/mnt/data/ask4help/results/expert_feedback_pca_probe_v1/forward_v1/samples.json'));tables={};rows=[]
for s in samples:
 if s['path'] not in tables:tables[s['path']]=pq.read_table(s['path'],columns=['state']).to_pydict()['state']
 rows.append(tables[s['path']][s['offset']])
buf=io.BytesIO();np.savez_compressed(buf,qpos=np.asarray(rows,dtype=np.float32));sys.stdout.buffer.write(buf.getvalue())
'''

if __name__=='__main__':
 r=subprocess.run(['ssh','-o','ConnectTimeout=12','-o','BatchMode=yes','-p','1012','root@39.101.70.188','/root/.venvs/xvla-h20/bin/python -'],input=REMOTE.encode(),stdout=subprocess.PIPE,check=True)
 a=np.load(io.BytesIO(r.stdout));samples=json.loads((ROOT/'forward_v1/samples.json').read_text());assert len(a['qpos'])==len(samples)
 (ROOT/'forward_v1/proprio.npz').write_bytes(r.stdout);print('PROPRIO_ALIGNED',a['qpos'].shape)

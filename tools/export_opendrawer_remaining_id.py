"""Complete the original ID corpus without resending the fixed 32-demo subset."""
import argparse,json,tarfile,time,os
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
root=Path('/data/zhaozhixuan/Ask4Help-open-drawer/results/id_oracle_collection_v1/lerobot_datasets/open_drawer_retrieve_place/id_oracle_128_retry1_v1')
count=json.loads((root/'meta/info.json').read_text())['total_episodes']
present=set(np.linspace(0,count-1,32,dtype=int).tolist());remaining=[i for i in range(count) if i not in present]
with tarfile.open(args.output/'remaining_ID.tar','w') as tar:
    for j,i in enumerate(remaining):
        file=root/f'data/chunk-000/episode_{i:06d}.parquet'
        tar.add(file,arcname='ID_subset/data/chunk-000/'+file.name)
        (args.output/'progress.json').write_text(json.dumps({'pid':os.getpid(),'done':j+1,'total':len(remaining)}))
report={'status':'EXISTING_ID_REMAINDER_EXPORTED','source':str(root),'episodes':remaining,'count':len(remaining),'archive_bytes':(args.output/'remaining_ID.tar').stat().st_size,'purpose':'complete original 128-ID corpus for runtime-consistent PCA and later SFT; no new or altered demonstrations'}
(args.output/'EXPORT_COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

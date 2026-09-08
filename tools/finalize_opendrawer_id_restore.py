"""Restore only the known missing original ID files, then verify all anchors."""
import argparse,json,tarfile
from pathlib import Path
import pyarrow.parquet as pq

p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);args=p.parse_args()
archive=args.inputs/'remaining_ID.tar'
with tarfile.open(archive) as tar:
    members=tar.getmembers()
    for m in members:
        path=Path(m.name)
        assert m.isfile() and path.parts[:3]==('ID_subset','data','chunk-000') and path.name.startswith('episode_') and path.suffix=='.parquet'
    assert len(members)==96
    tar.extractall(args.inputs,filter='data')
dataset=args.inputs/'ID_subset'
episodes=[json.loads(x) for x in (dataset/'meta/episodes.jsonl').read_text().splitlines()]
total=0
for row in episodes:
    file=dataset/f'data/chunk-000/episode_{row["episode_index"]:06d}.parquet'
    actual=pq.ParquetFile(file).metadata.num_rows;assert actual==row['length'];total+=actual
assert len(episodes)==128 and total==22973
result={'status':'FULL_ORIGINAL_ID_RESTORED','episodes':128,'anchors':total,'dataset':str(dataset),'no_data_regenerated':True}
(args.inputs/'FULL_ID_RESTORE_COMPLETE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

"""Resolve immutable episode artifacts across restarted collection chunks."""
import json
from pathlib import Path
import numpy as np


def episode_path(root,row):
    return Path(row.get('artifact_directory',str(Path(root)/('episode_%04d'%row['episode']))))


def completed_rows(root):
    root=Path(root)
    if (root/'summary.json').exists():
        rows=json.loads((root/'summary.json').read_text())['rows']
    else:
        paths=sorted(root.glob('episode_*/result.json'))
        provenance=json.loads((root/'provenance.json').read_text())
        rows=completed_rows(provenance['resume_from']) if provenance.get('resume_from') else []
        rows += [json.loads(path.read_text()) for path in paths]
    for i,row in enumerate(rows):
        assert row['episode']==i,'Only a contiguous completed prefix can be resumed'
        directory=episode_path(root,row)
        assert all((directory/name).is_file() for name in ['trace.npz','result.json','trajectory.mp4'])
    return [{**row,'artifact_directory':str(episode_path(root,row))} for row in rows]


def restore_memory(gate,rows):
    for row in rows:
        gate.begin_episode(row['episode'])
        if row['cue'] is None:continue
        with np.load(Path(row['artifact_directory'])/'trace.npz') as data:
            index=-2 if row['cue']['attribution']=='previous_query' else -1
            assert int(data['query_steps'][index])==row['cue']['credit_step']
            feature=data['query_features'][index].copy()
        gate.commit(row['cue'],feature,completed_episode=row['episode'])

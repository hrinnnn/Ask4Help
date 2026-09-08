"""CPU-only export of existing Bridge statistics and fixed ID calibration demos."""
import argparse
import json
import os
import tarfile
import time
from pathlib import Path
import numpy as np
import torch


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(2)
    source=Path('/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_v9_formal_failure_ood_v1_retry12_metrics_retry4/failure_detection/assets/internal')
    assets=torch.load(source/'detector_assets.pt',map_location='cpu',mmap=True,weights_only=False)
    cache=torch.load(source/'feature_cache.pt',map_location='cpu',mmap=True,weights_only=False)
    assert assets['checkpoint']==cache['checkpoint'] and assets['dataset_root']==cache['dataset_root']
    pca=assets['detectors']['vlm_bridge_final_mean__pca_residual']['statistics']
    compact={'format':'open_drawer_bridge_reference_export_v1','checkpoint':assets['checkpoint'],
             'dataset_root':assets['dataset_root'],'indices':cache['indices'],
             'detectors':{'vlm_bridge_final_mean__pca_residual':{'statistics':{k:v.clone() if torch.is_tensor(v) else v for k,v in pca.items()}}},
             'layers':{'vlm_bridge_final_mean':cache['layers']['vlm_bridge_final_mean'].clone()}}
    torch.save(compact,args.output/'bridge_reference.pt')
    dataset=Path(assets['dataset_root']);meta=json.loads((dataset/'meta/info.json').read_text())
    selected=np.linspace(0,meta['total_episodes']-1,32,dtype=int).tolist()
    entries=[]
    with tarfile.open(args.output/'opendrawer_calibration_inputs.tar','w') as tar:
        tar.add(args.output/'bridge_reference.pt',arcname='bridge_reference.pt')
        for file in sorted((dataset/'meta').iterdir()):
            if file.is_file():tar.add(file,arcname='ID_subset/meta/'+file.name)
        for i in selected:
            file=dataset/f'data/chunk-000/episode_{i:06d}.parquet'
            tar.add(file,arcname=f'ID_subset/data/chunk-000/{file.name}')
            entries.append({'episode':i,'source':str(file),'bytes':file.stat().st_size})
            (args.output/'progress.json').write_text(json.dumps({'pid':os.getpid(),'episodes':len(entries),'total':32,'elapsed':time.time()-start}))
    report={'status':'EXPORTED_EXISTING_ID_CALIBRATION_ASSETS','source_checkpoint':assets['checkpoint'],
            'source_assets':str(source),'source_ID_dataset':str(dataset),'original_ID_episodes':meta['total_episodes'],
            'original_ID_anchors':meta['total_frames'],'selected_calibration_episodes':selected,
            'archive_bytes':(args.output/'opendrawer_calibration_inputs.tar').stat().st_size,
            'bridge_shape':list(compact['layers']['vlm_bridge_final_mean'].shape),'entries':entries,
            'scope':'Only 32 original ID demo files restored for calibration; metadata and features describe all 128. This is not a complete SFT replay dataset.'}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    (args.output/'EXPORT_COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='entries'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())

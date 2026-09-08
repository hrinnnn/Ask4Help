"""CPU audit of the actual OpenPI transforms and exact source-balanced batches."""
import argparse,dataclasses,json,sys
from pathlib import Path


def run(a):
    sys.path[:0]=[str(a.source),str(a.source/'RLinf')]
    import torch
    from omegaconf import OmegaConf
    import openpi.training.data_loader as official
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.data.openpi_mixture import (_unwrap_pytorch_loader,ActionHorizonMaskDataset,
          attach_source_balanced_openpi_dataloader,OpenPIActionMaskDataLoader)
    from pi05_feedback_sft_worker import SizedMaskedLoader
    torch.set_num_threads(2)
    provenance=json.loads((a.collection/'fixed/provenance.json').read_text())
    manifest=json.loads(a.manifest.read_text());assets=manifest['task_assets'][provenance['runtime']['task']]
    ID=Path(assets.get('ID_dataset',assets.get('ID_dataset_candidate')))
    if not (ID/'meta/info.json').exists():ID=ID/'lerobot'
    reports=[]
    for arm in ['fixed','feedback']:
        loaders=[];lengths=[];datasets=[]
        for data_root in [ID,a.export/arm]:
            cfg=get_openpi_config('pi05_rlt_maniskill_joint',model_path=assets['checkpoint'],batch_size=4,
                repo_id=str(data_root),data_kwargs=OmegaConf.create({'default_prompt':provenance['runtime']['instruction'],
                                                                  'norm_stats_path':assets['norm']}))
            # This check isolates data semantics; multiprocessing is checked in the later real-model smoke.
            cfg=dataclasses.replace(cfg,num_workers=0)
            loader=official.create_data_loader(cfg,framework='pytorch',shuffle=True)
            _,raw=_unwrap_pytorch_loader(loader)
            masked=ActionHorizonMaskDataset(raw.dataset,action_horizon=10)
            assert len(masked)==len(raw.dataset)
            loaders.append(loader);datasets.append(masked);lengths.append(len(masked))
        mixed=attach_source_balanced_openpi_dataloader(loaders[0],datasets=datasets,seed=1787000)
        wrapped=SizedMaskedLoader(OpenPIActionMaskDataLoader(mixed))
        indices=next(iter(wrapped.sampler))
        assert sum(i<lengths[0] for i in indices)==2 and len(indices)==4
        batch=next(iter(wrapped));actions=batch['actions'];mask=batch['action_valid_mask'];obs=batch['observation']
        assert actions.shape==(4,10,32) and mask.shape==(4,10)
        assert torch.isfinite(actions).all() and bool(mask.any(dim=1).all())
        assert torch.count_nonzero(actions[:,:,8:])==0
        report={'arm':arm,'source_lengths':lengths,'source_counts_first_microbatch':[2,2],
                'actions_shape':list(actions.shape),'mask_shape':list(mask.shape),'mask_valid_steps':mask.sum(1).tolist(),
                'image_shapes':{k:list(v.shape) for k,v in obs.images.items()},
                'image_masks':{k:v.tolist() for k,v in obs.image_masks.items()},
                'loader_length':len(wrapped),'norm':assets['norm'],'instruction':provenance['runtime']['instruction']}
        reports.append(report);print(json.dumps(report),flush=True)
    result={'status':'NATIVE_TRANSFORMS_AND_BALANCED_BATCH_PASS','rows':reports,
            'scope':'CPU single-process data check; no model forward, no new checkpoint and no SR'}
    (a.export/'NATIVE_BATCH_AUDIT.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','collection','export','manifest']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

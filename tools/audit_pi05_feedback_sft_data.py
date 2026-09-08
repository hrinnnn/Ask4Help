"""CPU-only native LeRobot tail-mask and worker real-dimension contract checks."""
import argparse,json,sys
from pathlib import Path


def run(args):
    sys.path[:0]=[str(args.source),str(args.source/'RLinf')]
    import numpy as np
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from rlinf.data.openpi_mixture import ActionHorizonMaskDataset
    from rlinf.algorithms.awbc import weighted_flow_matching_loss
    from pi05_feedback_sft_worker import FeedbackVlaSftWorker,SizedMaskedLoader
    from types import SimpleNamespace
    from contextlib import nullcontext
    torch.set_num_threads(2)
    reports=[]
    for arm in ['fixed','feedback']:
        info=json.loads((args.root/f'{arm}_export_manifest.json').read_text())
        ds=LeRobotDataset(repo_id='local/pi05_feedback_audit_'+arm,root=args.root/arm,
                          delta_timestamps={'actions':[i/10 for i in range(10)]})
        masked=ActionHorizonMaskDataset(ds,action_horizon=10)
        assert len(ds)==len(masked)==info['anchors']
        ends=[int(x) for x in ds.episode_data_index['to']]
        starts=[int(x) for x in ds.episode_data_index['from']]
        assert [b-a for a,b in zip(starts,ends)]==[r['anchors'] for r in info['episode_map']]
        checked=[]
        for a,b,record in zip(starts,ends,info['episode_map']):
            source=np.load(record['source_trace'])
            for offset in range(1,min(10,b-a)+1):
                sample=masked[b-offset]
                mask=sample['action_valid_mask']
                assert mask.tolist()==[i<offset for i in range(10)]
                assert sample['actions'].shape==(10,8)
                if offset==1:
                    target=torch.as_tensor(source['actions'][-1])
                    assert torch.allclose(sample['actions'],target.expand(10,-1))
            checked.append({'last_anchor':b-1,'valid_targets':1,'tail_checked':min(10,b-a)})
        reports.append({'arm':arm,'anchors':len(masked),'tail_checks':checked,'status':'PASS'})
    flags=[]
    def model(**kwargs):
        flags.append(kwargs['use_action_chunk_loss'])
        element=torch.ones(2,10,32);element[:,:,8:]=1e6;element[0,1:,:8]=1e6
        if kwargs['use_action_chunk_loss']:element=element[:,:,:8]
        return weighted_flow_matching_loss(element,element_mask=kwargs['data']['action_valid_mask'])[0]
    fake=SimpleNamespace(awbc_cfg={},amp_context=nullcontext(),model=model,
        cfg=SimpleNamespace(data=SimpleNamespace(openpi_mask_padded_action_targets=True,get=lambda key,default:False),
                            actor=SimpleNamespace(model=SimpleNamespace(openpi=SimpleNamespace(action_env_dim=8,action_chunk=10)))))
    mask=torch.tensor([[True]+[False]*9,[True]*10])
    loss,metrics=FeedbackVlaSftWorker.get_train_model_output(fake,{'action_valid_mask':mask})
    assert flags==[True] and loss.item()==1. and metrics['real_action_dimensions']==8
    report={'status':'NATIVE_DATA_AND_LOSS_MASK_CONTRACT_PASS','rows':reports,
            'synthetic_masked_loss':loss.item(),'scope':'CPU native reader + worker call contract, not model reload or training smoke'}
    (args.root/'NATIVE_DATA_MASK_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    run(p.parse_args())

"""Ordinary BC worker with explicit real-action dimension and temporal masking."""
import torch
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


class SizedMaskedLoader:
    """Expose native mask loader length and its source-balanced epoch sampler."""
    def __init__(self,loader):self.loader=loader
    def __len__(self):return len(self.loader.pytorch_loader)
    def __iter__(self):return iter(self.loader)
    def data_config(self):return self.loader.data_config()
    @property
    def sampler(self):return self.loader.pytorch_loader.batch_sampler
    @property
    def dataset(self):return self.loader.pytorch_loader.dataset


class FeedbackVlaSftWorker(FSDPVlaSftWorker):
    def build_dataloader(self,data_paths,eval_dataset=False):
        loader,config=super().build_dataloader(data_paths,eval_dataset=eval_dataset)
        if not eval_dataset:
            assert hasattr(loader,'pytorch_loader'),'The native temporal-mask loader must be enabled'
            loader=SizedMaskedLoader(loader)
        return loader,config

    def get_train_model_output(self,batch):
        assert not self.awbc_cfg.get('enabled',False),'Timing ablation uses ordinary BC only'
        assert self.cfg.data.openpi_mask_padded_action_targets
        assert not self.cfg.data.get('openpi_exclude_padded_action_targets',False)
        assert self.cfg.actor.model.openpi.action_env_dim==8
        assert self.cfg.actor.model.openpi.action_chunk==10
        mask=torch.as_tensor(batch['action_valid_mask'])
        assert mask.ndim==2 and mask.shape[1]==10 and bool(mask.any(dim=1).all())
        with self.amp_context:
            output=self.model(forward_type=ForwardType.SFT,data=batch,use_action_chunk_loss=True)
        loss=output if torch.is_tensor(output) else output['loss']
        return loss,{'loss':loss.detach().item(),'real_action_dimensions':8,
                     'temporal_mask_valid_fraction':mask.float().mean().item(),
                     'temporal_mask_min_valid_steps':mask.sum(1).min().item()}

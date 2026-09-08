"""Native RLinf SFT entry with a task-scoped real-dimension loss worker.

Derived from RLinf examples/sft/train_vla_sft.py (Apache-2.0).
Only the worker class changes; optimizer, accumulation and checkpointing stay native.
"""
import json,logging,os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from pi05_feedback_sft_worker import FeedbackVlaSftWorker
from pi05_feedback_ray_isolation import install as isolate_ray

mp.set_start_method('spawn',force=True)


@hydra.main(version_base='1.1',config_path=None,config_name=None)
def main(cfg):
    assert cfg.cluster.num_nodes==1
    isolate_ray()
    cfg=validate_cfg(cfg)
    logging.info(json.dumps(OmegaConf.to_container(cfg,resolve=True),indent=2))
    cluster=Cluster(cluster_cfg=cfg.cluster);placement=HybridComponentPlacement(cfg,cluster)
    assert cfg.actor.training_backend in ['fsdp','fsdp2']
    actor=FeedbackVlaSftWorker.create_group(cfg).launch(cluster,name=cfg.actor.group_name,
                                                       placement_strategy=placement.get_strategy('actor'))
    runner=SFTRunner(cfg=cfg,actor=actor);runner.init_workers();runner.run()


if __name__=='__main__':main()

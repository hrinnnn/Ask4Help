"""Keep native SFT/checkpoints, archiving each closed checkpoint to durable storage."""
import json,os,shutil
from pathlib import Path
from rlinf.runners.sft_runner import SFTRunner

class ArchivedSFTRunner(SFTRunner):
    def _save_checkpoint(self,is_best=False):
        super()._save_checkpoint(is_best=is_best)
        archive=os.environ.get('FEEDBACK_CHECKPOINT_ARCHIVE')
        if not archive:return
        assert not is_best
        scratch=Path(self.cfg.runner.logger.log_path)
        assert scratch.parent==Path('/dev/shm') and scratch.name.startswith('pi05_budget30_')
        name=f'global_step_{self.global_step}'
        source=scratch/self.cfg.runner.logger.experiment_name/'checkpoints'/name
        destination=Path(archive)/name
        assert source.is_dir() and not destination.exists()
        shutil.copytree(source,destination)
        files=[p for p in source.rglob('*') if p.is_file()]
        assert files and (destination/'actor/model_state_dict/full_weights.pt').exists()
        for p in files:assert (destination/p.relative_to(source)).stat().st_size==p.stat().st_size
        (destination/'ARCHIVE_COMPLETE.json').write_text(json.dumps({'step':self.global_step,'files':len(files),'source':str(source)}))
        # Only this run's verified, archived checkpoint cache is removed.
        shutil.rmtree(source)

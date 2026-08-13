#!/usr/bin/env python3
"""Restart-tolerant H20 setup and StackPyramid smoke controller."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


OFFICIAL_XVLA = "https://github.com/2toinf/X-VLA.git"


class Controller:
    def __init__(self, output: Path, official_repo: Path, venv: Path) -> None:
        self.output = output
        self.repo = official_repo
        self.venv = venv
        self.logs = output / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state_path = output / "pipeline_state.json"
        self.log_path = self.logs / "controller.log"

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}\n"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        print(line, end="", flush=True)

    def state(self, stage: str, status: str, **extra) -> None:
        payload = {
            "stage": stage,
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **extra,
        }
        self.state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def run(self, args: list[str], *, env: dict[str, str] | None = None) -> None:
        self.log("RUN " + " ".join(args))
        merged = os.environ.copy()
        if env:
            merged.update(env)
        with self.log_path.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(args, env=merged, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            raise RuntimeError(f"command failed with exit code {completed.returncode}: {args}")

    def git_commit(self, repo: Path) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()

    def prepare_source(self) -> None:
        self.state("source", "running")
        if not (self.repo / ".git").exists():
            self.repo.parent.mkdir(parents=True, exist_ok=True)
            self.run(["git", "clone", "--depth", "1", OFFICIAL_XVLA, str(self.repo)])
        commit = self.git_commit(self.repo)
        dirty = subprocess.check_output(
            ["git", "-C", str(self.repo), "status", "--porcelain"], text=True
        ).strip()
        if dirty:
            raise RuntimeError(f"official X-VLA checkout is dirty: {self.repo}")
        self.state("source", "complete", repo=str(self.repo), commit=commit)

    def prepare_environment(self) -> None:
        self.state("environment", "running")
        python = Path("/opt/conda/envs/robo-dopamine/bin/python")
        if not python.exists():
            raise RuntimeError(f"native H20 Python missing: {python}")
        if not self.venv.exists():
            self.run([str(python), "-m", "venv", "--system-site-packages", str(self.venv)])
        vpy = self.venv / "bin/python"
        self.run([str(vpy), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        self.run([str(vpy), "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "only-if-needed", "-r", str(self.repo / "requirements.txt")])
        self.run([str(vpy), "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "only-if-needed", "mani_skill==3.0.0b22", "sapien==3.0.1", "gymnasium==0.29.1", "imageio[ffmpeg]"])
        self.state("environment", "complete", python=str(vpy))

    def smoke(self) -> None:
        self.state("smoke", "running")
        vpy = self.venv / "bin/python"
        smoke_dir = self.output / "stackpyramid_smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        env = {"CUDA_VISIBLE_DEVICES": "1", "STACKPYRAMID_SMOKE_DIR": str(smoke_dir)}
        self.run([str(vpy), "-c", "import torch; import mani_skill; import transformers; assert torch.cuda.is_available() and torch.cuda.device_count() >= 1; print(torch.__version__, transformers.__version__, mani_skill.__file__)"], env=env)
        smoke_script = Path(__file__).with_name("stackpyramid_h20_smoke.py")
        self.run([str(vpy), str(smoke_script)], env=env)
        record_dir = self.output / "stackpyramid_motionplanning"
        record_dir.mkdir(parents=True, exist_ok=True)
        self.run([str(vpy), "-m", "mani_skill.examples.motionplanning.panda.run", "-e", "StackPyramid-v1", "-n", "1", "--only-count-success", "--save-video", "--render-mode", "rgb_array", "--record-dir", str(record_dir), "--sim-backend", "gpu"], env=env)
        self.state("smoke", "complete", smoke_dir=str(smoke_dir), motionplanning_dir=str(record_dir))

    def persist(self) -> None:
        self.state("persist", "running")
        vpy = self.venv / "bin/python"
        env_dir = Path("/mnt/data/ask4help/environments/xvla")
        env_dir.mkdir(parents=True, exist_ok=True)
        with (env_dir / "xvla-h20-pip-freeze.txt").open("w", encoding="utf-8") as handle:
            subprocess.run([str(vpy), "-m", "pip", "freeze"], check=True, stdout=handle)
        versions = subprocess.check_output([str(vpy), "-c", "import torch, mani_skill, transformers, sys; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(mani_skill.__version__ if hasattr(mani_skill, '__version__') else 'unknown'); print(transformers.__version__)"], text=True).splitlines()
        manifest = {
            "environment_name": "xvla-h20",
            "python": str(vpy),
            "official_xvla_repo": str(self.repo),
            "official_xvla_commit": self.git_commit(self.repo),
            "versions": versions,
            "restore_command": f"python3 -m venv --system-site-packages {self.venv}",
            "pip_freeze": str(env_dir / "xvla-h20-pip-freeze.txt"),
            "stackpyramid_smoke": str(self.output / "stackpyramid_smoke/stackpyramid_smoke.json"),
            "motionplanning_output": str(self.output / "stackpyramid_motionplanning"),
            "gpu_policy": "GPU1 only for smoke; GPU0 reserved for existing StackCube workload",
        }
        (env_dir / "xvla-h20-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        self.state("persist", "complete", manifest=str(env_dir / "xvla-h20-manifest.json"))

    def execute(self) -> None:
        if (self.output / "XVLA_ENV_READY").exists():
            self.log("already complete")
            return
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            self.prepare_source()
            self.prepare_environment()
            self.smoke()
            self.persist()
            (self.output / "XVLA_ENV_READY").write_text("xvla-h20 and StackPyramid smoke passed\n", encoding="utf-8")
            self.state("complete", "complete", marker=str(self.output / "XVLA_ENV_READY"))
            self.log("XVLA_ENV_READY")
        except Exception as exc:
            self.state("failed", "failed", error=repr(exc))
            self.log("FAILED " + repr(exc))
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, default=Path("/root/X-VLA-stackpyramid-clean"))
    parser.add_argument("--venv", type=Path, default=Path("/root/.venvs/xvla-h20"))
    args = parser.parse_args()
    Controller(args.output, args.official_repo, args.venv).execute()


if __name__ == "__main__":
    main()

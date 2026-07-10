# Ask4Help Agent Notes

## Source Control Workflow

- Treat GitHub as the source of truth for this workspace.
- Do code editing and documentation updates on the local machine first, then push to `https://github.com/hrinnnn/Ask4Help`.
- Treat cloud servers as run environments by default. On a new server, clone or pull from GitHub instead of writing long-lived code directly there.
- For a fresh server clone, use:

```bash
git clone --recurse-submodules https://github.com/hrinnnn/Ask4Help.git
```

- For updating an existing server checkout, use:

```bash
cd /root/Ask4Help
git pull
git submodule update --init --recursive
```

- Server-side GitHub write access is only needed when intentionally pushing changes made on the server. Prefer avoiding that for disposable Aliyun instances.
- If server-side push is needed, use a repository-scoped deploy key with write access, and remove stale keys when the server is replaced.

## Server Environment Workflow

- Use the server mainly to install dependencies, run RL experiments, and validate GPU/runtime behavior.
- Once CUDA, Docker, RLinf, openpi, LIBERO, model weights, and smoke tests are working, save an Aliyun custom image.
- Keep code updates in GitHub even when using a saved server image; after booting from the image, pull the latest code.
- Do not rely on a disposable server filesystem as the only copy of code, configs, or experiment notes.

## Persistence Strategy

- Use GitHub for source code and lightweight reproducible configuration:
  - workspace notes and docs
  - custom YAML configs
  - launch scripts
  - small patches
  - experiment manifests
  - RLinf submodule pointers
- Do not put large or frequently changing artifacts in GitHub:
  - model weights
  - LIBERO datasets
  - checkpoints
  - TensorBoard logs
  - evaluation videos
  - large generated rollouts
- Use Aliyun custom images for the reusable system/runtime environment:
  - Docker
  - NVIDIA Container Toolkit
  - CUDA-compatible runtime setup
  - base apt packages
  - conda/uv/venv tooling
  - RLinf/openpi/LIBERO dependency environments
  - smoke-tested runtime fixes
- Avoid baking fast-changing experiment artifacts into images. Images should make a new server boot quickly into a usable environment, while GitHub and OSS provide current code and data.
- Use OSS or a persistent cloud disk for large durable artifacts:
  - Hugging Face / ModelScope model checkpoints
  - LIBERO datasets
  - RL training checkpoints
  - logs
  - evaluation videos
  - result archives
- Preferred server layout:

```bash
/root/Ask4Help              # GitHub workspace
/data/ask4help/models       # model weights
/data/ask4help/datasets     # datasets and simulator assets
/data/ask4help/results      # logs, checkpoints, videos, eval outputs
```

- Preferred lifecycle:
  1. Edit locally and push to GitHub.
  2. On a server, clone or pull GitHub code.
  3. Sync large inputs from OSS or mount a persistent data disk.
  4. Run experiments and write outputs under `/data/ask4help/results`.
  5. Sync results/checkpoints/videos back to OSS.
  6. After the runtime is verified, save an Aliyun custom image for future instances.

## RLinf / pi0.5 Setup Guidance

- For pi0.5 RL, prefer the RLinf + openpi + LIBERO path first.
- Use Flow-SDE for pi0.5 unless there is a specific reason to compare Flow-Noise.
- Confirm that configs keep `openpi.noise_method: "flow_sde"` and `openpi.train_expert_only: True` when the goal is RL on the flow-based action expert.
- Prefer Docker for reproducible server setup. If Docker or NVIDIA container runtime fails, debug that layer before changing RLinf code.

## Documentation First

- When blocked or uncertain, consult official documentation before guessing:
  - RLinf docs for installation, Docker images, configs, and run scripts.
  - RLinf repository configs under `RLinf/examples/embodiment/config/`.
  - openpi documentation or repository notes for pi0/pi0.5 model behavior.
  - LIBERO documentation for environment setup and task suites.
  - NVIDIA Docker / NVIDIA Container Toolkit docs for GPU container issues.
- Record any non-obvious fixes or server-specific setup steps in this workspace so future servers can be recreated faster.

## Official Setup Discipline

- For RLinf/openpi/LIBERO environment setup, follow the official RLinf installation path first:

```bash
cd /root/Ask4Help/RLinf
bash requirements/install.sh embodied --model openpi --env maniskill_libero
```

- If Docker is available and working, prefer the official RLinf Docker image and switch into the documented `openpi` environment before running experiments.
- Do not patch dependency versions, copy package files, or bypass `uv sync` manually unless the official install path has failed and the failure has been recorded.
- For long official installs or downloads, start the command in a durable background session or with `nohup`, record the PID and log path, confirm it is actively running, then report back instead of continuously watching it.
- For stable long training jobs, create a periodic monitor after launch. The monitor should check the PID, GPU allocation, recent log errors, newest metrics, and OSS/result sync status at a sensible interval; do not continuously poll a healthy job.
- Do not stop an official install merely because it has reached a stable download/install phase. Stop it only if the user explicitly asks to stop, or if it is clearly failing or corrupting the environment.
- Smoke tests should run only after the official environment checks pass:

```bash
cd /root/Ask4Help/RLinf
.venv/bin/python -c "import torch, ray, transformers, openpi, libero; print(torch.cuda.is_available(), transformers.__version__)"
```

- Treat ad hoc fixes as temporary diagnostics, not as the reproducible setup. Convert any necessary fix into a documented command or script before relying on it.

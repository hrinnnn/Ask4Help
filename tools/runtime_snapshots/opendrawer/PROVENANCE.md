# OpenDrawer task snapshot

The two environment files were copied unchanged from the actual 5090 task source at
`/data/zhaozhixuan/Ask4Help-open-drawer/RLinf/rlinf/envs/maniskill/`.
They are byte-identical to the main local workspace's RLinf counterparts as
verified with a direct diff on 2026-09-09. No geometry, camera, distribution,
predicate or reward edits were made.

They register the existing ID, handle_ood, grasp_ood and goal_ood environments in
the native H20 runtime. The restored OpenDrawer checkpoint and norm remain
task-specific; the H20 model loader is a version-pinned pi0.5 implementation,
not an Airplane checkpoint. Smoke success does not qualify a new ID base or
unlock old blocked pipelines.

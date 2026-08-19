# StackPyramid ID Recovery 512

## Objective

Extend the repaired StackPyramid v4 ID dataset from 256 to 512 accepted
demonstrations and test whether additional same-distribution supervision and
20,000 fresh SFT steps can produce an accepted ID base. This is an ID-only
recovery pipeline. It does not change the v4 geometry, task instruction,
success predicate, horizon, Oracle sequence, or formal gate definition, and it
does not start OOD, PCA, timing, or four-method work.

## Frozen data boundary

The existing repaired 256-ID collection with seeds `886000--886255` is the
only historical source allowed in the merged dataset. A new disjoint batch of
256 accepted ID demonstrations uses seeds `886300--886555`; raw attempts use
the same sequential seed order and retain the complete failure denominator.
Old v1/v2/v3 assets, all failed formal retries, and the current failed
`ckpt-10000` are excluded.

Each new episode must use the repaired monotone Oracle sequence:

`single red grasp -> lift -> closed direct transport -> target lower/release -> retreat -> blue stage`.

The collection audit must verify 256 accepted action/state trajectories, 256
videos, event order, reset invariants, action/observation boundaries, episode
lengths, truncation/stall evidence, and video decodability. The collector must
record geometry=v4, the environment ID, reset metadata, event history and
first-event steps for every raw attempt.

## Merge and training

The two audited 256-ID HDF5 sources are copied into a new 512-episode ID-only
training root. The merged root records source provenance, `ood_included=false`,
recomputed X-VLA action-space normalization, anchor counts, tail-anchor
counts, valid-target distribution and source balance. The final real-action
tail anchor must remain in the dataset with an explicit temporal mask.

Training starts fresh from
`/mnt/data/ask4help/models/X-VLA-Pt_from5090_v4`; no failed StackPyramid
checkpoint may be used as an initialization. The run lasts at most 20,000
steps and saves only the pre-registered checkpoints at 5,000, 10,000, 15,000
and 20,000 steps. A 2-step reload/forward smoke is mandatory before formal
training.

## Selection and formal gate

Each saved checkpoint is evaluated independently on the 20-seed selection
manifest `88500--88519`. Selection is diagnostic only and cannot write a
formal validation marker. The selected checkpoint is then evaluated once on
the frozen formal ID manifest `84400--84499` with explicit geometry=v4,
fresh-environment-per-episode lifecycle, reset metadata, and formal
per-episode action/state/video evidence. The formal gate requires at least
`80/100` strict successes and complete evidence. A failed gate remains a
diagnostic and locks all OOD/PCA/timing/downstream stages.

## Outputs and stop conditions

The active recovery root is
`/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/`. Collection,
audit, merge, smoke, training, selection and formal evaluation use separate
subdirectories and completion markers. Any missing evidence, geometry or
reset mismatch, non-finite loss, failed smoke, raw-attempt overflow, or formal
ID failure stops the pipeline without changing the frozen protocol. GPU0 is
reserved for this pipeline; GPU1 and unrelated processes remain untouched.

## Oracle Provenance Recovery

The previous `ORACLE_SMOKE_FAILED_BLUE_STAGE` conclusion is retracted as a
scientific claim. Its v2 smoke outputs remain engineering diagnostics. The
canonical 256-ID video/HDF5 action replay was audited at the event level:
red close occurs around step 47, red release around step 120, blue close
around step 196, blue lift begins around step 210, and blue release occurs
around step 260. Replaying the 280-action canonical template succeeds on
seeds `886000`, `886280`, `886281`, `886300`, and `886301` in the current v4
runtime.

The active collector therefore uses a hybrid path: it plans the red
single-grasp, lift, direct transport and target release from the current
reset, then appends only the known-good blue suffix beginning at historical
step 130. This preserves state-conditioned red trajectories and avoids
expanding a fixed action template as if it were new data. The first hybrid
smoke (`retry10`) passed on `886280/886281` with strict success and red/blue
event evidence; formal collection is continuing in a fresh retry2 directory.

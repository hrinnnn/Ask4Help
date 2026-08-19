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

The durable root is
`/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v2/`. Collection,
audit, merge, smoke, training, selection and formal evaluation use separate
subdirectories and completion markers. Any missing evidence, geometry or
reset mismatch, non-finite loss, failed smoke, raw-attempt overflow, or formal
ID failure stops the pipeline without changing the frozen protocol. GPU0 is
reserved for this pipeline; GPU1 and unrelated processes remain untouched.

## Oracle Smoke Closure

The pipeline stopped before formal collection. Under the explicit v4 reset
and red event contract, five fresh-environment smoke retries and seed
comparisons consistently reached red placement but produced no blue grasp or
blue lift. The first smoke retry was separately invalid because the controller
did not export geometry=v4; it is retained as an engineering diagnostic. The
v4 retries used full fresh lifecycle and reset metadata, and none produced an
accepted demonstration. The reconciliation is
`ORACLE_SMOKE_FAILED_BLUE_STAGE` under
`/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v2/`.

The historical 256-ID collection is not evidence for this protocol because
its successful episodes retained the red cube into the blue phase. Continuing
to alter blue waypoints would change the Oracle design rather than repair the
approved pipeline. Therefore no new 256-ID collection, 512 merge, training,
selection probe, formal gate, or OOD/PCA/timing stage was started.

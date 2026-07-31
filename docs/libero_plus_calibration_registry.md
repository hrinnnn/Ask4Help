# LIBERO-Plus Calibration Registry

Every reusable detector threshold bundle lives outside Git under the durable
results root:

```text
/data/zhaozhixuan/libero_plus_failure/calibration_registry/<calibration-id>/
```

Each immutable bundle contains `thresholds.json` and `registry_manifest.json`.
The manifest binds the thresholds to the exact feature-bank SHA, policy
checkpoint path, source commit, conformal protocol, and successful policy
rollout count. A downstream experiment must verify the feature-bank SHA before
using a bundle. A calibration is never silently overwritten: create a new ID
when the policy, feature bank, calibration distribution, or threshold protocol
changes.

The initial LIBERO-10 bundle is deliberately named `lightweight`: it uses 42
clean policy-success trajectories at `q=0.95`, not the planned 100-trajectory
strict calibration. Results using it must retain that qualifier.

The small, versioned reproduction spec is also tracked in Git at
`configs/calibrations/libero10_pi05_clean_policy_q95_n42_lightweight_v1.json`.
It records every threshold, the compatible feature-bank SHA, policy checkpoint,
and the exact successful calibration-attempt schedule. The full registry bundle
remains a runtime artifact because it includes the selected rollout paths.

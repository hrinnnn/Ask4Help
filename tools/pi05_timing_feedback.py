"""Action-block timing feedback used identically by the two ablation arms.

This module has no access to task labels, success, or future episodes. The
collector owns observations, normal policy generation, episode completion and
the expert budget; only commit() changes the memory used by later episodes.
"""
from dataclasses import dataclass, field
from contextlib import contextmanager
import random
import numpy as np


@contextmanager
def isolated_python_numpy_rng(seed):
    """RLinf Flow-SDE chooses its stochastic step with Python random.

    Forking only Torch leaves this source unpaired. Restore external state so
    post-intervention diagnostics cannot perturb subsequent rollout randomness.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)


def action_block_error(predictions, expert_actions, block=5):
    """M samples, one physical action scale, exactly block real targets."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(expert_actions, dtype=float)
    if targets.shape[0] < block:
        return None
    generated = predictions[:, :block, :targets.shape[1]]
    if generated.shape[1:] != targets[:block].shape:
        raise ValueError("prediction and expert execution-action shapes differ")
    return float(np.mean((generated - targets[None, :block]) ** 2))


def corrective_motion(tcp_before, tcp_takeover, tcp_after, grip_before,
                      grip_takeover, grip_after, position_scale=.02,
                      gripper_scale=.01):
    previous = np.asarray(tcp_takeover) - np.asarray(tcp_before)
    following = np.asarray(tcp_after) - np.asarray(tcp_takeover)
    distance = float(np.linalg.norm(previous))
    opposing = max(0., -float(following @ previous) / distance) if distance > 1e-8 else 0.
    previous_grip = grip_takeover - grip_before
    following_grip = grip_after - grip_takeover
    reverse_grip = max(0., -np.sign(previous_grip) * following_grip) if abs(previous_grip) > 1e-8 else 0.
    return float(np.hypot(opposing / position_scale, reverse_grip / gripper_scale))


def timing_cue(errors, error_reference, *, reversal=None,
               reversal_reference=None, has_previous_query=False, block=5):
    """Return one cue and its real policy-state attribution after intervention.

    Each element is an error computed using a complete real execution block.
    None/nonfinite ends the observed prefix; it is not a low-error observation.
    """
    if (has_previous_query and reversal is not None and
            reversal_reference is not None and reversal > reversal_reference):
        return {"direction": 1, "attribution": "previous_query", "reason": "corrective_motion"}
    observed = []
    for error in errors:
        if error is None or not np.isfinite(error):
            break
        observed.append(float(error))
    crossing = next((i for i, error in enumerate(observed) if error > error_reference), None)
    if crossing is not None and crossing < block:
        return {"direction": 0, "attribution": "takeover_query", "reason": "immediate_disagreement"}
    if len(observed) >= block and all(e <= error_reference for e in observed[:block]):
        return {"direction": -1, "attribution": "takeover_query", "reason": "observed_low_error_prefix"}
    return None


def gripper_commitment_opening(previous_commands, expert_commands, width_before, width_after):
    """Opening against a sustained closing command, measured in metres."""
    closing=max(0.,-float(np.mean(previous_commands)))
    opening=max(0.,float(np.mean(expert_commands)))
    return closing*opening*max(0.,float(width_after-width_before))


def commitment_timing_cue(errors, error_reference, *, opening_score,
                          opening_reference, reversal=None,
                          reversal_reference=None, has_previous_query=False,
                          block=5):
    """Exploratory v2: distinguish saturated closed-gripper correction.

    Ordinary release without policy/expert disagreement does not activate
    this added rule. Existing displacement-based reversal has precedence.
    """
    original=timing_cue(errors,error_reference,reversal=reversal,
                        reversal_reference=reversal_reference,
                        has_previous_query=has_previous_query,block=block)
    if original is not None and original['direction']==1:return original
    disagreement=bool(len(errors) and errors[0] is not None and
                      np.isfinite(errors[0]) and errors[0]>error_reference)
    if has_previous_query and disagreement and opening_score>opening_reference:
        return {'direction':1,'attribution':'previous_query','reason':'gripper_commitment_undo'}
    return original


@dataclass
class TimingFeedbackGate:
    baseline_threshold: float
    center: np.ndarray
    scale: float
    radius: float
    regularization: float = 2.
    strength: float = .5
    min_support: int = 2
    min_vote: float = .25
    block: int = 5
    enabled: bool = True
    support_mode: str = "hard_radius"
    max_wait_blocks: int | None = 1
    remember_deferred_alarm: bool = False
    use_later_duration: bool = False
    memory: list = field(default_factory=list)
    episode: int | None = None
    deadline: int | None = None
    pending_alarm: int | None = None

    def begin_episode(self, episode):
        self.episode = int(episode)
        self.deadline = None
        self.pending_alarm = None

    def query(self, feature, score, step):
        normalized = (np.asarray(feature, dtype=float) - self.center) / self.scale
        nearby = []
        for cue in self.memory:
            if cue["episode"] >= self.episode:
                raise ValueError("current/future episode feedback leaked into query")
            distance = float(np.linalg.norm(normalized - cue["feature"]))
            cue_direction = cue["direction"]
            if self.use_later_duration and cue_direction < 0:
                elapsed = 0 if self.pending_alarm is None else int(step)-self.pending_alarm
                if elapsed >= cue.get("later_valid_steps",0):
                    cue_direction = 0
            if self.support_mode in ("soft_mass", "continuous") or distance <= self.radius:
                nearby.append((np.exp(-.5 * (distance / self.radius) ** 2), cue_direction))
        direction = 0.
        vote = 0.
        mass = sum(w for w, _ in nearby)
        ready=len(nearby)>=self.min_support
        if self.support_mode=="soft_mass":
            # Preserve the minimum total weight guaranteed by the old
            # min_support neighbors inside one Gaussian bandwidth.
            ready=ready and mass>=self.min_support*np.exp(-.5)
        if self.support_mode=="continuous":
            ready=mass>0.
        if self.enabled and ready:
            total = sum(w * y for w, y in nearby)
            vote = total / mass
            if self.support_mode=="continuous" or abs(vote) >= self.min_vote:
                direction = total / (self.regularization + mass)
        threshold = float(self.baseline_threshold * np.exp(-self.strength * direction))
        baseline_stop = bool(score > self.baseline_threshold)
        stop = bool(score > threshold)
        if self.enabled and self.remember_deferred_alarm and baseline_stop and not stop and self.pending_alarm is None:
            self.pending_alarm = int(step)
        # An earlier risk crossing remains pending while Later evidence
        # supports deferral. A drop in residual alone cannot erase it.
        deferred_evidence_lost = bool(self.remember_deferred_alarm and self.pending_alarm is not None and direction >= 0.)
        if self.enabled and self.max_wait_blocks is not None and self.deadline is None and baseline_stop and not stop:
            self.deadline = int(step) + self.block * self.max_wait_blocks
        deadline_reached = self.deadline is not None and step >= self.deadline
        return {"threshold": threshold, "baseline_stop": baseline_stop,
                "stop": bool(stop or deadline_reached or deferred_evidence_lost), "deadline": self.deadline,
                "deadline_reached": deadline_reached, "support": len(nearby),
                "support_mass": float(mass), "support_ready": bool(ready),
                "vote": float(vote), "direction": float(direction),
                "pending_alarm": self.pending_alarm, "deferred_evidence_lost": deferred_evidence_lost}

    def commit(self, cue, feature, *, completed_episode):
        if completed_episode != self.episode:
            raise ValueError("commit must belong to the completed current episode")
        if cue is None or not self.enabled:
            return
        if any(c["episode"] == completed_episode for c in self.memory):
            raise ValueError("only one cue per intervention episode")
        self.memory.append({"episode": int(completed_episode),
                            "feature": (np.asarray(feature, dtype=float) - self.center) / self.scale,
                            "direction": int(cue["direction"]),
                            "later_valid_steps": int(cue.get("later_valid_steps",0))})


def observed_agreement_steps(blocks, error_reference, block_size=5):
    """Only a contiguous measured prefix licenses Later; gripper disagreement vetoes it.

    This is expert-path evidence, not a guarantee of autonomous recoverability.
    The caller provides chronological nonoverlapping, fully observed blocks.
    """
    length=0
    for b in blocks:
        if b['offset'] != length:
            break
        if not np.isfinite(b['overall_MSE']) or b['overall_MSE']>error_reference:
            break
        if not np.isfinite(b['gripper_sign_disagreement']) or b['gripper_sign_disagreement']>0:
            break
        length+=block_size
    return length

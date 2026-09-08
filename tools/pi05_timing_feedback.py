"""Action-block timing feedback used identically by the two ablation arms.

This module has no access to task labels, success, or future episodes. The
collector owns observations, normal policy generation, episode completion and
the expert budget; only commit() changes the memory used by later episodes.
"""
from dataclasses import dataclass, field
import numpy as np


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
    memory: list = field(default_factory=list)
    episode: int | None = None
    deadline: int | None = None

    def begin_episode(self, episode):
        self.episode = int(episode)
        self.deadline = None

    def query(self, feature, score, step):
        normalized = (np.asarray(feature, dtype=float) - self.center) / self.scale
        nearby = []
        for cue in self.memory:
            if cue["episode"] >= self.episode:
                raise ValueError("current/future episode feedback leaked into query")
            distance = float(np.linalg.norm(normalized - cue["feature"]))
            if distance <= self.radius:
                nearby.append((np.exp(-.5 * (distance / self.radius) ** 2), cue["direction"]))
        direction = 0.
        vote = 0.
        if self.enabled and len(nearby) >= self.min_support:
            mass = sum(w for w, _ in nearby)
            total = sum(w * y for w, y in nearby)
            vote = total / mass
            if abs(vote) >= self.min_vote:
                direction = total / (self.regularization + mass)
        threshold = float(self.baseline_threshold * np.exp(-self.strength * direction))
        baseline_stop = bool(score > self.baseline_threshold)
        stop = bool(score > threshold)
        if self.enabled and self.deadline is None and baseline_stop and not stop:
            self.deadline = int(step) + self.block
        deadline_reached = self.deadline is not None and step >= self.deadline
        return {"threshold": threshold, "baseline_stop": baseline_stop,
                "stop": bool(stop or deadline_reached), "deadline": self.deadline,
                "deadline_reached": deadline_reached, "support": len(nearby),
                "vote": float(vote), "direction": float(direction)}

    def commit(self, cue, feature, *, completed_episode):
        if completed_episode != self.episode:
            raise ValueError("commit must belong to the completed current episode")
        if cue is None or not self.enabled:
            return
        if any(c["episode"] == completed_episode for c in self.memory):
            raise ValueError("only one cue per intervention episode")
        self.memory.append({"episode": int(completed_episode),
                            "feature": (np.asarray(feature, dtype=float) - self.center) / self.scale,
                            "direction": int(cue["direction"])})

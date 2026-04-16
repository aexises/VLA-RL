from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SimpleVLA-RL"))

from verl.trainer.ppo.core_algos import compute_grpo_dense_advantage
from verl.trainer.ppo.rob_reward import build_binary_reward_tensor, combine_dense_rewards, reward_spec_from_config, validate_reward_support


def _config(task_suite_name: str = "libero_10", reward_type: str = "dense", tau_clip: float = 0.1):
    return SimpleNamespace(
        data=SimpleNamespace(task_suite_name=task_suite_name),
        reward=SimpleNamespace(
            type=reward_type,
            impl="libero_native_dense" if reward_type != "binary" else "baseline_terminal",
            tau_clip=tau_clip,
            normalize_components=False,
            log_components=True,
            weights=SimpleNamespace(subgoal=0.0, progress=1.0, smoothness=0.05, terminal=1.0),
        ),
        actor_rollout_ref=SimpleNamespace(
            model=SimpleNamespace(action_token_len=7, action_chunks_len=8, vla="openvla-oft")
        ),
        trainer=SimpleNamespace(seed=0),
    )


def test_binary_reward_marks_last_valid_token() -> None:
    complete = torch.tensor([True, False], dtype=torch.bool)
    valid_lengths = torch.tensor([3, 2], dtype=torch.int64)
    reward = build_binary_reward_tensor(complete, valid_lengths, torch.Size([2, 1, 4]), device=torch.device("cpu"))
    assert reward.shape == (2, 4)
    assert reward[0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert reward[1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_dense_reward_combination_and_clipping() -> None:
    config = _config(reward_type="clipped_dense", tau_clip=0.1)
    reward_spec = reward_spec_from_config(config)
    batch = {
        "responses": torch.zeros((1, 1, 4), dtype=torch.int64),
        "dense_progress_scores": torch.tensor([[[0.0, -0.2, 0.5, 0.1]]], dtype=torch.float32),
        "dense_smoothness_scores": torch.tensor([[[0.0, -1.0, 0.0, 0.0]]], dtype=torch.float32),
        "dense_terminal_scores": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32),
        "dense_subgoal_scores": torch.zeros((1, 1, 4), dtype=torch.float32),
    }
    reward_tensor_dict, metrics = combine_dense_rewards(batch, reward_spec)
    assert reward_tensor_dict["all"].shape == (1, 4)
    assert torch.all(reward_tensor_dict["all"] >= 0.1)
    assert metrics["clip_activation_ratio"] > 0.0


def test_dense_grpo_advantage_is_masked_and_finite() -> None:
    rewards = torch.tensor([[0.1, 0.2, 0.0, 0.0], [0.3, 0.1, 0.4, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    index = ["prompt-a", "prompt-a"]
    advantages, returns = compute_grpo_dense_advantage(rewards, mask, index=index, gamma=0.9)
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert torch.isfinite(advantages).all()
    assert torch.equal(advantages[~mask], torch.zeros_like(advantages[~mask]))


def test_robotwin_dense_rejected() -> None:
    config = _config(task_suite_name="robotwin2_lift_pot", reward_type="dense")
    try:
        validate_reward_support(config)
    except ValueError as exc:
        assert "Robotwin currently supports only reward.type=binary" in str(exc)
    else:
        raise AssertionError("Expected validate_reward_support to reject Robotwin dense reward.")

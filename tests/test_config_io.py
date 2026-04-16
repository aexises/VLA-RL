from __future__ import annotations

from pathlib import Path

from src.config import (
    EnvConfig,
    ExperimentConfig,
    GRPOConfig,
    LoggingConfig,
    RewardConfig,
    save_experiment_config,
    load_experiment_config,
)


def test_config_round_trip(tmp_path: Path) -> None:
    config = ExperimentConfig(
        env=EnvConfig(name="libero_spatial_pick", seed=22),
        reward=RewardConfig(reward_type="dense", tau_clip=0.2),
        grpo=GRPOConfig(),
        logging=LoggingConfig(experiment_name="config-io"),
    )
    target = tmp_path / "config.yaml"
    save_experiment_config(config, target)

    loaded = load_experiment_config(target)
    assert loaded.env.name == "libero_spatial_pick"
    assert loaded.reward.reward_type == "dense"
    assert loaded.reward.tau_clip == 0.2
    assert loaded.env.seed == 22

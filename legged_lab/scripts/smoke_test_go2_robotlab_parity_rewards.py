#!/usr/bin/env python3
"""CPU-only numerical smoke test for RobotLab parity reward functions."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_rewards_without_isaac_sim():
    """Load the real reward module with type-only Isaac Lab dependencies stubbed."""
    isaaclab = types.ModuleType("isaaclab")
    utils = types.ModuleType("isaaclab.utils")
    math_utils = types.ModuleType("isaaclab.utils.math")
    assets = types.ModuleType("isaaclab.assets")
    managers = types.ModuleType("isaaclab.managers")
    sensors = types.ModuleType("isaaclab.sensors")
    assets.Articulation = type("Articulation", (), {})
    managers.SceneEntityCfg = type("SceneEntityCfg", (), {"__init__": lambda self, *args, **kwargs: None})
    managers.ManagerTermBase = type("ManagerTermBase", (), {})
    managers.RewardTermCfg = type("RewardTermCfg", (), {})
    sensors.ContactSensor = type("ContactSensor", (), {})
    sensors.RayCaster = type("RayCaster", (), {})
    isaaclab.utils = utils
    utils.math = math_utils
    sys.modules.update(
        {
            "isaaclab": isaaclab,
            "isaaclab.utils": utils,
            "isaaclab.utils.math": math_utils,
            "isaaclab.assets": assets,
            "isaaclab.managers": managers,
            "isaaclab.sensors": sensors,
        }
    )
    module_path = Path(__file__).parents[1] / "mdp" / "rewards.py"
    spec = importlib.util.spec_from_file_location("robotlab_parity_rewards_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    rewards = _load_rewards_without_isaac_sim()
    joint_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1, 2])
    feet_cfg = SimpleNamespace(name="robot", body_ids=[0, 1])
    scanner_cfg = SimpleNamespace(name="height_scanner")

    data = SimpleNamespace(
        joint_vel=torch.tensor([[1.0, -2.0, 3.0], [2.0, 1.0, -1.0]]),
        applied_torque=torch.tensor([[2.0, 3.0, -4.0], [1.0, -2.0, 3.0]]),
        joint_pos=torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.2, -0.1]]),
        default_joint_pos=torch.zeros(2, 3),
        root_lin_vel_b=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        root_pos_w=torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.7]]),
        body_pos_w=torch.tensor(
            [[[0.0, 0.0, 0.1], [0.1, 0.0, 0.1]], [[0.0, 0.0, 0.3], [0.1, 0.0, 0.3]]]
        ),
        body_lin_vel_w=torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]]
        ),
    )
    robot = SimpleNamespace(data=data)
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            ray_hits_w=torch.tensor(
                [
                    [[0.0, 0.0, 0.1], [0.1, 0.0, 0.1]],
                    [[0.0, 0.0, float("nan")], [0.1, 0.0, 0.2]],
                ]
            )
        )
    )
    action_history = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [0.5, -0.5]], [[0.2, 0.3], [0.4, 0.5], [0.7, 0.9]]]
    )
    env = SimpleNamespace(
        scene={"robot": robot, "height_scanner": scanner},
        command_generator=SimpleNamespace(command=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])),
        action_buffer=SimpleNamespace(_circular_buffer=SimpleNamespace(buffer=action_history)),
        episode_length_buf=torch.tensor([1, 3]),
        sim=SimpleNamespace(cfg=SimpleNamespace(gravity=(0.0, 0.0, -9.81))),
        device="cpu",
    )

    outputs = {
        "joint_power": rewards.joint_power(env, joint_cfg),
        "action_rate": rewards.action_rate_l2_robotlab(env),
        "action_smoothness": rewards.action_smoothness_l2(env),
        "joint_pos": rewards.joint_pos_penalty_l1(env, joint_cfg, 1.0, 0.1, 0.1),
        "hip_pos": rewards.hip_pos_penalty_l1(env, joint_cfg, 1.0, 0.1),
        "base_height": rewards.base_height_l2(env, 0.4, joint_cfg, scanner_cfg),
        "feet_regulation": rewards.feet_regulation(env, 0.4, feet_cfg, scanner_cfg),
    }
    for name, value in outputs.items():
        assert value.shape == (2,), (name, value.shape)
        assert torch.isfinite(value).all(), (name, value)

    # Environment 0 has a valid 0.1 m terrain estimate; environment 1's NaN
    # falls back independently to target height instead of contaminating env 0.
    torch.testing.assert_close(outputs["base_height"], torch.zeros(2))
    torch.testing.assert_close(outputs["joint_power"], torch.tensor([20.0, 7.0]))
    torch.testing.assert_close(outputs["action_rate"][0], torch.tensor(0.5))
    torch.testing.assert_close(outputs["action_smoothness"][0], torch.tensor(0.0))
    print("RobotLab parity reward smoke test passed (7 terms, shape/finite/reset/fallback checks).")


if __name__ == "__main__":
    main()

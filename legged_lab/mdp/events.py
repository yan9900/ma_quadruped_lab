from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.envs import ManagerBasedEnv

# Per-env "already pushed this episode" flag registry.
# Keyed by env object id to support multiple env instances.
_push_done_registry: dict[int, torch.Tensor] = {}


def push_by_setting_velocity_body_frame(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting root velocity sampled in the robot's body frame.

    Unlike the default ``push_by_setting_velocity`` which treats x/y/z as world-frame
    axes, this function samples the delta velocity in the robot's local (body) frame
    and then rotates it into the world frame before adding it to the current velocity.

    This means ``"x"`` always means "forward relative to the robot" regardless of its
    yaw orientation, which is more physically meaningful for locomotion training.

    The velocity_range dictionary accepts the same keys as the original function:
    ``x``, ``y``, ``z``, ``roll``, ``pitch``, ``yaw``.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Current world-frame velocity  shape (N, 6): [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
    vel_w = asset.data.root_vel_w[env_ids].clone()

    # Sample delta in body frame
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    delta_b = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # Robot orientation quaternion in world frame  (w, x, y, z)
    quat_w = asset.data.root_quat_w[env_ids]

    # Rotate linear and angular delta from body frame to world frame
    delta_lin_w = math_utils.quat_apply(quat_w, delta_b[:, :3])
    delta_ang_w = math_utils.quat_apply(quat_w, delta_b[:, 3:])

    vel_w[:, :3] += delta_lin_w
    vel_w[:, 3:] += delta_ang_w

    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def push_robot_once_per_episode(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the robot in body frame at most once per episode.

    This wraps ``push_by_setting_velocity_body_frame`` with a per-env flag that
    prevents the event from firing more than once per episode, regardless of how
    short the ``interval_range_s`` is configured on the EventTerm.

    On episode reset the flag is cleared automatically because Isaac Lab calls
    the event manager's ``reset`` step which reinitializes interval timers; we
    hook into that by checking ``env_ids`` at mode="reset" time externally.
    However, the flag is cleared here by tracking which envs are being reset via
    the global registry — it is reset lazily on the next interval call when
    ``env_ids`` indicates a freshly-reset environment (step count == 0 after reset).

    Usage in config::

        push_robot = EventTerm(
            func=mdp.push_robot_once_per_episode,
            mode="interval",
            interval_range_s=(0.5, 2.0),
            params={"velocity_range": {"x": (0.8, 0.8), "y": (-0.5, 0.5)}},
        )
        # Add a reset-mode term to clear the flag each episode:
        push_robot_reset = EventTerm(
            func=mdp.reset_push_once_flag,
            mode="reset",
        )
    """
    global _push_done_registry
    env_key = id(env)

    # Initialise flag tensor lazily
    if env_key not in _push_done_registry:
        _push_done_registry[env_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    push_done = _push_done_registry[env_key]

    # Only push envs that have NOT been pushed yet this episode
    eligible = env_ids[~push_done[env_ids]]
    if eligible.numel() == 0:
        return

    push_by_setting_velocity_body_frame(env, eligible, velocity_range, asset_cfg)
    push_done[eligible] = True


def reset_push_once_flag(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    """Clear the per-episode push flag for the given environments.

    Must be registered as a separate EventTerm with mode="reset" alongside
    ``push_robot_once_per_episode`` so the flag resets at the start of each episode.
    """
    global _push_done_registry
    env_key = id(env)
    if env_key in _push_done_registry:
        _push_done_registry[env_key][env_ids] = False

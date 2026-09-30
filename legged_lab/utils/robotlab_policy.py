"""Batched adapter for RobotLab MoE-CTS student policies (TorchScript export).

RobotLab (``go2_rl_robotlab``) exports its MoE-CTS student as a stateful
TorchScript module whose ``forward`` accepts one 45-dim frame at batch size 1
and keeps a zero-initialised 10-frame history internally. That entry point is
unusable for vectorised LeggedLab collection, so this adapter calls the two
exported sub-modules directly::

    obs["policy"]  [N, H*45]   LeggedLab: frame-major history, LeggedLab joint order
        │  per frame: permute joint_pos / joint_vel / last_action → RobotLab joint order
        │  frame-major [N, H, 45] → term-major [ang×H | grav×H | cmd×H | jp×H | jv×H | act×H]
        ▼
    student_moe_encoder(hist)      → latent [N, 32]
    actor(cat(latent, frame_t))    → action [N, 12] in RobotLab joint order
        │  inverse permutation
        ▼
    action  [N, 12]  LeggedLab joint order

Only the student is used. The teacher (275-dim privileged obs incl. 187 height
rays) is not needed for collection, so the privileged schema recorded for the
world model is independent of this policy.

The adapter is stateless: history lives in ``BaseEnv.actor_obs_buffer`` whose
reset semantics (fill with first post-reset frame) match RobotLab training.
The export's own ``reset()`` zero-fills instead, which is why it is bypassed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

# RobotLab ``JOINT_NAMES`` (leg-major), from
# go2_rl_robotlab/source/robot_lab/robot_lab/tasks/go2/env_cfg.py.
ROBOTLAB_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
# Single-frame layout shared by RobotLab PolicyCfg and BaseEnv.compute_current_observations:
# base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel, last_action.
ROBOTLAB_TERM_DIMS = (3, 3, 3, 12, 12, 12)
JOINT_TERM_OFFSETS = (9, 21, 33)  # start of joint_pos / joint_vel / last_action in a frame

# RobotLab training settings the LeggedLab env must reproduce for the student
# to see in-distribution inputs.  Checked by ``check_env_compat``.
ROBOTLAB_OBS_SCALES = {
    "ang_vel": 0.25,
    "projected_gravity": 1.0,
    "commands": 1.0,
    "joint_pos": 1.0,
    "joint_vel": 0.05,
    "actions": 1.0,
}
ROBOTLAB_ACTION_SCALE = 0.25
ROBOTLAB_DEFAULT_JOINT_POS = {
    "FL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
}


class RobotLabStudentPolicy:
    """Callable ``policy(obs) -> actions`` wrapping a RobotLab exported ``policy.pt``.

    Args:
        policy_path: TorchScript file exported by RobotLab (``exported/policy.pt``).
        env_joint_names: joint names in the order the env uses for obs and actions
            (``robot.data.joint_names``).
        history_len: actor history length of the env; must equal the export's.
        device: inference device.
    """

    def __init__(self, policy_path: str, env_joint_names: Sequence[str], history_len: int, device):
        module = torch.jit.load(policy_path, map_location=device)
        module.eval()

        num_single_obs = int(module.num_single_obs)
        feature_dims = tuple(int(d) for d in module.feature_dims)
        export_history_len = int(module.history_len)
        if num_single_obs != sum(ROBOTLAB_TERM_DIMS) or feature_dims != ROBOTLAB_TERM_DIMS:
            raise ValueError(
                f"Unexpected RobotLab export layout: num_single_obs={num_single_obs}, "
                f"feature_dims={feature_dims}; expected {ROBOTLAB_TERM_DIMS}."
            )
        if export_history_len != history_len:
            raise ValueError(
                f"History length mismatch: export uses {export_history_len}, env uses {history_len}. "
                "Set robot.actor_obs_history_length to match."
            )
        if bool(module.state_dependent_std):
            raise ValueError("state_dependent_std exports are not supported by this adapter.")

        missing = [n for n in ROBOTLAB_JOINT_NAMES if n not in env_joint_names]
        if missing or len(env_joint_names) != len(ROBOTLAB_JOINT_NAMES):
            raise ValueError(f"Env joints {list(env_joint_names)} do not match RobotLab joints (missing {missing}).")

        self.device = device
        self.history_len = history_len
        self.frame_dim = num_single_obs
        self.env_joint_names = list(env_joint_names)
        self.encoder = module.student_moe_encoder
        self.actor = module.actor
        # Identity in the released checkpoints, but applied for generality.
        self.actor_obs_normalizer = module.actor_obs_normalizer
        self.single_obs_normalizer = module.single_obs_normalizer
        # robotlab_from_env[i] = env index of RobotLab joint i  (gather env → RobotLab)
        self.robotlab_from_env = torch.tensor(
            [self.env_joint_names.index(n) for n in ROBOTLAB_JOINT_NAMES], dtype=torch.long, device=device
        )
        # env_from_robotlab[j] = RobotLab index of env joint j  (gather RobotLab → env)
        self.env_from_robotlab = torch.argsort(self.robotlab_from_env)
        self._module = module  # keep the parent alive

    def to_robotlab_frames(self, obs: torch.Tensor) -> torch.Tensor:
        """[N, H*45] env layout → [N, H, 45] frames in RobotLab joint order."""
        frames = obs.reshape(obs.shape[0], self.history_len, self.frame_dim)
        out = frames.clone()
        for start in JOINT_TERM_OFFSETS:
            out[..., start:start + 12] = frames[..., start:start + 12][..., self.robotlab_from_env]
        return out

    @staticmethod
    def to_term_major(frames: torch.Tensor) -> torch.Tensor:
        """[N, H, 45] → [N, H*45] with each term's H frames contiguous (IsaacLab flatten_history_dim)."""
        blocks, offset = [], 0
        for dim in ROBOTLAB_TERM_DIMS:
            blocks.append(frames[..., offset:offset + dim].reshape(frames.shape[0], -1))
            offset += dim
        return torch.cat(blocks, dim=-1)

    @torch.inference_mode()
    def __call__(self, obs) -> torch.Tensor:
        if isinstance(obs, Mapping) or hasattr(obs, "keys"):
            obs = obs["policy"]
        obs = obs.to(self.device)
        if obs.shape[-1] != self.history_len * self.frame_dim:
            raise ValueError(f"policy obs has {obs.shape[-1]} dims, expected {self.history_len * self.frame_dim}.")
        frames = self.to_robotlab_frames(obs)
        history = self.actor_obs_normalizer(self.to_term_major(frames))
        current = self.single_obs_normalizer(frames[:, -1])
        latent = self.encoder(history)[0]
        actions = self.actor(torch.cat([latent, current], dim=-1))
        return actions[:, self.env_from_robotlab]


def check_env_compat(env, atol: float = 1e-6) -> list[str]:
    """Return mismatches between a LeggedLab ``BaseEnv`` and RobotLab training settings.

    Only covers what the student reads or what maps its output to joint targets.
    Physics differences (actuator model, friction combine mode, asset) are
    intentionally out of scope; see docs/ROBOTLAB_POLICY_COLLECTION.md.
    """
    problems = []
    for name, want in ROBOTLAB_OBS_SCALES.items():
        got = float(getattr(env.obs_scales, name))
        if abs(got - want) > atol:
            problems.append(f"obs_scales.{name}={got} (RobotLab {want})")
    if abs(float(env.action_scale) - ROBOTLAB_ACTION_SCALE) > atol:
        problems.append(f"action_scale={env.action_scale} (RobotLab {ROBOTLAB_ACTION_SCALE})")
    step_dt = env.cfg.sim.dt * env.cfg.sim.decimation
    if abs(step_dt - 0.02) > atol:
        problems.append(f"policy dt={step_dt} (RobotLab 0.02)")
    names = list(env.robot.data.joint_names)
    default = env.robot.data.default_joint_pos[0].tolist()
    for name, want in ROBOTLAB_DEFAULT_JOINT_POS.items():
        got = default[names.index(name)]
        if abs(got - want) > 1e-4:
            problems.append(f"default_joint_pos[{name}]={got:.4f} (RobotLab {want})")
    return problems

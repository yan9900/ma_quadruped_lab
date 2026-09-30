#!/usr/bin/env python3
"""Offline check: RobotLabStudentPolicy (batched) == RobotLab export's own forward (batch 1).

No Isaac Sim needed.  Random frames are generated in LeggedLab joint order and
fed through a LeggedLab-style history (first frame fills the buffer, frame-major
flatten).  The reference path converts each frame to RobotLab joint order and
drives the export's stateful batch-1 ``forward``.  Agreement to float precision
verifies the joint permutation, the frame→term reordering and the action
un-permutation together.

    python scripts/verify_robotlab_policy.py \
        --policy logs/robotlab_policies/symmetry_v1_77k_0.7006_bb4a078_20260706/exported/policy.pt
"""

import argparse
import importlib.util
import os

import torch

# Import the adapter without triggering legged_lab.utils.__init__ (which needs Isaac Lab).
_spec = importlib.util.spec_from_file_location(
    "robotlab_policy", os.path.join(os.path.dirname(__file__), "..", "utils", "robotlab_policy.py")
)
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

# LeggedLab Go2 USD joint order (type-major), as recorded in go2_robotlab_parity_manifest.json.
LEGGEDLAB_JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--num_envs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(0)
    H, D = 10, 45
    adapter = rp.RobotLabStudentPolicy(args.policy, LEGGEDLAB_JOINT_NAMES, H, "cpu")
    ref = torch.jit.load(args.policy, map_location="cpu")

    # frames in LeggedLab joint order, [T, N, 45]
    frames = torch.randn(args.steps, args.num_envs, D)

    def env_to_robotlab(frame):  # [.., 45]
        out = frame.clone()
        for s in rp.JOINT_TERM_OFFSETS:
            out[..., s:s + 12] = frame[..., s:s + 12][..., adapter.robotlab_from_env]
        return out

    # Batched adapter over a LeggedLab-style history.
    hist = frames[0].unsqueeze(1).repeat(1, H, 1)  # CircularBuffer: first push fills all slots
    adapter_actions = []
    for t in range(args.steps):
        if t > 0:
            hist = torch.cat([hist[:, 1:], frames[t].unsqueeze(1)], dim=1)
        adapter_actions.append(adapter(hist.reshape(args.num_envs, -1)))
    adapter_actions = torch.stack(adapter_actions)  # [T, N, 12] env order

    # Reference: export forward, one env at a time.
    max_err = 0.0
    for i in range(args.num_envs):
        ref.reset()  # zero-fills; pre-feed H-1 copies so the buffer equals first-frame fill
        first = env_to_robotlab(frames[0, i:i + 1])
        for _ in range(H - 1):
            ref(first)
        for t in range(args.steps):
            a_rl = ref(env_to_robotlab(frames[t, i:i + 1]))[0]
            a_env = a_rl[adapter.env_from_robotlab]
            max_err = max(max_err, (a_env - adapter_actions[t, i]).abs().max().item())

    # The permutation must be non-trivial for this test to mean anything.
    assert not torch.equal(adapter.robotlab_from_env, torch.arange(12))
    print(f"robotlab_from_env = {adapter.robotlab_from_env.tolist()}")
    print(f"max |adapter - export| over {args.num_envs} envs x {args.steps} steps = {max_err:.3e}")
    if max_err > args.tol:
        raise SystemExit(f"FAIL: exceeds tol {args.tol}")
    print("PASS")


if __name__ == "__main__":
    main()

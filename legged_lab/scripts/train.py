from legged_lab.utils import task_registry

import argparse

from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner
# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.torch as torch_utils
from legged_lab.envs import *  # noqa:F401, F403


def train():
    runner: OnPolicyRunner

    torch_utils.set_seed(args_cli.seed)
    env = task_registry.make_env(args_cli=args_cli)
    runner, agent_cfg = task_registry.make_alg_runner(env=env, args_cli=args_cli)
    runner.learn(num_learning_iterations=agent_cfg['max_iterations'], init_at_random_ep_len=True)


if __name__ == '__main__':
    train()
    simulation_app.close()

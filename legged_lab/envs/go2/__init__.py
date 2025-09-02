"""Go2 environments registration."""

import gymnasium as gym

from . import go2_config  # noqa: F401, F403

##
# Register Gym environments.
##

from legged_lab.envs.base.base_env import BaseEnv

gym.register(
    id="Go2-Fall-Recovery",
    entry_point="legged_lab.envs.base:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_config.Go2FallRecoveryFlatEnvCfg,
        "rl_games_cfg_entry_point": go2_config.Go2FallRecoveryAgentCfg,
    },
)

gym.register(
    id="Go2-Flat-v0",
    entry_point="legged_lab.envs.base:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_config.Go2FlatEnvCfg,
        "rl_games_cfg_entry_point": go2_config.Go2FlatAgentCfg,
    },
)

gym.register(
    id="Go2-Rough-v0",
    entry_point="legged_lab.envs.base:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_config.Go2RoughEnvCfg,
        "rl_games_cfg_entry_point": go2_config.Go2RoughAgentCfg,
    },
)

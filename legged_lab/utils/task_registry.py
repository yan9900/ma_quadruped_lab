from typing import Tuple
import os
from datetime import datetime
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from typing import TYPE_CHECKING
from .cli_args import update_rsl_rl_cfg, class_to_dict

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnvConfig
    from legged_lab.envs.base.base_env_config import BaseAgentConfig


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: VecEnv, env_cfg: "BaseEnvConfig", train_cfg: "BaseAgentConfig"):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple["BaseEnvConfig", "BaseAgentConfig"]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        return env_cfg, train_cfg

    def make_env(self, args_cli) -> VecEnv:
        env_class_name = args_cli.task
        env_cfg, _ = self.get_cfgs(env_class_name)
        env_class = self.get_task_class(env_class_name)
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        env = env_class(env_cfg, args_cli.headless)
        return env

    def make_alg_runner(self, env, args_cli) -> OnPolicyRunner:
        env_class_name = args_cli.task
        _, agent_cfg = self.get_cfgs(env_class_name)
        agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
        log_root_path = os.path.join("logs", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        # specify directory for logging runs: {time-stamp}_{run_name}
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)
        agent_cfg = class_to_dict(agent_cfg)
        runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=agent_cfg['device'])
        return runner, agent_cfg


# make global task registry
task_registry = TaskRegistry()

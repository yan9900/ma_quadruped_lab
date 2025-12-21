"""
Go2 Safety Monitor for LeggedLab/IsaacSim

这个模块提供安全监督功能，基于世界模型的Margin Head实时评估机器人状态安全性。
当检测到危险状态时，会过滤掉不安全的动作。

工作原理:
1. 世界模型 Encoder 将观测（深度图 + 状态）编码到 latent space
2. RSSM 维护时序状态
3. Margin Head 从 latent feature 预测安全边界 g(x)
4. 当 g(x) < threshold 时，触发安全过滤，使用后备动作

使用方法:
    from safety_monitor import Go2SafetyMonitor
    
    monitor = Go2SafetyMonitor(
        world_model_path="path/to/rssm_ckpt.pt",
        config_path="path/to/configs.yaml",  # 可选
        threshold=0.0,
        device="cuda:0"
    )
    
    # 在评估循环中
    for step in range(max_steps):
        action_ppo = policy(obs_dict)
        action, filtered, g_val = monitor.filter_action(obs_dict, action_ppo)
        obs_dict, reward, done, info = env.step(action)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque
import sys


@dataclass 
class SafetyStats:
    """安全统计信息"""
    total_steps: int = 0
    filter_triggers: int = 0
    episodes: int = 0
    falls_prevented: int = 0  # 过滤后仍然摔倒的次数（用于评估）
    g_values: deque = field(default_factory=lambda: deque(maxlen=50000))
    
    def log_step(self, g_value: float, was_filtered: bool):
        self.total_steps += 1
        self.g_values.append(g_value)
        if was_filtered:
            self.filter_triggers += 1
    
    def log_episode_end(self, fell: bool = False):
        self.episodes += 1
        if fell:
            self.falls_prevented += 1
    
    def summary(self) -> Dict[str, float]:
        g_array = np.array(self.g_values) if len(self.g_values) > 0 else np.array([0.0])
        return {
            "total_steps": self.total_steps,
            "total_episodes": self.episodes,
            "filter_rate": self.filter_triggers / max(self.total_steps, 1) * 100,
            "fall_rate": self.falls_prevented / max(self.episodes, 1) * 100,
            "g_mean": float(np.mean(g_array)),
            "g_std": float(np.std(g_array)),
            "g_min": float(np.min(g_array)),
            "g_max": float(np.max(g_array)),
            "g_negative_rate": float(np.mean(g_array < 0)) * 100,
        }


class Go2SafetyMonitor:
    """
    Go2 安全监督器
    
    使用世界模型的 Margin Head 实时评估状态安全性，
    并在检测到危险时过滤动作。
    """
    
    def __init__(
        self, 
        world_model_path: str,
        config_path: Optional[str] = None,
        config: Optional[Any] = None,
        threshold: float = 0.0,
        fallback_strategy: str = "stop",
        device: str = "cuda:0",
        image_key: str = "depth",  # LeggedLab中深度图的key
        state_key: str = "policy",  # LeggedLab中状态的key
        env: Optional[Any] = None,  # LeggedLab环境对象（用于直接获取相机数据）
    ):
        """
        Args:
            world_model_path: 世界模型 checkpoint 路径
            config_path: 配置文件路径 (configs.yaml)
            config: 配置对象 (如果提供则忽略 config_path)
            threshold: 安全阈值，g(x) < threshold 被认为危险
            fallback_strategy: 后备策略 
                - "stop": 零动作（停止）
                - "previous": 使用上一个安全动作
                - "zero_velocity": 尝试保持当前位置
            device: 计算设备
            image_key: 观测字典中深度图的键名
            state_key: 观测字典中状态向量的键名
            env: LeggedLab环境对象（用于直接获取相机数据）
        """
        self.device = device
        self.threshold = threshold
        self.fallback_strategy = fallback_strategy
        self.image_key = image_key
        self.state_key = state_key
        self.env = env  # 保存环境引用
        
        # 加载配置
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            # 使用默认配置
            self.config = self._get_default_config()
        
        # 添加必要的属性
        self.config.device = device
        
        # 加载世界模型
        self._load_world_model(world_model_path)
        
        # 状态管理
        self.latent = None
        self.is_first = True
        self.last_safe_action = None
        self.action_history = deque(maxlen=10)
        
        # 统计信息
        self.stats = SafetyStats()
        
        # 计算 feature 维度
        if hasattr(self.config, 'dyn_discrete') and self.config.dyn_discrete:
            self.feat_size = self.config.dyn_stoch * self.config.dyn_discrete + self.config.dyn_deter
        else:
            self.feat_size = getattr(self.config, 'dyn_stoch', 32) + getattr(self.config, 'dyn_deter', 512)
        
        print(f"[SafetyMonitor] Initialized with threshold={threshold}")
        print(f"[SafetyMonitor] Feature size: {self.feat_size}")
        print(f"[SafetyMonitor] Fallback strategy: {fallback_strategy}")
    
    def _load_config(self, config_path: str) -> Any:
        """从YAML文件加载配置"""
        try:
            import ruamel.yaml as yaml_loader
            yml = yaml_loader.YAML(typ="safe", pure=True)
            with open(config_path) as f:
                config_dict = yml.load(f)
        except ImportError:
            # Fallback to standard yaml
            import yaml as yaml_loader
            with open(config_path) as f:
                config_dict = yaml_loader.safe_load(f)
        
        from types import SimpleNamespace
        
        # 合并 defaults 和 go2 配置
        defaults = config_dict.get('defaults', {})
        go2_config = config_dict.get('go2', config_dict.get('go2_default', {}))
        
        # 深度合并
        merged = {**defaults, **go2_config}
        
        # 处理字符串数值转换 (YAML可能把1e-4加载为字符串)
        def convert_numeric(obj):
            if isinstance(obj, dict):
                return {k: convert_numeric(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numeric(v) for v in obj]
            elif isinstance(obj, str):
                # 尝试转换为数值
                try:
                    if '.' in obj or 'e' in obj.lower():
                        return float(obj)
                    return int(obj)
                except ValueError:
                    return obj
            return obj
        
        merged = convert_numeric(merged)
        
        # 注意: encoder, decoder 等需要保持为字典格式，因为WorldModel使用**解包
        # 只把顶层转换为SimpleNamespace
        config = SimpleNamespace(**merged)
        
        return config
    
    def _get_default_config(self) -> Any:
        """获取默认配置"""
        from types import SimpleNamespace
        
        encoder = SimpleNamespace(
            mlp_keys='obs_state',
            cnn_keys='image',
            act='SiLU',
            norm=True,
            cnn_depth=32,
            kernel_size=4,
            minres=4,
            mlp_layers=5,
            mlp_units=1024,
            symlog_inputs=True
        )
        
        decoder = SimpleNamespace(
            mlp_keys='obs_state',
            cnn_keys='image',
            act='SiLU',
            norm=True,
            cnn_depth=32,
            kernel_size=4,
            minres=4,
            mlp_layers=5,
            mlp_units=1024,
            cnn_sigmoid=False,
            image_dist='mse',
            vector_dist='symlog_mse',
            outscale=1.0
        )
        
        margin_head = SimpleNamespace(
            layers=3,
            loss_scale=5.0
        )
        
        cont_head = SimpleNamespace(
            layers=2,
            loss_scale=1.0,
            outscale=1.0
        )
        
        return SimpleNamespace(
            # RSSM
            dyn_stoch=32,
            dyn_deter=512,
            dyn_hidden=512,
            dyn_discrete=32,
            dyn_rec_depth=1,
            dyn_mean_act='none',
            dyn_std_act='sigmoid2',
            dyn_min_std=0.1,
            unimix_ratio=0.01,
            initial='learned',
            
            # Network
            units=512,
            act='SiLU',
            norm=True,
            
            # Heads
            encoder=encoder,
            decoder=decoder,
            margin_head=margin_head,
            cont_head=cont_head,
            grad_heads=['decoder', 'margin', 'cont'],
            
            # Optimizer (不实际使用，但模型初始化需要)
            model_lr=1e-4,
            opt_eps=1e-8,
            grad_clip=1000,
            weight_decay=0.0,
            opt='adam',
            precision=32,
            
            # 观测空间
            size=[64],
            obs_state_dim=45,
            num_actions=12,
        )
    
    def _load_world_model(self, path: str):
        """加载世界模型"""
        # 添加 dreamerv3 路径
        dreamer_paths = [
            Path(__file__).parent.parent.parent.parent / "latent-safety" / "dreamerv3-torch",
            Path("/home/lcy/latent-safety/dreamerv3-torch"),
        ]
        
        for dreamer_path in dreamer_paths:
            if dreamer_path.exists() and str(dreamer_path) not in sys.path:
                sys.path.insert(0, str(dreamer_path))
                print(f"[SafetyMonitor] Added dreamerv3 path: {dreamer_path}")
                break
        
        import models
        import gymnasium as gym
        
        # 创建观测空间 (与训练时一致)
        # Go2配置使用 image_height/image_width 而不是 size
        if hasattr(self.config, 'image_height'):
            image_size = getattr(self.config, 'image_height', 64)
        elif hasattr(self.config, 'size'):
            size_val = getattr(self.config, 'size', [64])
            image_size = size_val[0] if isinstance(size_val, list) else size_val
        else:
            image_size = 64
        
        obs_state_dim = getattr(self.config, 'obs_state_dim', 45)
        
        # 打印调试信息
        print(f"[SafetyMonitor] Using image_size={image_size}, obs_state_dim={obs_state_dim}")
        
        obs_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (image_size, image_size, 1), np.uint8),
            'obs_state': gym.spaces.Box(-np.inf, np.inf, (obs_state_dim,), np.float32),
            'is_first': gym.spaces.Box(0., 1., (1,)),
            'is_last': gym.spaces.Box(0., 1., (1,)),
            'is_terminal': gym.spaces.Box(0., 1., (1,)),
        })
        
        act_space = gym.spaces.Box(-1., 1., (12,), np.float32)
        
        # 设置 num_actions
        self.config.num_actions = 12
        
        # 创建世界模型
        self.wm = models.WorldModel(obs_space, act_space, 0, self.config)
        
        # 加载权重
        checkpoint = torch.load(path, map_location=self.device)
        
        # 处理不同格式的checkpoint
        if isinstance(checkpoint, dict):
            if 'agent_state_dict' in checkpoint:
                # Dreamer 完整checkpoint格式
                state_dict = {}
                for k, v in checkpoint['agent_state_dict'].items():
                    if k.startswith('_wm.'):
                        state_dict[k[4:]] = v  # 去掉 '_wm.' 前缀
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # 直接是state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 加载权重
        self.wm.load_state_dict(state_dict, strict=False)
        self.wm.eval()
        self.wm.to(self.device)
        
        # 保存组件引用
        self.encoder = self.wm.encoder
        self.dynamics = self.wm.dynamics
        
        print(f"[SafetyMonitor] Loaded world model from {path}")
        print(f"[SafetyMonitor] World model device: {next(self.wm.parameters()).device}")
    
    def reset(self):
        """重置内部状态 (在 episode 开始时调用)"""
        self.latent = None
        self.is_first = True
        self.last_safe_action = None
        self.action_history.clear()
    
    def _preprocess_obs(
        self, 
        obs_dict: Dict[str, torch.Tensor],
        env_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        预处理LeggedLab的观测数据，转换为世界模型的输入格式
        
        Args:
            obs_dict: LeggedLab环境的观测字典
            env_ids: 要处理的环境索引 (如果为None则处理所有环境)
        
        Returns:
            处理后的数据字典，格式符合世界模型输入要求
        """
        # 获取深度图 - 优先从环境对象的相机传感器获取
        depth_image = None
        
        # 方式1: 从环境对象的相机传感器获取
        if self.env is not None:
            try:
                if hasattr(self.env, 'scene') and hasattr(self.env.scene, 'sensors'):
                    if 'front_camera' in self.env.scene.sensors:
                        camera = self.env.scene.sensors['front_camera']
                        if hasattr(camera, 'data') and hasattr(camera.data, 'output'):
                            output_data = camera.data.output
                            if 'distance_to_image_plane' in output_data:
                                depth_image = output_data['distance_to_image_plane']
            except Exception as e:
                print(f"[SafetyMonitor] Warning: Failed to get camera data from env: {e}")
        
        # 方式2: 从obs_dict获取
        if depth_image is None:
            for key in ['depth', 'depth_image', 'camera', self.image_key]:
                if key in obs_dict:
                    depth_image = obs_dict[key]
                    break
        
        if depth_image is None:
            raise ValueError(f"Cannot find depth image in obs_dict. Available keys: {list(obs_dict.keys())}")
        
        # 获取状态向量
        # LeggedLab中可能的键名: 'policy', 'obs', 'proprioceptive'
        obs_state = None
        for key in ['policy', 'obs', 'proprioceptive', self.state_key]:
            if key in obs_dict:
                obs_state = obs_dict[key]
                break
        
        if obs_state is None:
            raise ValueError(f"Cannot find state vector in obs_dict. Available keys: {list(obs_dict.keys())}")
        
        # 如果指定了env_ids，只处理这些环境
        if env_ids is not None:
            depth_image = depth_image[env_ids]
            obs_state = obs_state[env_ids]
        
        batch_size = depth_image.shape[0]
        
        # 处理深度图
        # LeggedLab格式: (N, H, W) 或 (N, 1, H, W)
        # 世界模型格式: (N, T, H, W, C) 其中 T=1, C=1
        if depth_image.dim() == 3:  # (N, H, W)
            depth_image = depth_image.unsqueeze(1).unsqueeze(-1)  # (N, 1, H, W, 1)
        elif depth_image.dim() == 4:  # (N, 1, H, W) 或 (N, H, W, 1)
            if depth_image.shape[1] == 1:  # (N, 1, H, W)
                depth_image = depth_image.permute(0, 2, 3, 1).unsqueeze(1)  # (N, 1, H, W, 1)
            else:  # (N, H, W, 1)
                depth_image = depth_image.unsqueeze(1)  # (N, 1, H, W, 1)
        
        # 归一化到 0-255 范围
        if depth_image.max() <= 1.0:
            depth_image = depth_image * 255.0
        
        # 处理状态向量
        # 世界模型格式: (N, T, D) 其中 T=1
        if obs_state.dim() == 2:  # (N, D)
            obs_state = obs_state.unsqueeze(1)  # (N, 1, D)
        
        # 构建数据字典
        data = {
            'image': depth_image.float().to(self.device),
            'obs_state': obs_state.float().to(self.device),
            'is_first': torch.ones((batch_size, 1), device=self.device) if self.is_first 
                       else torch.zeros((batch_size, 1), device=self.device),
            'is_last': torch.zeros((batch_size, 1), device=self.device),
            'is_terminal': torch.zeros((batch_size, 1), device=self.device),
        }
        
        # 动作 (用于 RSSM observe)
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            if isinstance(last_action, np.ndarray):
                last_action = torch.from_numpy(last_action)
            data['action'] = last_action.float().to(self.device)
            if data['action'].dim() == 2:  # (N, A)
                data['action'] = data['action'].unsqueeze(1)  # (N, 1, A)
        else:
            data['action'] = torch.zeros((batch_size, 1, 12), device=self.device)
        
        return data
    
    @torch.no_grad()
    def compute_safety(
        self, 
        obs_dict: Dict[str, torch.Tensor],
        env_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算当前观测的安全边界值
        
        Args:
            obs_dict: LeggedLab环境的观测字典
            env_ids: 要处理的环境索引
        
        Returns:
            g_values: 安全边界值张量 (正=安全, 负=危险)
            latent: 当前 latent 状态字典
        """
        # 预处理
        data = self._preprocess_obs(obs_dict, env_ids)
        processed = self.wm.preprocess(data)
        
        # 编码
        embed = self.encoder(processed)
        
        # 更新 latent
        self.latent, _ = self.dynamics.observe(
            embed,
            data['action'],
            processed['is_first']
        )
        
        if self.is_first:
            self.is_first = False
        
        # 计算安全边界
        # latent shape: {stoch: (N, T, ...), deter: (N, T, D)}
        latent_last = {k: v[:, -1:] for k, v in self.latent.items()}
        feat = self.dynamics.get_feat(latent_last)  # (N, 1, F)
        
        # Margin head 输出
        g_raw = self.wm.heads['margin'](feat)  # (N, 1, 1) 或 (N, 1)
        g_values = torch.tanh(g_raw).squeeze(-1).squeeze(-1)  # (N,)
        
        return g_values, latent_last
    
    def get_fallback_action(
        self, 
        batch_size: int,
        original_action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取安全后备动作
        
        Args:
            batch_size: 需要的动作数量
            original_action: 原始动作 (用于某些策略)
        
        Returns:
            fallback_action: 后备动作张量 (N, 12)
        """
        if self.fallback_strategy == "stop":
            # 策略 1: 零动作（停止）
            return torch.zeros((batch_size, 12), device=self.device)
        
        elif self.fallback_strategy == "previous":
            # 策略 2: 使用上一个安全动作
            if self.last_safe_action is not None:
                action = self.last_safe_action
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action)
                # 扩展到batch_size
                if action.shape[0] != batch_size:
                    action = action[0:1].expand(batch_size, -1)
                return action.to(self.device)
            else:
                return torch.zeros((batch_size, 12), device=self.device)
        
        elif self.fallback_strategy == "zero_velocity":
            # 策略 3: 小幅制动动作
            # 这里返回一个小的负值，尝试减速
            return torch.full((batch_size, 12), -0.1, device=self.device)
        
        else:
            return torch.zeros((batch_size, 12), device=self.device)
    
    def filter_action(
        self, 
        obs_dict: Dict[str, torch.Tensor],
        action: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        安全过滤动作 (支持批量处理)
        
        Args:
            obs_dict: LeggedLab环境的观测字典
            action: 策略建议的动作 (N, 12)
            env_ids: 要处理的环境索引
        
        Returns:
            filtered_action: 过滤后的动作 (N, 12)
            was_filtered: 是否进行了过滤的布尔张量 (N,)
            g_values: 安全边界值 (N,)
        """
        # 确保action在正确的设备上
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        action = action.to(self.device)
        
        batch_size = action.shape[0]
        
        # 计算安全边界
        g_values, _ = self.compute_safety(obs_dict, env_ids)
        
        # 判断哪些环境需要过滤
        was_filtered = g_values < self.threshold  # (N,) bool
        
        # 获取后备动作
        fallback_action = self.get_fallback_action(batch_size, action)
        
        # 根据安全性选择动作
        # was_filtered: True -> 使用fallback, False -> 使用original
        filtered_action = torch.where(
            was_filtered.unsqueeze(-1).expand(-1, 12),
            fallback_action,
            action
        )
        
        # 更新状态
        safe_mask = ~was_filtered
        if safe_mask.any():
            self.last_safe_action = action[safe_mask].clone()
        self.action_history.append(filtered_action.clone())
        
        # 记录统计
        for i in range(batch_size):
            self.stats.log_step(g_values[i].item(), was_filtered[i].item())
        
        return filtered_action, was_filtered, g_values
    
    def get_safety_values(
        self, 
        obs_dict: Dict[str, torch.Tensor],
        env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        仅获取安全边界值（不过滤动作）
        
        Args:
            obs_dict: LeggedLab环境的观测字典
            env_ids: 要处理的环境索引
        
        Returns:
            g_values: 安全边界值 (N,)
        """
        g_values, _ = self.compute_safety(obs_dict, env_ids)
        return g_values
    
    def get_stats_summary(self) -> Dict[str, float]:
        """获取统计摘要"""
        return self.stats.summary()
    
    def print_stats(self):
        """打印统计信息"""
        summary = self.get_stats_summary()
        print("\n" + "=" * 60)
        print("Safety Monitor Statistics")
        print("=" * 60)
        print(f"Total steps:           {summary['total_steps']}")
        print(f"Total episodes:        {summary['total_episodes']}")
        print(f"Filter trigger rate:   {summary['filter_rate']:.2f}%")
        print(f"Fall rate (with filter):{summary['fall_rate']:.2f}%")
        print(f"g(x) mean:             {summary['g_mean']:.4f}")
        print(f"g(x) std:              {summary['g_std']:.4f}")
        print(f"g(x) min:              {summary['g_min']:.4f}")
        print(f"g(x) max:              {summary['g_max']:.4f}")
        print(f"g(x) < 0 rate:         {summary['g_negative_rate']:.2f}%")
        print("=" * 60 + "\n")


# ==================== 便捷函数 ====================

def create_safety_monitor(
    world_model_path: str,
    config_path: Optional[str] = None,
    threshold: float = 0.0,
    device: str = "cuda:0",
    fallback_strategy: str = "stop"
) -> Go2SafetyMonitor:
    """
    创建安全监督器的便捷函数
    
    Args:
        world_model_path: 世界模型路径
        config_path: 配置文件路径 (如果为 None，使用默认配置)
        threshold: 安全阈值
        device: 计算设备
        fallback_strategy: 后备策略
    
    Returns:
        Go2SafetyMonitor 实例
    """
    return Go2SafetyMonitor(
        world_model_path=world_model_path,
        config_path=config_path,
        threshold=threshold,
        fallback_strategy=fallback_strategy,
        device=device
    )


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """
    测试 Safety Monitor 加载
    """
    print("=" * 60)
    print("Go2 Safety Monitor - Test")
    print("=" * 60)
    
    # 测试配置
    test_wm_path = "/home/lcy/latent-safety/logs/go2_world_model_v2/rssm_ckpt.pt"
    test_config_path = "/home/lcy/latent-safety/configs.yaml"
    
    try:
        monitor = create_safety_monitor(
            world_model_path=test_wm_path,
            config_path=test_config_path,
            threshold=0.0,
            device="cuda:0"
        )
        print("\n[SUCCESS] Safety Monitor created successfully!")
        print(f"  - Feature size: {monitor.feat_size}")
        print(f"  - Threshold: {monitor.threshold}")
        print(f"  - Fallback strategy: {monitor.fallback_strategy}")
        
        # 测试预处理（使用假数据）
        fake_obs = {
            'depth': torch.randn(4, 64, 64),  # (N, H, W)
            'policy': torch.randn(4, 45),      # (N, D)
        }
        
        g_values, _ = monitor.compute_safety(fake_obs)
        print(f"\n[TEST] Safety values for fake data: {g_values}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to create Safety Monitor: {e}")
        import traceback
        traceback.print_exc()

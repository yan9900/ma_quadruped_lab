#!/usr/bin/env python3
"""
机器人注册系统详解
展示如何将新机器人和任务注册到框架中
"""

# 模拟框架的注册系统
class MockTaskRegistry:
    """模拟任务注册系统"""
    
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
        print("🏗️  任务注册系统初始化")
    
    def register(self, task_name: str, env_class, env_cfg, agent_cfg):
        """注册一个新任务"""
        self.task_classes[task_name] = env_class
        self.env_cfgs[task_name] = env_cfg
        self.train_cfgs[task_name] = agent_cfg
        
        print(f"✅ 已注册任务: '{task_name}'")
        print(f"   - 环境类: {env_class.__name__}")
        print(f"   - 环境配置: {env_cfg.__class__.__name__}")
        print(f"   - 智能体配置: {agent_cfg.__class__.__name__}")
        print()
    
    def get_task_info(self, task_name: str):
        """获取任务信息"""
        if task_name not in self.task_classes:
            raise ValueError(f"任务 '{task_name}' 未注册!")
        
        return {
            "env_class": self.task_classes[task_name],
            "env_cfg": self.env_cfgs[task_name],
            "agent_cfg": self.train_cfgs[task_name]
        }
    
    def list_all_tasks(self):
        """列出所有已注册的任务"""
        print(f"📋 已注册的任务 ({len(self.task_classes)} 个):")
        for i, task_name in enumerate(self.task_classes.keys(), 1):
            env_cfg = self.env_cfgs[task_name]
            print(f"   {i}. {task_name}")
            print(f"      └─ 机器人: {getattr(env_cfg, 'robot_name', '未知')}")


# 模拟配置类
class MockBaseEnv:
    """模拟基础环境类"""
    def __init__(self, cfg, headless=True):
        self.cfg = cfg
        print(f"🤖 创建环境: {cfg.robot_name} ({cfg.task_type})")

class MockEnvConfig:
    """模拟环境配置"""
    def __init__(self, robot_name: str, task_type: str):
        self.robot_name = robot_name
        self.task_type = task_type
        self.max_episode_length = 1000
        print(f"⚙️  创建环境配置: {robot_name}_{task_type}")

class MockAgentConfig:
    """模拟智能体配置"""
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.max_iterations = 10000
        print(f"🧠 创建智能体配置: {experiment_name}")


def demonstrate_robot_registration():
    """演示机器人注册过程"""
    
    print("=" * 70)
    print("🚀 LeggedLab 机器人注册系统演示")
    print("=" * 70)
    
    # 1. 创建注册系统
    registry = MockTaskRegistry()
    
    print("\n" + "=" * 50)
    print("📋 第一步：注册现有机器人任务")
    print("=" * 50)
    
    # 2. 注册现有机器人（模拟你的框架）
    robots_and_tasks = [
        ("go2", "flat"),
        ("go2", "rough"), 
        ("go2", "fall_recovery"),  # 这是你添加的！
        ("g1", "flat"),
        ("h1", "rough"),
    ]
    
    for robot, task in robots_and_tasks:
        task_name = f"{robot}_{task}"
        env_cfg = MockEnvConfig(robot, task)
        agent_cfg = MockAgentConfig(f"{robot}_{task}_experiment")
        
        registry.register(task_name, MockBaseEnv, env_cfg, agent_cfg)
    
    print("=" * 50)
    print("📝 第二步：查看注册结果")
    print("=" * 50)
    
    # 3. 展示注册结果
    registry.list_all_tasks()
    
    print("\n" + "=" * 50)
    print("🔍 第三步：演示任务查找过程")
    print("=" * 50)
    
    # 4. 模拟训练脚本的查找过程
    def simulate_training_command(task_name: str):
        """模拟 python train.py --task {task_name} 的过程"""
        print(f"\n🏃 执行: python train.py --task {task_name}")
        
        try:
            task_info = registry.get_task_info(task_name)
            print(f"✅ 找到任务配置:")
            print(f"   - 环境类: {task_info['env_class'].__name__}")
            print(f"   - 机器人: {task_info['env_cfg'].robot_name}")
            print(f"   - 任务类型: {task_info['env_cfg'].task_type}")
            print(f"   - 实验名称: {task_info['agent_cfg'].experiment_name}")
            return True
        except ValueError as e:
            print(f"❌ 错误: {e}")
            return False
    
    # 测试不同的训练命令
    test_commands = [
        "go2_fall_recovery",  # 你的任务
        "g1_flat", 
        "new_robot_jump",  # 不存在的任务
    ]
    
    for cmd in test_commands:
        simulate_training_command(cmd)
    
    print("\n" + "=" * 70)
    print("💡 关键洞察:")
    print("=" * 70)
    
    insights = [
        "1. 注册是框架的 '目录系统' - 任务名 → 配置映射",
        "2. 所有任务都通过统一接口访问，无需硬编码",
        "3. 添加新机器人 = 创建配置类 + 一行注册代码",
        "4. 框架自动处理任务发现和配置加载",
        "5. 这种设计让框架高度可扩展"
    ]
    
    for insight in insights:
        print(insight)
    
    print("\n" + "=" * 70)
    print("🛠️  添加新机器人的步骤:")
    print("=" * 70)
    
    steps = [
        "1. 创建机器人资产配置 (如 NEW_ROBOT_CFG)",
        "2. 创建奖励配置类 (如 NewRobotRewardCfg)", 
        "3. 创建环境配置类 (如 NewRobotEnvCfg)",
        "4. 创建智能体配置类 (如 NewRobotAgentCfg)",
        "5. 在 __init__.py 中导入并注册:",
        "   task_registry.register('new_robot_task', BaseEnv, NewRobotEnvCfg(), NewRobotAgentCfg())"
    ]
    
    for step in steps:
        print(step)

if __name__ == "__main__":
    demonstrate_robot_registration()

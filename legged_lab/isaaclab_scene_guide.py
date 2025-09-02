#!/usr/bin/env python3
"""
Isaac Lab Scene 代码结构分析指南
展示如何查看和理解 isaaclab.scene 模块
"""

def analyze_isaaclab_scene():
    """分析 Isaac Lab Scene 模块的代码结构"""
    
    print("=" * 80)
    print("🔍 Isaac Lab Scene 代码位置和结构分析")
    print("=" * 80)
    
    print("\n📂 代码位置:")
    locations = [
        "/home/lcy/IsaacLab/source/isaaclab/isaaclab/scene/",
        "   ├── __init__.py                    # 模块初始化和导出",
        "   ├── interactive_scene.py           # 核心场景类",
        "   └── interactive_scene_cfg.py       # 场景配置类",
    ]
    
    for location in locations:
        print(location)
    
    print("\n🔧 如何查看代码:")
    commands = [
        "# 1. 查看模块概览",
        "cat /home/lcy/IsaacLab/source/isaaclab/isaaclab/scene/__init__.py",
        "",
        "# 2. 查看核心场景类",
        "less /home/lcy/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py",
        "",  
        "# 3. 查看配置类",
        "less /home/lcy/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py",
        "",
        "# 4. 搜索特定功能",
        "grep -n 'clone_environments' /home/lcy/IsaacLab/source/isaaclab/isaaclab/scene/*.py",
    ]
    
    for cmd in commands:
        print(cmd)
    
    print("\n📋 关键类和方法:")
    
    key_components = {
        "InteractiveScene": [
            "__init__(self, cfg: InteractiveSceneCfg)",
            "clone_environments()",
            "reset(env_ids: torch.Tensor)",
            "write_to_sim()",
            "update_from_sim()",
            "各类资产访问方法: articulations, rigid_objects, sensors等"
        ],
        "InteractiveSceneCfg": [
            "num_envs: int",
            "env_spacing: float", 
            "replicate_physics: bool",
            "viewer: ViewerCfg",
            "各种资产配置属性"
        ]
    }
    
    for class_name, methods in key_components.items():
        print(f"\n🔸 {class_name}:")
        for method in methods:
            print(f"   • {method}")
    
    print("\n" + "=" * 80)
    print("💡 你的项目中如何使用 Scene")
    print("=" * 80)
    
    print("\n在你的 BaseEnv 中:")
    usage_example = """
# BaseEnv.__init__ 中的场景创建:
self.scene = InteractiveScene(self.cfg.scene, device=self.device)

# 这里 self.cfg.scene 就是 BaseSceneCfg 的实例
# BaseSceneCfg 继承自 InteractiveSceneCfg

# 常用操作:
robot = self.scene["robot"]              # 访问机器人
self.scene.reset(env_ids)               # 重置指定环境
self.scene.write_to_sim()               # 写入仿真
self.scene.update_from_sim()            # 从仿真更新
    """
    print(usage_example)
    
    print("\n在你的配置中:")
    config_example = """
# BaseSceneCfg 中的场景配置:
@configclass
class BaseSceneCfg:
    seed: int = 42
    num_envs: int = 4096
    env_spacing: float = 2.5
    
    # 机器人配置
    robot: ArticulationCfg = MISSING
    
    # 地形配置
    terrain_generator: TerrainGeneratorCfg | None = None
    terrain_type: str = "generator"
    
    # 传感器配置
    height_scanner: RayCasterCfg = RayCasterCfg(...)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(...)
    """
    print(config_example)
    
    print("\n" + "=" * 80)
    print("🛠️  实用的代码查看技巧")
    print("=" * 80)
    
    tips = [
        "1. 使用 VS Code 打开整个 IsaacLab 目录:",
        "   code /home/lcy/IsaacLab",
        "",
        "2. 使用搜索功能快速定位:",
        "   Ctrl+Shift+F 搜索 'InteractiveScene'", 
        "",
        "3. 查看类的继承关系:",
        "   右键点击类名 → Go to Definition",
        "",
        "4. 查看方法的使用示例:",
        "   Ctrl+Shift+F 搜索方法名",
        "",
        "5. 使用命令行快速预览:",
        "   grep -A 10 -B 5 'class InteractiveScene' /path/to/file.py"
    ]
    
    for tip in tips:
        print(tip)
    
    print("\n" + "=" * 80)
    print("🎯 重点关注的文件和概念")
    print("=" * 80)
    
    focus_areas = {
        "🤖 机器人相关": [
            "isaaclab/assets/articulation/articulation.py",
            "isaaclab/actuators/actuator_*.py"
        ],
        "🌍 场景管理": [
            "isaaclab/scene/interactive_scene.py",
            "isaaclab/terrains/terrain_*.py"
        ],
        "📊 传感器系统": [
            "isaaclab/sensors/ray_caster/ray_caster.py",
            "isaaclab/sensors/contact_sensor/contact_sensor.py"
        ],
        "⚙️  管理器模块": [
            "isaaclab/managers/action_manager.py",
            "isaaclab/managers/observation_manager.py",
            "isaaclab/managers/reward_manager.py"
        ]
    }
    
    for category, files in focus_areas.items():
        print(f"\n{category}:")
        for file in files:
            print(f"   📄 {file}")
    
    print(f"\n💫 快速开始:")
    quick_start = """
# 在VS Code中打开Isaac Lab源码:
code /home/lcy/IsaacLab/source/isaaclab

# 或者用你喜欢的编辑器:
subl /home/lcy/IsaacLab/source/isaaclab
vim /home/lcy/IsaacLab/source/isaaclab

# 重点文件路径:
/home/lcy/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py
    """
    print(quick_start)

if __name__ == "__main__":
    analyze_isaaclab_scene()

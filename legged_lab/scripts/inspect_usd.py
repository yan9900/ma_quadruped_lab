#!/usr/bin/env python3
"""
USD文件结构查看工具
用法: python inspect_usd.py path/to/file.usd
"""

import argparse
import sys
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Gf

def format_value(value):
    """格式化属性值为可读字符串"""
    if isinstance(value, (Gf.Vec3d, Gf.Vec3f)):
        return f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"
    elif isinstance(value, (Gf.Quatd, Gf.Quatf)):
        return f"quat({value.real:.3f}, {value.imaginary[0]:.3f}, {value.imaginary[1]:.3f}, {value.imaginary[2]:.3f})"
    elif isinstance(value, (float, int)):
        return f"{value:.3f}"
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, list):
        return f"[{len(value)} items]"
    else:
        return str(value)

def analyze_prim_properties(prim):
    """分析prim的重要属性"""
    properties = {}
    
    # 物理属性
    if prim.HasAttribute('physics:mass'):
        properties['mass'] = prim.GetAttribute('physics:mass').Get()
    
    # 关节属性
    if prim.GetTypeName() == 'PhysicsRevoluteJoint':
        if prim.HasAttribute('physics:axis'):
            properties['joint_axis'] = prim.GetAttribute('physics:axis').Get()
        if prim.HasAttribute('physics:lowerLimit'):
            properties['lower_limit'] = prim.GetAttribute('physics:lowerLimit').Get()
        if prim.HasAttribute('physics:upperLimit'):
            properties['upper_limit'] = prim.GetAttribute('physics:upperLimit').Get()
    
    # 几何属性
    if prim.HasAttribute('extent'):
        properties['extent'] = prim.GetAttribute('extent').Get()
    
    # 变换属性
    if UsdGeom.Xform(prim):
        xform = UsdGeom.Xform(prim)
        if xform.GetXformOpOrderAttr().Get():
            properties['xform_ops'] = xform.GetXformOpOrderAttr().Get()
    
    return properties

def print_usd_tree(prim, depth=0, max_depth=3):
    """打印USD树状结构"""
    if depth > max_depth:
        return
    
    indent = "  " * depth
    prim_type = prim.GetTypeName() or "Scope"
    
    # 根据类型选择图标
    icons = {
        'Xform': '🔧',
        'PhysicsRevoluteJoint': '⚙️',
        'PhysicsPrismaticJoint': '🔩', 
        'Mesh': '📐',
        'Material': '🎨',
        'Shader': '✨',
        'Scope': '📁',
        'Camera': '📷',
    }
    icon = icons.get(prim_type, '📦')
    
    print(f"{indent}{icon} {prim.GetName()} ({prim_type})")
    
    # 显示重要属性
    properties = analyze_prim_properties(prim)
    for key, value in list(properties.items())[:3]:  # 只显示前3个属性
        formatted_value = format_value(value)
        print(f"{indent}   └─ {key}: {formatted_value}")
    
    # 递归子节点
    for child in prim.GetChildren():
        print_usd_tree(child, depth + 1, max_depth)

def inspect_usd_file(usd_path):
    """主要的USD文件检查函数"""
    if not Path(usd_path).exists():
        print(f"❌ 文件不存在: {usd_path}")
        return
    
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"❌ 无法打开USD文件: {usd_path}")
        return
    
    print(f"🎯 分析 USD 文件: {usd_path}")
    print("=" * 60)
    
    # 文件基本信息
    print(f"📊 基本信息:")
    try:
        up_axis = UsdGeom.GetStageUpAxis(stage)
        print(f"   └─ 上轴: {up_axis}")
    except:
        print(f"   └─ 上轴: 未设置")
        
    try:
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        print(f"   └─ 单位: {meters_per_unit} 米")
    except:
        print(f"   └─ 单位: 未设置")
    
    # 子层信息
    sublayers = stage.GetRootLayer().subLayerPaths
    if sublayers:
        print(f"   └─ 子层: {sublayers}")
    
    print()
    
    # 打印树状结构
    print("🏗️  文件结构:")
    root = stage.GetPseudoRoot()
    print_usd_tree(root)
    
    print()
    
    # 统计信息
    all_prims = [prim for prim in stage.Traverse()]
    joint_count = len([p for p in all_prims if 'Joint' in p.GetTypeName()])
    mesh_count = len([p for p in all_prims if p.GetTypeName() == 'Mesh'])
    xform_count = len([p for p in all_prims if p.GetTypeName() == 'Xform'])
    
    print(f"📈 统计信息:")
    print(f"   └─ 总Prims: {len(all_prims)}")
    print(f"   └─ 关节数: {joint_count}")
    print(f"   └─ 网格数: {mesh_count}")  
    print(f"   └─ 变换节点: {xform_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USD文件结构查看工具")
    parser.add_argument("usd_file", help="要分析的USD文件路径")
    parser.add_argument("--max-depth", type=int, default=3, help="最大显示深度")
    
    args = parser.parse_args()
    inspect_usd_file(args.usd_file)

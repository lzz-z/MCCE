import re
import numpy as np
import math
import os
from typing import Dict, List, Tuple, Optional
import json


class SacsVolumeCalculator:
    """SACS结构体积计算器 - 修复版本"""

    def __init__(self, project_path: str = None):
        if project_path is None:
            project_path = "/mnt/d/Python project/sacs_llm/demo06_project/Demo06"

        self.project_path = project_path
        self.inp_file_path = os.path.join(project_path, "sacinp.demo06")

        self.content = ""
        self.groups = {}
        self.joints = {}
        self.members = {}
        self._initialized = False

    def initialize(self) -> bool:
        """初始化计算器"""
        try:
            self._load_file()
            self._parse_groups()
            self._parse_joints()
            self._parse_members()

            print(f"DEBUG: 解析结果 - 组数:{len(self.groups)}, 节点数:{len(self.joints)}, 杆件数:{len(self.members)}")

            self._initialized = True
            return True
        except Exception as e:
            print(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_file(self):
        """读取SACS输入文件"""
        if not os.path.exists(self.inp_file_path):
            raise FileNotFoundError(f"文件不存在: {self.inp_file_path}")

        encodings = ['utf-8', 'gbk', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(self.inp_file_path, 'r', encoding=encoding) as f:
                    self.content = f.read()
                print(f"成功读取文件 (编码: {encoding})")
                return
            except UnicodeDecodeError:
                continue

        raise Exception("无法读取文件")

    def _parse_groups(self):
        """解析GRUP定义"""
        self.groups = {}

        # 使用更精确的正则表达式
        grup_pattern = r'^GRUP\s+(\w+)\s+(.+?)$'

        for line in self.content.splitlines():
            line = line.strip()
            if line.startswith('GRUP '):
                match = re.match(grup_pattern, line)
                if match:
                    group_name = match.group(1)
                    params = match.group(2).strip()

                    # 跳过CONE类型
                    if 'CONE' in params:
                        continue

                    group_info = self._parse_group_parameters(group_name, params)
                    if group_info:
                        self.groups[group_name] = group_info

    def _parse_group_parameters(self, group_name: str, param_str: str) -> Optional[Dict]:
        """解析组参数"""
        try:
            # 工字钢截面处理
            if any(steel_type in param_str for steel_type in ['W24X162', 'W24X131']):
                steel_section = self._extract_steel_section(param_str)
                if steel_section:
                    return {
                        'type': 'steel',
                        'section': steel_section,
                        'area': self._get_steel_area(steel_section)
                    }

            # 管状截面处理
            diameter, thickness = self._extract_pipe_dimensions(param_str)
            if diameter and thickness:
                area = math.pi * diameter * thickness  # 简化面积计算
                return {
                    'type': 'pipe',
                    'diameter': diameter,
                    'thickness': thickness,
                    'area': area
                }

            return None

        except Exception as e:
            print(f"解析组 {group_name} 参数失败: {e}")
            return None

    def _extract_pipe_dimensions(self, param_str: str) -> Tuple[Optional[float], Optional[float]]:
        """提取管状截面的直径和壁厚"""
        clean_str = param_str.strip()
        parts = clean_str.split()

        # 标准空格分隔格式
        if len(parts) >= 2:
            try:
                if self._is_number(parts[0]) and self._is_number(parts[1]):
                    diameter = float(parts[0])
                    thickness = float(parts[1])
                    if 0 < thickness < diameter and diameter > 0:
                        return diameter, thickness
            except ValueError:
                pass

        # 连续格式处理（如 43.9071.255）
        if len(parts) >= 1:
            first_param = parts[0]
            # 尝试识别连续的数字格式
            match = re.match(r'(\d+\.\d+)(\d+\.\d+)', first_param)
            if match:
                try:
                    diameter = float(match.group(1))
                    thickness = float(match.group(2))
                    if 0 < thickness < diameter:
                        return diameter, thickness
                except ValueError:
                    pass

        return None, None

    def _is_number(self, s: str) -> bool:
        """检查字符串是否为数字"""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _extract_steel_section(self, param_str: str) -> Optional[str]:
        """提取工字钢截面规格"""
        steel_match = re.search(r'(W\d+X\d+)', param_str)
        return steel_match.group(1) if steel_match else None

    def _get_steel_area(self, section: str) -> float:
        """获取工字钢截面面积"""
        steel_areas = {
            'W24X162': 47.7,
            'W24X131': 38.5
        }
        return steel_areas.get(section, 0.0)

    def _parse_joints(self):
        """解析节点坐标"""
        self.joints = {}

        lines = self.content.splitlines()
        in_joint_section = False

        for line in lines:
            line_stripped = line.strip()

            # 检测JOINT段开始
            if line_stripped == 'JOINT':
                in_joint_section = True
                continue

            # 检测段结束
            if in_joint_section and line_stripped and not line_stripped.startswith('JOINT'):
                if any(line_stripped.startswith(keyword) for keyword in ['MEMBER', 'GRUP', 'LOAD', 'PGRP']):
                    in_joint_section = False
                    continue

            if in_joint_section and line_stripped.startswith('JOINT'):
                self._parse_joint_line(line_stripped)

    def _parse_joint_line(self, line: str):
        """解析单个节点行"""
        try:
            # 移除JOINT关键字
            line_data = line.replace('JOINT', '').strip()
            parts = line_data.split()

            if len(parts) >= 4:
                joint_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                self.joints[joint_id] = {'x': x, 'y': y, 'z': z}

        except (ValueError, IndexError):
            pass

    def _parse_members(self):
        """解析杆件定义"""
        self.members = {}

        lines = self.content.splitlines()
        in_member_section = False

        for line in lines:
            line_stripped = line.strip()

            # 检测MEMBER段开始
            if line_stripped == 'MEMBER':
                in_member_section = True
                continue

            # 检测段结束
            if in_member_section and line_stripped and not line_stripped.startswith('MEMBER'):
                if any(line_stripped.startswith(keyword) for keyword in ['JOINT', 'GRUP', 'LOAD', 'PGRP', 'PLATE']):
                    in_member_section = False
                    continue

            if in_member_section and line_stripped.startswith('MEMBER'):
                self._parse_member_line(line_stripped)

    def _parse_member_line(self, line: str):
        """解析单个杆件行"""
        try:
            # 移除MEMBER关键字
            line_data = line.replace('MEMBER', '').strip()
            parts = line_data.split()

            if len(parts) >= 3:
                joint1 = int(parts[0])
                joint2 = int(parts[1])
                group = parts[2]

                member_name = f"{joint1}-{joint2}"

                # 计算杆件长度
                length = self._calculate_member_length(joint1, joint2)

                if length > 0:  # 只保存有效长度的杆件
                    self.members[member_name] = {
                        'joint1': joint1,
                        'joint2': joint2,
                        'group': group,
                        'length': length
                    }

        except (ValueError, IndexError):
            pass

    def _calculate_member_length(self, joint1: int, joint2: int) -> float:
        """计算杆件长度"""
        if joint1 not in self.joints or joint2 not in self.joints:
            return 0.0

        j1 = self.joints[joint1]
        j2 = self.joints[joint2]

        dx = j2['x'] - j1['x']
        dy = j2['y'] - j1['y']
        dz = j2['z'] - j1['z']

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def get_volume_summary(self) -> Dict:
        """获取体积计算摘要"""
        if not self._initialized:
            if not self.initialize():
                return {"error": "初始化失败", "status": "failed"}

        try:
            volume_data = self._calculate_volumes()
            return {
                "total_volume_m3": volume_data["total_volume"],
                "total_members": volume_data["total_members"],
                "valid_members": volume_data["valid_members"],
                "group_volumes": volume_data["group_volumes"],
                "status": "success"
            }
        except Exception as e:
            return {"error": f"体积计算失败: {e}", "status": "failed"}

    def _calculate_volumes(self) -> Dict:
        """计算所有杆件的体积"""
        total_volume = 0.0
        group_volumes = {}
        member_details = {}
        valid_members = 0

        for member_name, member_info in self.members.items():
            group_name = member_info['group']
            length = member_info['length']

            if group_name in self.groups and length > 0:
                group_info = self.groups[group_name]
                area_in2 = group_info['area']

                # 转换为立方米
                length_m = length * 0.0254  # 英寸转米
                area_m2 = area_in2 * 0.00064516  # 平方英寸转平方米
                volume_m3 = length_m * area_m2

                total_volume += volume_m3
                valid_members += 1

                # 按组统计
                if group_name not in group_volumes:
                    group_volumes[group_name] = {
                        'volume': 0.0,
                        'members': 0,
                        'avg_length': 0.0
                    }

                group_volumes[group_name]['volume'] += volume_m3
                group_volumes[group_name]['members'] += 1

                # 记录杆件详情
                member_details[member_name] = {
                    'group': group_name,
                    'length_m': length_m,
                    'area_m2': area_m2,
                    'volume_m3': volume_m3
                }

        # 计算平均长度
        for group_name, group_data in group_volumes.items():
            if group_data['members'] > 0:
                total_length = sum(
                    member_details[member]['length_m']
                    for member, details in member_details.items()
                    if details['group'] == group_name
                )
                group_data['avg_length'] = total_length / group_data['members']

        return {
            'total_volume': total_volume,
            'total_members': len(self.members),
            'valid_members': valid_members,
            'group_volumes': group_volumes,
            'member_details': member_details
        }


# 简化的接口函数
def calculate_sacs_volume(project_path: str = None) -> Dict:
    """
    简化的SACS体积计算接口

    Args:
        project_path: SACS项目路径

    Returns:
        包含体积信息的字典
    """
    try:
        calculator = SacsVolumeCalculator(project_path)
        result = calculator.get_volume_summary()

        # 添加调试信息
        print(f"DEBUG: 体积计算结果 = {result}")

        return result
    except Exception as e:
        print(f"ERROR: 体积计算异常 = {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"体积计算失败: {e}", "status": "failed"}


def get_detailed_volume_analysis(project_path: str = None) -> Dict:
    """
    详细的SACS体积分析接口

    Args:
        project_path: SACS项目路径

    Returns:
        包含详细分析的字典
    """
    calculator = SacsVolumeCalculator(project_path)
    return calculator.get_detailed_analysis()

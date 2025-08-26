import os
import sqlite3
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging


class UCValueExtractor:
    """UC值提取器 - 模块化版本"""

    def __init__(self, work_dir: str = None):
        """
        初始化UC值提取器

        Args:
            work_dir: 工作目录路径，如果为None则使用默认路径
        """
        if work_dir is None:
            work_dir = "/mnt/d/Python project/sacs_llm/demo06_project/Demo06"

        self.work_dir = Path(work_dir)
        self.db_path = self.work_dir / "sacsdb.db"
        self.logger = self._setup_logger()

        # 检查数据库文件
        self._check_database()

    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger('UCValueExtractor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - UCExtractor - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _check_database(self):
        """检查数据库文件是否存在"""
        if self.db_path.exists():
            self.logger.info(f"找到数据库文件: {self.db_path}")
        else:
            self.logger.warning(f"数据库文件不存在: {self.db_path}")

    def get_uc_summary(self) -> Dict[str, Any]:
        """
        获取UC值摘要 - 主要接口方法

        Returns:
            包含UC值统计信息的字典
        """
        try:
            uc_data = self.extract_uc_values()

            if uc_data['summary']['total_members'] == 0:
                return {"error": "未找到有效的UC数据", "status": "failed"}

            return {
                "max_uc": uc_data['summary']['max_uc'],
                "mean_uc": uc_data['summary']['mean_uc'],
                "total_members": uc_data['summary']['total_members'],
                "critical_members": len(uc_data['summary']['critical_members']),
                "high_risk_members": len(uc_data['summary']['high_risk_members']),
                "feasible_design": uc_data['summary']['max_uc'] <= 1.0,
                "status": "success"
            }

        except Exception as e:
            self.logger.error(f"获取UC摘要失败: {e}")
            return {"error": f"获取UC摘要失败: {e}", "status": "failed"}

    def get_detailed_uc_analysis(self) -> Dict[str, Any]:
        """获取详细UC分析"""
        try:
            uc_data = self.extract_uc_values()

            return {
                "summary": self.get_uc_summary(),
                "member_uc": uc_data['member_uc'],
                "statistics": uc_data['summary']['uc_statistics'],
                "critical_members": uc_data['summary']['critical_members'],
                "high_risk_members": uc_data['summary']['high_risk_members'],
                "uc_distribution": uc_data['summary']['uc_distribution'],
                "status": "success"
            }

        except Exception as e:
            self.logger.error(f"获取详细UC分析失败: {e}")
            return {"error": f"获取详细UC分析失败: {e}", "status": "failed"}

    def extract_uc_values(self) -> Dict[str, Any]:
        """提取所有杆件的UC值"""
        self.logger.info("开始提取UC值...")

        uc_data = {
            'member_uc': {},
            'summary': {
                'total_members': 0,
                'max_uc': 0.0,
                'min_uc': float('inf'),
                'mean_uc': 0.0,
                'std_uc': 0.0,
                'median_uc': 0.0,
                'critical_members': [],
                'high_risk_members': [],
                'uc_distribution': [],
                'uc_statistics': {}
            }
        }

        if not self.db_path.exists():
            self.logger.error(f"数据库文件不存在: {self.db_path}")
            return uc_data

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 查询UC值数据
            query = """
            SELECT MemberName, MaxUC, AxialUC, YYBendingUC, ZZBendingUC, 
                   TotalShearUC, VonMisesUC, LocalBucklingUC
            FROM R_POSTMEMBERRESULTS
            WHERE MaxUC IS NOT NULL AND MaxUC > 0
            """

            self.logger.info("执行查询...")
            cursor.execute(query)
            results = cursor.fetchall()

            if not results:
                self.logger.warning("查询结果为空")
                conn.close()
                return uc_data

            self.logger.info(f"查询到 {len(results)} 条记录")

            # 处理结果
            uc_values = []
            processed_members = set()

            for row in results:
                member_name = row[0]
                max_uc = float(row[1]) if row[1] is not None else 0.0

                # 避免重复处理
                if member_name in processed_members:
                    continue
                processed_members.add(member_name)

                # 存储杆件UC值
                uc_data['member_uc'][member_name] = {
                    'max_uc': max_uc,
                    'axial_uc': float(row[2]) if row[2] is not None else 0.0,
                    'yy_bending_uc': float(row[3]) if row[3] is not None else 0.0,
                    'zz_bending_uc': float(row[4]) if row[4] is not None else 0.0,
                    'total_shear_uc': float(row[5]) if row[5] is not None else 0.0,
                    'von_mises_uc': float(row[6]) if row[6] is not None else 0.0,
                    'local_buckling_uc': float(row[7]) if row[7] is not None else 0.0
                }

                uc_values.append(max_uc)

                # 识别关键杆件
                if max_uc > 1.0:
                    uc_data['summary']['critical_members'].append({
                        'member': member_name,
                        'uc': max_uc
                    })
                elif max_uc > 0.8:
                    uc_data['summary']['high_risk_members'].append({
                        'member': member_name,
                        'uc': max_uc
                    })

            conn.close()

            # 计算统计信息
            if uc_values:
                uc_array = np.array(uc_values)

                uc_data['summary'].update({
                    'total_members': len(uc_values),
                    'max_uc': float(np.max(uc_array)),
                    'min_uc': float(np.min(uc_array)),
                    'mean_uc': float(np.mean(uc_array)),
                    'std_uc': float(np.std(uc_array)),
                    'median_uc': float(np.median(uc_array))
                })

                # UC分布统计
                uc_data['summary']['uc_distribution'] = self._calculate_uc_distribution(uc_values)

                # 详细统计
                uc_data['summary']['uc_statistics'] = {
                    'percentile_95': float(np.percentile(uc_array, 95)),
                    'percentile_90': float(np.percentile(uc_array, 90)),
                    'percentile_75': float(np.percentile(uc_array, 75)),
                    'percentile_25': float(np.percentile(uc_array, 25)),
                    'over_1_0_count': len([uc for uc in uc_values if uc > 1.0]),
                    'over_0_8_count': len([uc for uc in uc_values if uc > 0.8]),
                    'over_0_5_count': len([uc for uc in uc_values if uc > 0.5])
                }

            self.logger.info(f"UC值提取完成，处理了 {len(uc_values)} 个杆件")
            return uc_data

        except Exception as e:
            self.logger.error(f"提取UC值失败: {e}")
            return uc_data

    def _calculate_uc_distribution(self, uc_values: List[float]) -> List[Dict]:
        """计算UC值分布"""
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, float('inf')]
        labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '>1.0']

        distribution = []
        for i in range(len(bins) - 1):
            count = len([uc for uc in uc_values if bins[i] <= uc < bins[i + 1]])
            percentage = (count / len(uc_values)) * 100 if uc_values else 0

            distribution.append({
                'range': labels[i],
                'count': count,
                'percentage': percentage
            })

        return distribution

    def get_critical_members(self, threshold: float = 1.0) -> List[Dict]:
        """获取超过阈值的关键杆件"""
        try:
            uc_data = self.extract_uc_values()
            critical_members = []

            for member_name, uc_info in uc_data['member_uc'].items():
                if uc_info['max_uc'] > threshold:
                    critical_members.append({
                        'member': member_name,
                        'max_uc': uc_info['max_uc'],
                        'axial_uc': uc_info['axial_uc'],
                        'bending_uc': max(uc_info['yy_bending_uc'], uc_info['zz_bending_uc'])
                    })

            # 按UC值降序排列
            critical_members.sort(key=lambda x: x['max_uc'], reverse=True)

            return critical_members

        except Exception as e:
            self.logger.error(f"获取关键杆件失败: {e}")
            return []

    def export_uc_results(self, output_path: str = None) -> str:
        """导出UC分析结果"""
        if output_path is None:
            output_path = self.work_dir / "uc_analysis_results.json"

        try:
            results = self.get_detailed_uc_analysis()

            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"UC分析结果已导出到: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"导出UC结果失败: {e}")
            raise


# 简化的接口函数
def get_sacs_uc_summary(work_dir: str = None) -> Dict:
    """
    简化的SACS UC值获取接口

    Args:
        work_dir: SACS工作目录

    Returns:
        包含UC值摘要的字典
    """
    extractor = UCValueExtractor(work_dir)
    return extractor.get_uc_summary()


def get_detailed_uc_analysis(work_dir: str = None) -> Dict:
    """
    详细的SACS UC分析接口

    Args:
        work_dir: SACS工作目录

    Returns:
        包含详细UC分析的字典
    """
    extractor = UCValueExtractor(work_dir)
    return extractor.get_detailed_uc_analysis()


def get_critical_members(work_dir: str = None, threshold: float = 1.0) -> List[Dict]:
    """
    获取关键杆件接口

    Args:
        work_dir: SACS工作目录
        threshold: UC值阈值

    Returns:
        关键杆件列表
    """
    extractor = UCValueExtractor(work_dir)
    return extractor.get_critical_members(threshold)

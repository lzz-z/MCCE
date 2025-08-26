#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II
"""

import numpy as np
import json
from datetime import datetime

# Pymoo 库
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class SacsOptimizationProblem(Problem):

    def __init__(self, optimizer, var_names, xl, xu, max_uc=1.0):

        self.optimizer = optimizer
        self.var_names = var_names
        self.max_uc_constraint = max_uc

        # pymoo问题定义
        # n_var: 变量数量
        # n_obj: 目标数量 (1. 体积, 2. -疲劳寿命)
        # n_constr: 约束数量 (1. max_uc)
        super().__init__(n_var=len(xl), n_obj=2, n_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):

        # 初始化目标函数值和约束违反值的数组
        f = np.full((x.shape[0], self.n_obj), np.inf)  # 惩罚值
        g = np.full((x.shape[0], self.n_constr), np.inf)  # 惩罚值

        for i in range(x.shape[0]):
            self.optimizer.logger.info(f"--- 评估个体 {i + 1}/{x.shape[0]} 在当前代 ---")

            design_vector = x[i]
            modifications = {name: value for name, value in zip(self.var_names, design_vector)}

            try:
                # 1. 应用修改
                if not self.optimizer.apply_design_modifications(modifications):
                    self.optimizer.logger.error(f"个体 {i + 1}：设计修改应用失败。跳过。")
                    continue

                # 2. 运行SACS分析
                if not self.optimizer.run_sacs_analysis():
                    self.optimizer.logger.error(f"个体 {i + 1}：SACS分析失败。跳过。")
                    continue

                # 3. 获取性能指标
                performance = self.optimizer.get_current_performance()
                metrics = self.optimizer._extract_key_metrics(performance)

                # 4. 提取目标和约束值
                volume = metrics.get('total_volume_m3')
                fatigue_life = metrics.get('min_fatigue_life')
                max_uc = metrics.get('max_unity_check')

                if volume is None or fatigue_life is None or max_uc is None:
                    self.optimizer.logger.warning(f"个体 {i + 1}：性能指标不完整。跳过。")
                    continue

                # 5. 填充目标函数和约束值
                f[i, 0] = volume
                f[i, 1] = -fatigue_life  # 最小化 -fatigue_life 等于 最大化 fatigue_life
                g[i, 0] = max_uc - self.max_uc_constraint  # 约束 g(x) <= 0

                self.optimizer.logger.info(
                    f"个体 {i + 1} 评估完成: 体积={volume:.2f}, 寿命={fatigue_life:.1f}, UC={max_uc:.3f}")

            except Exception as e:
                self.optimizer.logger.critical(f"个体 {i + 1} 评估过程中出现严重错误: {e}")

        out["F"] = f
        out["G"] = g


class Nsga2Runner:


    def __init__(self, optimizer):

        self.optimizer = optimizer
        self.logger = optimizer.logger

    def run(self, pop_size: int = 50, n_gen: int = 25, max_uc: float = 1.0):

        self.logger.info("=" * 20 + " 开始NSGA-II多目标优化 " + "=" * 20)
        self.logger.info(f"参数: 种群大小={pop_size}, 迭代代数={n_gen}, UC约束<={max_uc}")

        # 1. 准备问题定义所需参数
        var_names = list(self.optimizer.design_variables.keys())
        xl = np.array([self.optimizer.design_variables[name]['min'] for name in var_names])
        xu = np.array([self.optimizer.design_variables[name]['max'] for name in var_names])

        # 2. 定义问题实例
        problem = SacsOptimizationProblem(self.optimizer, var_names, xl, xu, max_uc)

        # 3. 生成初始种群 (尝试LLM, 失败则回退)
        initial_population = self.optimizer.generate_llm_initial_population(pop_size)
        if initial_population is None:
            self.logger.warning("LLM生成初始种群失败，将使用拉丁超立方采样（LHS）生成随机初始种群。")
            sampling = LHS()
        else:
            sampling = initial_population

        # 4. 定义NSGA-II算法
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        # 5. 定义终止条件
        termination = get_termination("n_gen", n_gen)

        # 6. 执行优化
        self.logger.info("开始遗传算法进化过程")
        start_time = datetime.now()

        res = minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            save_history=True,
            verbose=True
        )

        end_time = datetime.now()
        self.logger.info(f"进化过程结束，耗时: {end_time - start_time}")

        # 7. 处理和返回结果
        return self._process_results(res, var_names, start_time, end_time, pop_size, n_gen)

    def _process_results(self, res, var_names, start_time, end_time, pop_size, n_gen):
        """处理pymoo的返回结果"""
        final_results = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'pop_size': pop_size,
            'n_gen': n_gen,
            'pareto_front': []
        }

        if res.F is not None and len(res.F) > 0:
            self.logger.info(f"优化完成！找到 {len(res.F)} 个帕累托最优解。")
            for i in range(len(res.F)):
                solution = {
                    "objectives": {
                        "total_volume_m3": res.F[i, 0],
                        "min_fatigue_life": -res.F[i, 1]
                    },
                    "variables": {name: value for name, value in zip(var_names, res.X[i])}
                }
                final_results['pareto_front'].append(solution)
        else:
            self.logger.warning("优化未找到任何有效解。")

        return final_results


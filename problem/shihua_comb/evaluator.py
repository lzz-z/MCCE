#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import pandas as pd
import concurrent.futures
import multiprocessing as mp
from typing import Optional, Callable
from scipy.optimize import minimize
import numpy as np

def _opt_worker(q, optimizer_func, kwargs):
    try:
        res = optimizer_func(**kwargs)
        q.put(("ok", res))
    except Exception as e:
        q.put(("err", repr(e)))

def safe_optimize_mp(optimizer_func: Callable, timeout: int = 10, **kwargs):
    """
    在独立进程中运行优化器；超时则 terminate 并返回 None。
    用法：
        snapped = safe_optimize_mp(optimize_tail_multiobjective_with_cost, timeout=10, **optimizer_kwargs)
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_opt_worker, args=(q, optimizer_func, kwargs))
    p.daemon = True
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    if not q.empty():
        status, payload = q.get()
        if status == "ok":
            return payload
        else:
            raise RuntimeError(f"optimizer failed in worker: {payload}")
    return None

def _eval_worker(evaluator, solution_df, q):
    """在子进程中运行评估，把结果放进队列。"""
    try:
        res = evaluator.evaluate(solution_df)
        q.put(("ok", res))
    except Exception as e:
        q.put(("err", repr(e)))

def safe_evaluate_mp(evaluator, solution_df, timeout=5):
    """
    在独立进程中评估；超时则terminate进程并返回None。
    使用 'spawn' 上下文以避免fork造成的死锁/资源继承问题。
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_eval_worker, args=(evaluator, solution_df, q))
    p.daemon = True  # 防止僵尸进程
    p.start()
    p.join(timeout)
    if p.is_alive():
        # 超时：强制结束
        p.terminate()
        p.join()
        return None
    # 未超时：读回结果
    if not q.empty():
        status, payload = q.get()
        if status == "ok":
            return payload
        else:
            # 子进程异常
            raise RuntimeError(f"evaluate failed in worker: {payload}")
    return None



def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def split_interval_by_price(start: datetime, end: datetime, price_df: pd.DataFrame) -> List[Tuple[datetime, datetime, float]]:
    out = []
    cur = start
    base_midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur < end:
        hour = cur.hour + cur.minute / 60 + cur.second / 3600
        band = price_df[(price_df.start_hour <= hour) & (hour < price_df.end_hour)].iloc[0]
        band_end = base_midnight + timedelta(hours=float(band.end_hour))
        seg_end = min(end, band_end)
        out.append((cur, seg_end, float(band.price_yuan_per_kwh)))
        cur = seg_end
    return out


def count_switches_per_shift(schedule: pd.DataFrame, shifts: List[Dict[str, Any]], task_start: datetime) -> Dict[str, int]:
    switches = {}
    base_midnight = task_start.replace(hour=0, minute=0, second=0, microsecond=0)
    for sh in shifts:
        s = base_midnight + timedelta(hours=int(sh["start_hour"]))
        e = base_midnight + timedelta(hours=int(sh["end_hour"]))
        segs = schedule[(schedule["start_time"] < e) & (schedule["end_time"] > s)].copy()
        segs = segs.sort_values("start_time")
        cnt = 0
        prev_combo = None
        # combo active exactly at shift start
        active_at_s = schedule[(schedule["start_time"] <= s) & (schedule["end_time"] > s)]
        if not active_at_s.empty:
            prev_combo = active_at_s.iloc[0]["combination_id"]
        for _, row in segs.iterrows():
            st = row["start_time"]
            combo = row["combination_id"]
            if st >= s:
                if prev_combo is None or combo != prev_combo:
                    cnt += 1
                prev_combo = combo
            else:
                prev_combo = row["combination_id"]
        switches[sh["shift_name"]] = cnt
    return switches


def _parse_iso_mixed(series: pd.Series) -> pd.Series:
    """
    更鲁棒的 ISO 解析：优先以 ISO8601 方式解析，失败的再 fallback 到 mixed。
    """
    s = pd.to_datetime(series, format='ISO8601', errors='coerce')
    if s.isna().any():
        s2 = pd.to_datetime(series, errors='coerce', format='mixed')  # pandas >=2.0 支持
        s = s.fillna(s2)
    # 仍有 NaT 的话，保留 NaT 以便后续抛错或过滤
    return s

def normalize_solution_times_to_us(solution_df: pd.DataFrame) -> pd.DataFrame:
    """
    解析 start_time/end_time 为 Timestamp（微秒精度），去掉时区，统一到同一天内（不做跨天检查）。
    """
    sol = solution_df.copy()

    # 1) 解析为 pandas Timestamp（允许不同小数位数）
    if sol["start_time"].dtype == object:
        sol["start_time"] = _parse_iso_mixed(sol["start_time"])
    else:
        sol["start_time"] = pd.to_datetime(sol["start_time"], errors='coerce')

    if sol["end_time"].dtype == object:
        sol["end_time"]   = _parse_iso_mixed(sol["end_time"])
    else:
        sol["end_time"]   = pd.to_datetime(sol["end_time"], errors='coerce')

    # 若仍有 NaT，说明输入非法
    if sol["start_time"].isna().any() or sol["end_time"].isna().any():
        raise ValueError("Failed to parse start_time/end_time into timestamps.")

    # 2) 去时区（若有）
    if getattr(sol["start_time"].dt, 'tz', None) is not None:
        sol["start_time"] = sol["start_time"].dt.tz_localize(None)
    if getattr(sol["end_time"].dt, 'tz', None) is not None:
        sol["end_time"] = sol["end_time"].dt.tz_localize(None)

    # 3) 统一到微秒精度，避免 to_pydatetime 的纳秒告警
    sol["start_time"] = sol["start_time"].dt.round('us')
    sol["end_time"]   = sol["end_time"].dt.round('us')

    return sol

def ts_to_iso_us(ts: pd.Timestamp) -> str:
    """
    把 Timestamp 格式化到 ISO 串，最多微秒；去掉多余的尾随 0 和点。
    """
    s = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")
    s = s.rstrip("0").rstrip(".")
    return s


def compute_total_volume(solution_df: pd.DataFrame, combos_map: Dict[str, Dict[str, float]]) -> float:
    st = pd.to_datetime(solution_df["start_time"])
    et = pd.to_datetime(solution_df["end_time"])
    hours = (et - st).dt.total_seconds() / 3600.0
    flows = solution_df["combination_id"].map(lambda cid: combos_map[cid]["flow"]).astype(float)
    return float(np.dot(hours.to_numpy(), flows.to_numpy()))

def _total_cost_for_tail(
    starts_tail_ts: list,  # list of pd.Timestamp
    d: np.ndarray,         # hours
    cids_tail: list,
    price_df: pd.DataFrame,
    combos_map: Dict[str, Dict[str, float]],
    split_func: Callable[[datetime, datetime, pd.DataFrame], list],
) -> float:
    total_cost = 0.0
    for j in range(len(d)):
        st_ts: pd.Timestamp = starts_tail_ts[j].round('us')
        et_ts: pd.Timestamp = st_ts + pd.to_timedelta(float(d[j]), unit="h").round('us')
        cid = cids_tail[j]
        power = float(combos_map[cid]["power"])

        # ensure real python datetimes for split_func
        st_py = st_ts.to_pydatetime()
        et_py = et_ts.to_pydatetime()

        for seg_start, seg_end, price in split_func(st_py, et_py, price_df):
            hours = (seg_end - seg_start).total_seconds() / 3600.0
            total_cost += power * hours * price
    return total_cost

def optimize_tail_multiobjective_with_cost(
    solution_df: pd.DataFrame,
    combos_map: Dict[str, Dict[str, float]],
    price_df: pd.DataFrame,
    task_start_iso: str,
    required_volume: float,
    k_tail: int = 3,
    lambda_change: float = 1e-3,
    eps_each: float = 1e-6,
    split_func: Callable[[datetime, datetime, pd.DataFrame], list] = None,
) -> Optional[pd.DataFrame]:
    """
    Minimize: total electricity cost (with price_df) + lambda * ||d - d0||^2
    Variables: durations of the last k_tail segments (hours)
    Constraints:
      - exact volume match
      - sum(d) <= day_end - first_tail_start  (no cross-day)
      - d_i >= eps_each
    Keeps order/continuity. Returns optimized schedule or None if infeasible.
    """
    if split_func is None:
        # must be provided or defined in the same module
        raise NameError("split_interval_by_price (split_func) is not provided/visible")

    if solution_df.empty:
        return None

    sol = solution_df.copy()
    
    sol["start_time"] = pd.to_datetime(sol["start_time"])
    sol["end_time"]   = pd.to_datetime(sol["end_time"])
    sol = normalize_solution_times_to_us(sol)
    n = len(sol)
    k = min(k_tail, n)
    idx0 = n - k

    base_midnight = pd.to_datetime(task_start_iso).normalize()
    day_end = base_midnight + pd.Timedelta(days=1)

    # fixed (head) volume
    if idx0 > 0:
        st_fixed = sol.loc[:idx0-1, "start_time"]
        et_fixed = sol.loc[:idx0-1, "end_time"]
        hours_fixed = (et_fixed - st_fixed).dt.total_seconds() / 3600.0
        flows_fixed = sol.loc[:idx0-1, "combination_id"].map(lambda cid: combos_map[cid]["flow"]).astype(float)
        vol_fixed = float(np.dot(hours_fixed.to_numpy(), flows_fixed.to_numpy()))
    else:
        vol_fixed = 0.0

    # tail pieces (keep as pandas Timestamps!)
    starts_tail = list(sol.loc[idx0:, "start_time"].tolist())  # list[pd.Timestamp]
    ends_tail   = list(sol.loc[idx0:, "end_time"].tolist())    # list[pd.Timestamp]
    d0          = np.array([(et - st).total_seconds() / 3600.0 for st, et in zip(starts_tail, ends_tail)], dtype=float)
    cids_tail   = sol.loc[idx0:, "combination_id"].tolist()
    flows_tail  = sol.loc[idx0:, "combination_id"].map(lambda cid: combos_map[cid]["flow"]).astype(float).to_numpy()

    target_tail_volume = required_volume - vol_fixed
    Tmax = float((day_end - starts_tail[0]).total_seconds()) / 3600.0
    bounds = [(max(eps_each, 0.0), Tmax) for _ in range(k)]

    # make d0 feasible wrt equality if possible by adjusting the last one
    vol_tail0 = float(np.dot(flows_tail, d0))
    gap = target_tail_volume - vol_tail0
    if abs(gap) > 1e-9 and flows_tail[-1] > 0:
        d0[-1] += gap / float(flows_tail[-1])
        # clip to bounds; will rely on SLSQP to fix slight infeasibility if clipping occurred
        low, high = bounds[-1]
        d0[-1] = np.clip(d0[-1], low, high)

    def total_cost_of_d(d):
        # rebuild tail starts for continuity
        starts_rebuilt = [starts_tail[0]]
        cur = starts_tail[0]
        for j in range(1, k):
            cur = cur + pd.to_timedelta(float(d[j-1]), unit="h")
            starts_rebuilt.append(cur)
        return _total_cost_for_tail(starts_rebuilt, d, cids_tail, price_df, combos_map, split_func)

    def obj(d):
        return total_cost_of_d(d) #+ lambda_change * float(np.sum((d - d0)**2))

    cons = [
        {"type": "eq",   "fun": lambda d, f=flows_tail, tv=target_tail_volume: float(np.dot(f, d)) - tv},
        {"type": "ineq", "fun": lambda d, tmax=Tmax: tmax - float(np.sum(d))},
    ]

    res = minimize(
        obj, d0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 3000, "ftol": 1e-6, "disp": False}
    )
    if not res.success:
        return None

    d_opt = res.x

    # assemble optimized tail with continuity
    sol_opt = sol.copy()
    cur = starts_tail[0]
    for j in range(k):
        new_end = cur + pd.to_timedelta(float(d_opt[j]), unit="h")
        sol_opt.loc[idx0 + j, "start_time"] = cur
        sol_opt.loc[idx0 + j, "end_time"]   = new_end
        cur = new_end

    # sanity checks
    if sol_opt.loc[n-1, "end_time"] > day_end + pd.Timedelta(seconds=1e-6):
        return None
    if (sol_opt["end_time"] <= sol_opt["start_time"]).any():
        return None
    # exact volume check
    final_vol = compute_total_volume(sol_opt, combos_map)
    if abs(final_vol - required_volume) > 1e-4:
        return None

    # stringify ISO
    for col in ("start_time", "end_time"):
        s = sol_opt[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str.rstrip("0").str.rstrip(".")
        sol_opt[col] = s
    return sol_opt

def solution_to_csv_us(sol: pd.DataFrame) -> str:
    out = sol.copy()
    out["start_time"] = out["start_time"].apply(lambda x: ts_to_iso_us(pd.to_datetime(x)))
    out["end_time"]   = out["end_time"].apply(lambda x: ts_to_iso_us(pd.to_datetime(x)))
    return out.to_csv(index=False)

class PumpEvaluator:
    def __init__(self, task_path: str, combos_path: str, price_path: str):
        # load inputs once
        with open(task_path, "r", encoding="utf-8") as f:
            self.task = json.load(f)
        self.combos = pd.read_csv(combos_path)
        self.price = pd.read_csv(price_path)

        # make combo lookup
        self.combos_map = {
            row["combination_id"]: {
                "flow": float(row["flow_rate_m3h"]),
                "power": float(row["total_power_kw"]),
            }
            for _, row in self.combos.iterrows()
        }
        self.task_start = parse_iso(self.task["time_window_start"])

    def evaluate(self, solution_df: pd.DataFrame, tol_volume: float = 1e-6) -> Dict[str, Any]:
        sol = solution_df.copy()
        

        if set(sol.columns) != set(["start_time", "end_time", "combination_id"]):
            raise ValueError("solution DataFrame headers must be: start_time,end_time,combination_id")

        sol["start_time"] = sol["start_time"]
        sol["end_time"] = sol["end_time"]
        sol = normalize_solution_times_to_us(sol)
        sol = sol.sort_values("start_time").reset_index(drop=True)

        messages = []
        valid = True
        if sol.iloc[0]["start_time"] != self.task_start:
            valid = False
            messages.append(f"首行 start_time 必须等于 task.time_window_start = {self.task_start.isoformat()}")

        # continuity and ids
        for i, row in sol.iterrows():
            if row["end_time"] <= row["start_time"]:
                valid = False
                messages.append(f"第{i}行 end_time 必须晚于 start_time")
            if row["combination_id"] not in self.combos_map:
                valid = False
                messages.append(f"第{i}行 combination_id={row['combination_id']} 不在 pump_combinations.csv 中")
            if i > 0:
                prev_end = sol.iloc[i - 1]["end_time"]
                if row["start_time"] != prev_end:
                    valid = False
                    messages.append(f"第{i}行 start_time 与上一行 end_time 不连续 ({row['start_time']} != {prev_end})")

        # compute totals and check flow range
        required_volume = float(self.task["required_volume_m3"])
        flow_min = float(self.task["flow_rate_min_m3h"])
        flow_max = float(self.task["flow_rate_max_m3h"])

        total_volume = 0.0
        total_cost = 0.0

        for i, row in sol.iterrows():
            combo = self.combos_map[row["combination_id"]]
            flow = combo["flow"]
            power = combo["power"]
            if not (flow_min <= flow <= flow_max):
                valid = False
                messages.append(f"第{i}行 组合 {row['combination_id']} 的流量 {flow} 不在允许范围 [{flow_min}, {flow_max}]")

            for (seg_start, seg_end, price_yuan_per_kwh) in split_interval_by_price(row["start_time"], row["end_time"], self.price):
                hours = (seg_end - seg_start).total_seconds() / 3600.0
                total_volume += flow * hours
                total_cost += power * hours * price_yuan_per_kwh

        if abs(total_volume - required_volume) > tol_volume:
            valid = False
            messages.append(f"总输送量 {total_volume:.6f} 与任务要求 {required_volume:.6f} 不相等（允许误差 {tol_volume}）")

        switches = count_switches_per_shift(sol, self.task["shift_definitions"], self.task_start)
        for sh in self.task["shift_definitions"]:
            name = sh["shift_name"]
            if switches[name] > int(self.task["max_switches_per_shift"]):
                valid = False
                messages.append(f"班次[{name}] 切换次数 {switches[name]} 超过上限 {self.task['max_switches_per_shift']}")

        return {
            "valid": valid,
            "messages": messages,
            "total_volume_m3": total_volume,
            "total_cost_yuan": total_cost,
            "switches_per_shift": switches,
        }
        
import re
from io import StringIO

def extract_csv_from_candidate(raw: str) -> str:
    """提取 <candidate>...</candidate> 之间的 CSV; 若没有标签就原样返回。"""
    m = re.search(r"<candidate>(.*?)</candidate>", raw, flags=re.S)
    s = m.group(1) if m else raw
    # 去掉可能的代码块围栏和BOM
    s = s.strip()
    s = s.replace("\ufeff", "")         # BOM
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s, flags=re.M)
    return s.strip()

def read_solution_csv_robust(text: str) -> pd.DataFrame:
    """
    更鲁棒地读取 solution：
    - 保留 combination_id 为字符串
    - 去首尾空格、空行
    - 只保留需要的三列
    """
    payload = extract_csv_from_candidate(text)
    df = pd.read_csv(
        StringIO(payload),
        dtype={"combination_id": str},       # 防止 '1#&3#' 变 NaN/丢字符
        skip_blank_lines=True
    )
    # 规范列名
    df.columns = [c.strip() for c in df.columns]
    # 容忍用户给了多列；我们只取这三列
    needed = ["start_time", "end_time", "combination_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[needed]
    # 去掉全空行
    df = df.dropna(how="all")
    # 去空格
    for c in ["start_time", "end_time", "combination_id"]:
        df[c] = df[c].astype(str).str.strip()
    # 丢掉空的行
    df = df[(df["start_time"] != "") & (df["end_time"] != "") & (df["combination_id"] != "")]
    if df.empty:
        raise ValueError("Empty schedule after cleaning.")
    return df

from io import StringIO
class RewardingSystem:
    def __init__(self, config=None):
        self.config = config
        self.evaluator = PumpEvaluator(
            task_path="/root/nian/MOLLM/problem/shihua_comb/task001.json",
            combos_path="/root/nian/MOLLM/problem/shihua_comb/pump_combinations.csv",
            price_path="/root/nian/MOLLM/problem/shihua_comb/electricity_price.csv",
        )

    def evaluate(self, items):
        valid_items = []
        log_dict = {}
        i = 0

        required = float(self.evaluator.task["required_volume_m3"])
        tol = 1e-6  # 与 evaluator 一致

        for item in items:
            i += 1
            try:
                #print('item value:',item.value)
                # solution_raw = pd.read_csv(StringIO(item.value))
                solution_raw = read_solution_csv_robust(item.value)
                solution_raw = normalize_solution_times_to_us(solution_raw)
                #print('solution',solution_raw)
                # 10s 硬超时的优化器（失败/超时 -> None）
                snapped = safe_optimize_mp(
                    optimize_tail_multiobjective_with_cost,
                    timeout=10,
                    solution_df=solution_raw,
                    combos_map=self.evaluator.combos_map,
                    price_df=self.evaluator.price,
                    task_start_iso=self.evaluator.task["time_window_start"],
                    required_volume=required,
                    k_tail=len(solution_raw),          # 用当前解的段数
                    lambda_change=1e-3,
                    eps_each=1e-6,
                    split_func=split_interval_by_price,
                )

                # 优化失败或超时 -> 回退到原解
                to_eval = snapped if snapped is not None else solution_raw

                # 评估（5s 硬超时）
                result = safe_evaluate_mp(self.evaluator, to_eval, timeout=5)
                if result is None:
                    print(f'item {i}: evaluate 超时，跳过这个 item')
                    continue

                # 只接受：valid 且 体积==25000（在容差内）
                if (abs(result["total_volume_m3"] - required) > tol):
                    print(f'item {i}: 无效方案或体积未达标 ({result["total_volume_m3"]:.6f} != {required:.6f})，跳过, result:{result}')
                    continue

                # 通过 gate 才写回、计分
                item.value = solution_to_csv_us(to_eval)
                results_dict = {
                    "original_results":   {"cost": result["total_cost_yuan"]},
                    "transformed_results":{"cost": result["total_cost_yuan"]},
                    'constraint_results':{"constraints":''.join(result["messages"])},
                    "overall_score":      -result["total_cost_yuan"],
                }
                item.assign_results(results_dict)
                valid_items.append(item)

            except Exception as e:
                print(f'item {i}: execution error, error {e}')
                

        log_dict["invalid_num"]  = len(items) - len(valid_items)
        log_dict["repeated_num"] = 0
        return valid_items, log_dict

def generate_initial_population(config, seed=42):
    
    return [] # use llm to randomly initialize init pops
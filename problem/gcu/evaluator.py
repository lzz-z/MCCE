import numpy as np
import random
from pathlib import Path
import pickle
import requests
from algorithm.base import Item
import time

BASE_URL = "http://119.4.205.3:8008/api"
COOKIE = {"ocp_token": "66b5d454-f3f2-4836-814e-81bc48b2762b"}  # 你的登录 cookie

def auto_submit(problem_id: int, code: str, wait_interval: int = 10, timeout: int = 360, max_retries: int = 3):
    """
    自动提交代码并获取最终分数，带重试逻辑
    :param problem_id: 题目编号 (int)
    :param code: 代码 (str)
    :param wait_interval: 轮询间隔秒数
    :param timeout: 超时时间（秒）
    :param max_retries: 平台出错时最大重试次数
    :return: (status, score, log)
             status: "success" / "failed" / "platform_error"
             score: str or None
             log: str
    """

    for attempt in range(1, max_retries + 1):
        # Step 1. 提交代码
        submit_url = f"{BASE_URL}/answers/submit"
        payload = {"answer": code, "problems_id": problem_id}
        r = requests.post(submit_url, json=payload, cookies=COOKIE)
        e = None
        try:
            r.raise_for_status()
            res = r.json()
        except Exception as e:
            res = e
        log = ""
        if not res.get("success") or e:
            print(f"提交失败,准备重试: {res}")
            time.sleep(10)
        else:
            result_id = res["data"]["result_id"]
            print(f"[尝试 {attempt}/{max_retries}] 提交成功, result_id={result_id}")

            # Step 2. 轮询判题结果
            result_url = f"{BASE_URL}/submissions/result?result_id={result_id}"
            start_time = time.time()
            status = None

            while True:
                r = requests.get(result_url, cookies=COOKIE)
                r.raise_for_status()
                res = r.json()
                # print(res)  # 调试用

                jenkins_log = res.get("jenkins_log", "")
                if jenkins_log:
                    log = jenkins_log

                build_status = res["data"]["build"]["status"]
                test_status = res["data"]["test"]["status"]

                if test_status in ("success", "failed"):
                    status = test_status
                    break

                if time.time() - start_time > timeout:
                    raise TimeoutError("等待超时，判题还没完成")

                time.sleep(wait_interval)

            # Step 3. 查询最新提交分数
            list_url = f"{BASE_URL}/submissions/list?problem_id={problem_id}"
            r = requests.get(list_url, cookies=COOKIE)
            r.raise_for_status()
            res = r.json()

            score = None
            if res.get("success") and "data" in res:
                for item in res["data"]:
                    if item["result_id"] == result_id:
                        score = item.get("score")
                        break

            # Step 4. 判断结果
            if status == "success":
                return "success", score, log

            if status == "failed":
                if "no such file or directory" in log:  # 平台出错
                    print("⚠️ 平台异常: 文件缺失，30s 后重试")
                    if attempt < max_retries:
                        time.sleep(30)
                        continue
                    else:
                        return "platform_error", 0, log
                else:
                    return "failed", score, log
    # 如果所有尝试都失败
    return "platform_error", 0, log

class RewardingSystem:
    def __init__(self, config=None):
        self.config = config
        self.goal = self.config.get('goals')[0]
        if self.goal == 'gcu_var':
            self.problem_id = 11
        elif self.goal =='gcu_silu':
            self.problem_id = 12
        elif self.goal == 'gcu_gemm2':
            self.problem_id = 13
        else:
            raise NotImplementedError(f'golds should be gcu_var/silu/gemm2, yours is {self.goal}')
        
    def evaluate(self, items):
        invalid_num = 0
        for i,item in enumerate(items):
            code = item.value
            status,score,log = auto_submit(problem_id=self.problem_id,code=code)
            score = float(score)
            if status != 'success':
                invalid_num += 1
            results_dict = {
                'original_results': {
                    self.goal: score
                },
                'transformed_results': {
                    self.goal: 1-score/100
                },
                'constraint_results': {
                    'status': status,
                    'debug_log': log 
                },
                'overall_score': score # only this one cause passing mmseqs is enough
            }
            print(f' {i}th item result:',results_dict)
            item.assign_results(results_dict)
        log_dict = {}
        log_dict['invalid_num'] = invalid_num
        log_dict['repeated_num'] = 0
        return items, log_dict

import re

def generate_initial_population(config, seed=42, n_sample=6):
    random.seed(seed)

    goal = config.get('goals')[0]
    path = Path(f"/root/src/MOLLM/problem/gcu/{goal}_init.txt")
    text = path.read_text(encoding="utf-8")

    # 用正则提取 candidate, score, log
    # 注意：flags=re.S 允许跨行匹配
    pattern = re.compile(
        r"<candidate>(.*?)</candidate>\s*<score>(.*?)</score>\s*<log>(.*?)</log>",
        flags=re.S
    )
    matches = pattern.findall(text)
    indices = random.choices([i for i in range(len(matches))], k=n_sample-len(matches))
    indices.extend([i for i in range(len(matches))])
    # 每个元素是 (candidate, score, log)
    items = []
    i = 0
    for idx in indices:
        code, score, log = matches[idx]
        i+=1
        item = Item(code.strip(),config.get('goals'))
        score = float(score)
        status = 'success'
        log = log.strip()
        results_dict = {
                'original_results': {
                    goal: score
                },
                'transformed_results': {
                    goal: 1-score/100
                },
                'constraint_results': {
                    'status': status,
                    'debug_log': log 
                },
                'overall_score': score # only this one cause passing mmseqs is enough
            }
        print(f' {i}th item  result:',results_dict)
        item.assign_results(results_dict)
        items.append(item)

    return items



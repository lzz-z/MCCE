import requests
import json
import time
import argparse
import sys
import concurrent.futures
import threading
from typing import Dict, Any, List, Optional, Tuple

# 配置参数（参考alphafold3项目）
MAX_SEQUENCES = 52  # 最大批量序列数
DEFAULT_TIMEOUT = 30  # 默认请求超时
POLL_INTERVAL = 5  # 状态检查间隔
MAX_WAIT_TIME = 3600  # 最大等待时间（1小时）

# 参考alphafold3项目的测试序列
TEST_SEQUENCES = {
    "short": "YRWVFKAWGY",  # 19个氨基酸（原始D序列）
    "medium": "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALG",  # 50个氨基酸
    "default": "HKTDSFVGLM",  # 10个氨基酸（参考alphafold3项目默认序列）
}

class ABC_API_Client:
    """ABC API客户端（参考alphafold3项目模式）"""
    
    def __init__(self, base_url: str = "http://9.2.248.77:8001", timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.timeout = timeout
        self.session.timeout = timeout
        
        print(f"🌐 ABC+D API并行测试客户端")
        print(f"目标服务器: {self.base_url}")
        print(f"请求超时: {self.timeout}秒, 轮询间隔: {POLL_INTERVAL}秒")
        print("=" * 60)
    
    def test_health(self) -> bool:
        """测试健康检查（参考alphafold3项目）"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            data = response.json()
            
            print(f"✅ API服务健康: {data.get('status')}")
            print(f"   - 缓存可用: {data.get('cache_available')}")
            print(f"   - 时间戳: {data.get('timestamp')}")
            return data.get('cache_available', False)
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")
            return False
    
    def submit_prediction(self, sequence: str, job_name: str = None) -> str:
        """提交预测任务（参考alphafold3项目）"""
        print('submit sequence:',sequence)
        payload = {
            "sequence": sequence,
            "job_name": job_name
        }
        
        try:
            response = self.session.post(f"{self.base_url}/predict", json=payload)
            response.raise_for_status()
            data = response.json()
            
            job_id = data["job_id"]
            print(f"✅ 任务提交成功")
            print(f"   - 任务ID: {job_id}")
            print(f"   - 状态: {data['status']}")
            print(f"   - 消息: {data['message']}")
            
            return job_id
        except Exception as e:
            print(f"❌ 任务提交失败: {e}")
            return ""
    
    def get_job_status(self, job_id: str, retries: int = 3) -> Dict[str, Any]:
        """获取任务状态（参考alphafold3项目）"""
        for attempt in range(retries):
            try:
                # 为每次请求创建新的session，避免连接复用问题
                with requests.Session() as session:
                    session.timeout = min(self.timeout, 10)  # 限制单次请求超时
                    response = session.get(f"{self.base_url}/status/{job_id}", timeout=session.timeout)
                    response.raise_for_status()
                    return response.json()
            except (requests.exceptions.Timeout, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.HTTPError) as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间
                    print(f"⏰ 状态查询失败 (尝试{attempt+1}/{retries}): {type(e).__name__}, {wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 状态查询最终失败: {e}")
                    return {"status": "unknown", "message": f"连接失败: {e}"}
            except Exception as e:
                print(f"❌ 状态查询异常: {e}")
                return {"status": "unknown", "message": f"查询异常: {e}"}
        
        return {}
    
    def wait_for_completion(self, job_id: str, timeout: int = None) -> Dict[str, Any]:
        """等待任务完成（参考alphafold3项目）"""
        if timeout:
            print(f"⏳ 等待任务完成 (最大等待{timeout}秒)...")
        else:
            print(f"⏳ 等待任务完成 (无超时限制)...")
        
        start_time = time.time()
        last_status = ""
        
        while timeout is None or time.time() - start_time < timeout:
            status_data = self.get_job_status(job_id)
            if not status_data:
                break
            
            current_status = status_data.get("status", "unknown")
            message = status_data.get("message", "")
            gpu_id = status_data.get("assigned_gpu")
            
            # 显示状态更新
            if current_status != last_status:
                gpu_info = f" (GPU {gpu_id})" if gpu_id else ""
                print(f"   📊 {current_status}: {message}{gpu_info}")
                last_status = current_status
            
            if current_status == "completed":
                print(f"✅ 任务完成!")
                elapsed = time.time() - start_time
                print(f"   - 耗时: {elapsed:.1f}秒")
                if "output_dir" in status_data:
                    print(f"   - 输出目录: {status_data['output_dir']}")
                return status_data
            elif current_status == "failed":
                print(f"❌ 任务失败!")
                print(f"   - 错误信息: {status_data.get('message')}")
                return status_data
            
            time.sleep(5)  # 5秒检查一次
        
        if timeout:
            print(f"⏰ 等待超时 ({timeout}秒)")
        else:
            print(f"⏰ 等待被中断")
        return self.get_job_status(job_id)
    
    def list_jobs(self) -> Dict[str, Any]:
        """列出所有任务（参考alphafold3项目）"""
        try:
            response = self.session.get(f"{self.base_url}/jobs")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ 任务列表获取失败: {e}")
            return {}
    
    def run_prediction(self, sequence: str, job_name: str = None) -> bool:
        """运行完整预测流程（参考alphafold3项目）"""
        print(f"\n🧬 开始ABC+D预测")
        print(f"D序列: {sequence}")
        print(f"序列长度: {len(sequence)}")
        print("=" * 60)
        
        # 1. 健康检查
        if not self.test_health():
            print("❌ API服务不可用")
            return False
        
        print("\n" + "-" * 40)
        
        # 2. 提交任务
        job_id = self.submit_prediction(sequence, job_name)
        if not job_id:
            print("❌ 任务提交失败")
            return False
        
        print("\n" + "-" * 40)
        
        # 3. 等待完成
        final_status = self.wait_for_completion(job_id)
        
        print("\n" + "-" * 40)
        
        # 4. 显示结果
        if final_status.get("status") == "completed":
            print(f"🎉 预测成功完成!")
            if "result" in final_status:
                result = final_status["result"]
                if "summary_confidences" in result:
                    conf = result["summary_confidences"]
                    print(f"   - pTM: {conf.get('ptm', 'N/A')}")
                    print(f"   - ipTM: {conf.get('iptm', 'N/A')}")
                    print(f"   - ranking_score: {conf.get('ranking_score', 'N/A')}")
            return True
        else:
            print(f"❌ 预测失败: {final_status.get('message', 'Unknown error')}")
            return False

    def monitor_single_job(self, job_id: str, sequence: str, job_idx: int) -> Tuple[int, bool, Dict[str, Any]]:
        """监控单个任务（参考alphafold3项目）"""
        print(f"📊 任务{job_idx+1}: 开始监控 {job_id[:8]}... (D序列长度: {len(sequence)})")
        
        start_time = time.time()
        last_status = ""
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while time.time() - start_time < MAX_WAIT_TIME:
            try:
                status_data = self.get_job_status(job_id)
                
                # 如果获取状态失败
                if not status_data or status_data.get("status") == "unknown":
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"❌ 任务{job_idx+1}: 连续{max_consecutive_failures}次查询失败，可能任务已异常")
                        return job_idx, False, {"status": "failed", "message": "连续查询失败"}
                    
                    # 增加等待时间以减少服务器压力
                    backoff_time = min(POLL_INTERVAL * (consecutive_failures + 1), 30)
                    time.sleep(backoff_time)
                    continue
                
                # 成功获取状态，重置失败计数
                consecutive_failures = 0
                
                current_status = status_data.get("status", "unknown")
                message = status_data.get("message", "")
                gpu_id = status_data.get("assigned_gpu")
                
                # 显示状态更新
                if current_status != last_status:
                    gpu_info = f" (GPU {gpu_id})" if gpu_id else ""
                    elapsed = time.time() - start_time
                    print(f"   📊 任务{job_idx+1}: {current_status}: {message}{gpu_info} [耗时{elapsed:.0f}s]")
                    last_status = current_status
                
                if current_status == "completed":
                    elapsed = time.time() - start_time
                    print(f"✅ 任务{job_idx+1}: 完成! 总耗时: {elapsed:.1f}秒")
                    if "result" in status_data and "summary_confidences" in status_data["result"]:
                        conf = status_data["result"]["summary_confidences"]
                        print(f"   - pTM: {conf.get('ptm', 'N/A'):.3f}, ipTM: {conf.get('iptm', 'N/A'):.3f}, ranking_score: {conf.get('ranking_score', 'N/A'):.3f}")
                    return job_idx, True, status_data
                elif current_status == "failed":
                    print(f"❌ 任务{job_idx+1}: 失败! {message}")
                    return job_idx, False, status_data
                
                # 正常等待
                time.sleep(POLL_INTERVAL)
                
            except Exception as e:
                consecutive_failures += 1
                print(f"❌ 任务{job_idx+1}: 监控异常: {e}")
                time.sleep(POLL_INTERVAL * 2)  # 异常时延长等待
        
        print(f"⏰ 任务{job_idx+1}: 监控超时 ({MAX_WAIT_TIME}秒)")
        final_status = self.get_job_status(job_id, retries=1)  # 最后尝试一次
        return job_idx, False, final_status

    def run_batch_predictions(self, sequences: List[str], job_prefix: str = "batch") -> Dict[str, Any]:
        """运行批量预测（参考alphafold3项目模式）"""
        if len(sequences) > MAX_SEQUENCES:
            print(f"❌ 序列数量超过限制 ({len(sequences)} > {MAX_SEQUENCES})")
            return {"success": False, "error": "Too many sequences"}
        
        print(f"\n🧬 开始ABC+D批量预测")
        print(f"序列数量: {len(sequences)}")
        print(f"最大并发: {min(len(sequences), MAX_SEQUENCES)}")
        print("=" * 60)
        
        # 1. 健康检查
        if not self.test_health():
            print("❌ API服务不可用")
            return {"success": False, "error": "API not available"}
        
        print("\n" + "-" * 40)
        
        # 2. 批量提交任务
        print(f"📤 批量提交 {len(sequences)} 个任务...")
        job_infos = []
        submit_start = time.time()
        
        for i, sequence in enumerate(sequences):
            job_name = f"{job_prefix}_{i+1:02d}_D{len(sequence)}_{int(time.time())}"
            job_id = self.submit_prediction(sequence, job_name)
            if job_id:
                job_infos.append({
                    "index": i,
                    "job_id": job_id,
                    "sequence": sequence,
                    "job_name": job_name,
                    "submitted_at": time.time()
                })
                print(f"✅ 任务{i+1}: 已提交 {job_id[:8]}... (D序列长度: {len(sequence)})")
            else:
                print(f"❌ 任务{i+1}: 提交失败")
        
        submit_time = time.time() - submit_start
        print(f"📤 批量提交完成，耗时: {submit_time:.2f}秒")
        print(f"   - 成功提交: {len(job_infos)}/{len(sequences)} 个任务")
        
        if not job_infos:
            print("❌ 没有任务成功提交")
            return {"success": False, "error": "No jobs submitted"}
        
        print("\n" + "-" * 40)
        
        # 3. 并行监控所有任务
        print(f"🔍 开始并行监控 {len(job_infos)} 个任务...")
        monitor_start = time.time()
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(job_infos)) as executor:
            # 提交所有监控任务
            future_to_job = {
                executor.submit(
                    self.monitor_single_job, 
                    job_info["job_id"], 
                    job_info["sequence"], 
                    job_info["index"]
                ): job_info for job_info in job_infos
            }
            
            # 等待所有任务完成
            completed_count = 0
            failed_count = 0
            
            for future in concurrent.futures.as_completed(future_to_job):
                job_info = future_to_job[future]
                try:
                    job_idx, success, status_data = future.result()
                    results[job_idx] = {
                        "job_info": job_info,
                        "success": success,
                        "status_data": status_data,
                        "sequence": job_info["sequence"]
                    }
                    if success:
                        completed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"❌ 任务监控异常: {e}")
                    failed_count += 1
        
        monitor_time = time.time() - monitor_start
        print(f"\n🔍 批量监控完成，耗时: {monitor_time:.2f}秒")
        
        # 4. 汇总结果
        print("\n" + "-" * 40)
        print(f"📊 批量预测汇总结果:")
        print(f"   - 总任务数: {len(sequences)}")
        print(f"   - 成功提交: {len(job_infos)}")
        print(f"   - 成功完成: {completed_count}")
        print(f"   - 失败数量: {failed_count}")
        print(f"   - 总耗时: {submit_time + monitor_time:.2f}秒")
        
        # 显示详细结果
        if completed_count > 0:
            print(f"\n✅ 成功完成的任务:")
            for idx in sorted(results.keys()):
                if results[idx]["success"]:
                    result_data = results[idx]
                    sequence = result_data["sequence"]
                    status_data = result_data["status_data"]
                    if "result" in status_data and "summary_confidences" in status_data["result"]:
                        conf = status_data["result"]["summary_confidences"]
                        print(f"   任务{idx+1}: D{len(sequence)} -> pTM:{conf.get('ptm', 'N/A'):.3f}, ipTM:{conf.get('iptm', 'N/A'):.3f}, ranking_score:{conf.get('ranking_score', 'N/A'):.3f}")
        
        return {
            "success": completed_count > 0,
            "total_jobs": len(sequences),
            "submitted_jobs": len(job_infos),
            "completed_jobs": completed_count,
            "failed_jobs": failed_count,
            "submit_time": submit_time,
            "monitor_time": monitor_time,
            "total_time": submit_time + monitor_time,
            "results": results
        }


def parse_sequences_input(sequences_arg: str) -> List[str]:
    """解析序列输入（参考alphafold3项目）"""
    sequences = []
    
    # 如果是文件路径
    if sequences_arg.endswith('.json'):
        try:
            with open(sequences_arg, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                sequences = [seq.strip().upper() for seq in data if seq.strip()]
            elif isinstance(data, dict) and 'sequences' in data:
                sequences = [seq.strip().upper() for seq in data['sequences'] if seq.strip()]
            else:
                print(f"❌ JSON文件格式不支持: {sequences_arg}")
                return []
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return []
    else:
        # 逗号分隔的序列
        sequences = [seq.strip().upper() for seq in sequences_arg.split(',') if seq.strip()]
    
    # 验证序列
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    valid_sequences = []
    for i, seq in enumerate(sequences):
        if not all(c in valid_amino_acids for c in seq):
            print(f"❌ 序列{i+1}包含无效字符: {seq}")
        else:
            valid_sequences.append(seq)
    
    return valid_sequences


def main():
    parser = argparse.ArgumentParser(
        description="测试AlphaFold3 ABC+D API（参考alphafold3项目模式，支持批量处理）"
    )
    parser.add_argument("sequences", nargs="*", help="D序列（支持多个序列作为参数，或单个JSON文件路径）")
    parser.add_argument("--url", default="http://192.168.13.83:8001", help="API服务器URL")
    parser.add_argument("--preset", choices=list(TEST_SEQUENCES.keys()), 
                       help="使用预设序列",default='short')
    parser.add_argument("--job-name", help="任务名称（批量时作为前缀）")
    parser.add_argument("--action", choices=["predict", "batch", "health", "list"], 
                       default="predict", help="要执行的操作")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help=f"批量处理时的最大并发数（最大{MAX_SEQUENCES}）")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                       help="请求超时时间（秒）")
    
    args = parser.parse_args('')
    
    # 创建API客户端
    client = ABC_API_Client(args.url, args.timeout)
    
    if args.action == "health":
        client.test_health()
        return
    elif args.action == "list":
        jobs = client.list_jobs()
        print(f"📋 任务列表 (共{jobs.get('total_jobs', 0)}个):")
        for job in jobs.get('jobs', []):
            print(f"   - {job['job_id'][:8]}... | {job['status']} | 序列长度: {job.get('sequence_length', 'N/A')}")
        return
    
    # 确定要使用的序列
    sequences = []
    if args.sequences:
        # 支持多种输入方式
        if len(args.sequences) == 1:
            # 单个参数：可能是单个序列、逗号分隔的多个序列，或JSON文件
            single_arg = args.sequences[0]
            if single_arg.endswith('.json'):
                # JSON文件
                sequences = parse_sequences_input(single_arg)
            elif ',' in single_arg:
                # 逗号分隔的序列
                sequences = parse_sequences_input(single_arg)
            else:
                # 单个序列
                sequences = [single_arg.strip().upper()]
        else:
            # 多个参数：每个都是一个序列
            sequences = [seq.strip().upper() for seq in args.sequences if seq.strip()]
        
        # 验证序列
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        valid_sequences = []
        for i, seq in enumerate(sequences):
            if not all(c in valid_amino_acids for c in seq):
                print(f"❌ 序列{i+1}包含无效字符: {seq}")
            else:
                valid_sequences.append(seq)
        sequences = valid_sequences
        
    elif args.preset:
        sequences = [TEST_SEQUENCES[args.preset]]
    else:
        print("❌ 请指定序列或使用 --preset 参数")
        print(f"可用预设: {list(TEST_SEQUENCES.keys())}")
        print("示例:")
        print(f"  # 单个序列")
        print(f"  python {sys.argv[0]} YRWVFKAWGYRLVWQKIRW")
        print(f"  python {sys.argv[0]} --preset short")
        print(f"  # 多个序列（直接作为参数）")
        print(f"  python {sys.argv[0]} SEQ1 SEQ2 SEQ3")
        print(f"  # 逗号分隔的序列")
        print(f"  python {sys.argv[0]} 'SEQ1,SEQ2,SEQ3'")
        print(f"  # JSON文件")
        print(f"  python {sys.argv[0]} sequences.json")
        return
    
    if not sequences:
        print("❌ 没有有效的序列")
        return
    
    if len(sequences) > MAX_SEQUENCES:
        print(f"❌ 序列数量超过限制 ({len(sequences)} > {MAX_SEQUENCES})")
        return
    
    # 生成任务名称
    job_prefix = args.job_name or "test"
    
    # 根据action和序列数量决定执行模式
    if args.action == "batch" or len(sequences) > 1:
        print(f"🚀 批量模式: {len(sequences)} 个序列")
        result = client.run_batch_predictions(sequences, job_prefix)
        success = result.get("success", False)
        
        # 保存结果到文件
        if result.get("results"):
            output_file = f"batch_results_{int(time.time())}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"📄 详细结果已保存到: {output_file}")
        
        sys.exit(0 if success else 1)
    else:
        print(f"🚀 单任务模式: 1 个序列")
        sequence = sequences[0]
        job_name = f"{job_prefix}_D{len(sequence)}_{int(time.time())}"
        success = client.run_prediction(sequence, job_name)
        sys.exit(0 if success else 1)

from typing import List, Dict, Any
import time
import json

def run_peptide_batch(sequences: List[str],
                      base_url: str = "http://192.168.13.83:8001",
                      timeout: int = DEFAULT_TIMEOUT,
                      job_prefix: str = "test") -> Dict[str, Any]:
    """
    运行批量预测：输入若干条 peptide 序列，返回 {sequence: result_dict}
    """
    client = ABC_API_Client(base_url, timeout)

    # 验证序列合法性
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    clean_sequences = []
    for seq in sequences:
        seq = seq.strip().upper()
        if not seq:
            continue
        if all(c in valid_amino_acids for c in seq):
            clean_sequences.append(seq)
        else:
            print(f"⚠️ 序列包含非法字符: {seq}")

    if not clean_sequences:
        return {}

    # 调用原有批量预测逻辑
    result = client.run_batch_predictions(clean_sequences, job_prefix)
    
    
    # 整理成 {sequence: status_data}
    sequence_results: Dict[str, Any] = {}
    for idx, seq in enumerate(clean_sequences):
        job_data = result["results"].get(idx)
        if job_data:
            sequence_results[seq] = job_data["status_data"]
        else:
            sequence_results[seq] = {"status": "not_submitted"}

    return sequence_results,result

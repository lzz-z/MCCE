#!/usr/bin/env python3
"""
AlphaFold3 API 远程访问测试脚本
用于从其他机器测试API服务
"""

import requests
import time
import json
import sys
import argparse
from urllib.parse import urlparse

class RemoteAPITester:
    def __init__(self, api_host, api_port=8000, timeout=30):
        self.base_url = f"http://{api_host}:{api_port}"
        self.timeout = timeout
        self.session = requests.Session()
        
        # 设置请求头
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'AlphaFold3-Remote-Client/1.0'
        })
        
        print(f"🌐 远程API测试客户端")
        print(f"目标服务器: {self.base_url}")
        print(f"请求超时: {self.timeout}秒")
        print("="*50)
    
    def test_connectivity(self):
        """测试网络连通性"""
        print("=== 网络连通性测试 ===")
        try:
            # 解析URL
            parsed = urlparse(self.base_url)
            host = parsed.hostname
            port = parsed.port or 8000
            
            print(f"测试连接: {host}:{port}")
            
            # 简单的HTTP请求测试
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            print(f"✅ HTTP连接成功")
            print(f"响应状态码: {response.status_code}")
            print(f"响应时间: {response.elapsed.total_seconds():.2f}秒")
            return True
            
        except requests.exceptions.ConnectTimeout:
            print("❌ 连接超时 - 检查网络连接或防火墙设置")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"❌ 连接错误: {e}")
            print("可能的原因:")
            print("  1. API服务器未启动")
            print("  2. 防火墙阻止了连接")
            print("  3. IP地址或端口错误")
            return False
        except Exception as e:
            print(f"❌ 其他错误: {e}")
            return False
    
    def test_health(self):
        """健康检查"""
        print("\n=== API健康检查 ===")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            
            health = response.json()
            print(f"健康状态: {health}")
            
            if health.get("status") == "healthy":
                print("✅ API服务健康")
                if health.get("cache_available"):
                    print("✅ 预计算缓存可用")
                else:
                    print("⚠️ 预计算缓存不可用")
                return True
            else:
                print("❌ API服务不健康")
                return False
                
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP错误: {e}")
            return False
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
            return False
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")
            return False
    
    def submit_prediction(self, sequence, job_name=None):
        """提交预测任务"""
        print("\n=== 提交预测任务 ===")
        
        if not job_name:
            job_name = f"远程测试_{sequence[:10]}"
        
        print(f"序列: {sequence}")
        print(f"序列长度: {len(sequence)}")
        print(f"任务名称: {job_name}")
        
        payload = {
            "sequence": sequence,
            "job_name": job_name
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict", 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            job_info = response.json()
            print(f"✅ 任务提交成功")
            print(f"任务ID: {job_info.get('job_id')}")
            print(f"初始状态: {job_info.get('status')}")
            
            return job_info.get('job_id')
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP错误: {e}")
            if hasattr(e.response, 'text'):
                print(f"错误详情: {e.response.text}")
            return None
        except Exception as e:
            print(f"❌ 提交任务失败: {e}")
            return None
    
    def monitor_job(self, job_id, max_wait_time=1800, poll_interval=60):
        """监控任务进度"""
        print(f"\n=== 监控任务进度 ===")
        print(f"任务ID: {job_id}")
        print(f"最大等待时间: {max_wait_time}秒 ({max_wait_time//60}分钟)")
        print(f"查询间隔: {poll_interval}秒")
        
        start_time = time.time()
        last_status = None
        check_count = 0
        
        while time.time() - start_time < max_wait_time:
            try:
                response = self.session.get(
                    f"{self.base_url}/status/{job_id}", 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                status_info = response.json()
                current_status = status_info.get("status")
                message = status_info.get("message", "")
                
                # 显示进度信息
                elapsed = time.time() - start_time
                check_count += 1
                
                if current_status != last_status or check_count % 6 == 1:  # 每90秒显示一次
                    print(f"[{elapsed:.0f}s] 状态: {current_status} - {message}")
                    last_status = current_status
                
                if current_status == "completed":
                    print(f"✅ 任务完成! 总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
                    return status_info
                    
                elif current_status == "failed":
                    print(f"❌ 任务失败: {message}")
                    return status_info
                
                # 等待下次查询
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                print("\n⚠️ 用户中断监控")
                return None
            except requests.exceptions.Timeout:
                print(f"⚠️ 查询超时，继续等待...")
                time.sleep(5)
            except Exception as e:
                print(f"⚠️ 查询状态异常: {e}")
                time.sleep(poll_interval)
        
        print("❌ 监控超时")
        return None
    
    def analyze_result(self, result_info, save_file=None):
        """分析预测结果"""
        print("\n=== 分析预测结果 ===")
        
        if not result_info or result_info.get("status") != "completed":
            print("❌ 无有效结果可分析")
            return False,0
        
        result = result_info.get("result", {})
        confidence_data = result.get("summary_confidences", {})
        
        if confidence_data:
            print("🎯 置信度指标:")
            print(f"  iptm (接口置信度):     {confidence_data.get('iptm', 'N/A')}")
            print(f"  ptm (蛋白质置信度):    {confidence_data.get('ptm', 'N/A')}")
            print(f"  ranking_score (排名):  {confidence_data.get('ranking_score', 'N/A')}")
            print(f"  fraction_disordered:   {confidence_data.get('fraction_disordered', 'N/A')}")
            print(f"  has_clash (结构冲突):  {confidence_data.get('has_clash', 'N/A')}")
            
            # 解释置信度
            iptm = confidence_data.get('iptm', 0)
            ptm = confidence_data.get('ptm', 0)
            
            print(f"\n📊 结果解读:")
            if iptm >= 0.8:
                print(f"  接口质量: 优秀 (iptm={iptm})")
            elif iptm >= 0.5:
                print(f"  接口质量: 良好 (iptm={iptm})")
            else:
                print(f"  接口质量: 较低 (iptm={iptm})")
            
            if ptm >= 0.8:
                print(f"  结构质量: 优秀 (ptm={ptm})")
            elif ptm >= 0.5:
                print(f"  结构质量: 良好 (ptm={ptm})")
            else:
                print(f"  结构质量: 较低 (ptm={ptm})")
            
            # 保存结果
            if not save_file:
                save_file = f"remote_result_{int(time.time())}.json"
            
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 完整结果已保存: {save_file}")
            
            return True, iptm
        else:
            print("❌ 未找到置信度数据")
            return False, 0
    
    def run_full_test(self, sequence, job_name=None):
        """运行完整的远程测试"""
        print("🚀 开始远程AlphaFold3 API测试")
        print("="*60)
        
        # 1. 网络连通性测试
        if not self.test_connectivity():
            return False
        
        # 2. 健康检查
        if not self.test_health():
            return False
        
        # 3. 提交任务
        job_id = self.submit_prediction(sequence, job_name)
        if not job_id:
            return False
        
        # 4. 监控任务
        result_info = self.monitor_job(job_id)
        if not result_info:
            return False
        
        # 5. 分析结果
        success,iptm = self.analyze_result(result_info)
        
        print("="*60)
        if success:
            print("🎉 远程测试完成!")
        else:
            print("❌ 远程测试失败")
        
        return success,iptm

def main():
    parser = argparse.ArgumentParser(description="AlphaFold3 API 远程访问测试")
    parser.add_argument("host", help="API服务器IP地址或域名")
    parser.add_argument("-p", "--port", type=int, default=8000, help="API服务器端口 (默认: 8000)")
    parser.add_argument("-s", "--sequence", default="HKTDSFVGLML", help="测试序列 (默认: HKTDSFVGLML)")
    parser.add_argument("-n", "--name", help="任务名称")
    parser.add_argument("-t", "--timeout", type=int, default=30, help="请求超时时间 (默认: 30秒)")
    parser.add_argument("--quick", action="store_true", help="快速测试模式 (仅连通性和健康检查)")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = RemoteAPITester(args.host, args.port, args.timeout)
    
    if args.quick:
        # 快速测试模式
        print("🔍 快速测试模式")
        success = tester.test_connectivity() and tester.test_health()
        if success:
            print("✅ API服务可正常访问")
        else:
            print("❌ API服务访问异常")
        return
    
    # 完整测试
    try:
        success = tester.run_full_test(args.sequence, args.name)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        sys.exit(1)

if __name__ == "__main__":
    # 如果没有命令行参数，提供交互式输入
    if len(sys.argv) == 1:
        print("🌐 AlphaFold3 API 远程测试工具")
        print("="*40)
        
        try:
            host = input("请输入API服务器IP地址或域名: ").strip()
            if not host:
                print("❌ 必须提供服务器地址")
                sys.exit(1)
            
            port_input = input("请输入端口号 (默认8000): ").strip()
            port = int(port_input) if port_input else 8000
            
            sequence_input = input("请输入测试序列 (默认HKTDSFVGLML): ").strip()
            sequence = sequence_input if sequence_input else "HKTDSFVGLML"
            
            job_name = input("请输入任务名称 (可选): ").strip() or None
            
            # 创建测试器并运行
            tester = RemoteAPITester(host, port)
            success = tester.run_full_test(sequence, job_name)
            
            sys.exit(0 if success else 1)
            
        except KeyboardInterrupt:
            print("\n⚠️ 测试被用户中断")
            sys.exit(1)
        except ValueError as e:
            print(f"❌ 输入错误: {e}")
            sys.exit(1)
    else:
        main()
        
        
        


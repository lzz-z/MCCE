#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import time
import shutil
import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

# --- 设置顶级日志记录器 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


class SacsRunner:
    """
    SACS运行器，专为在WSL (Windows Subsystem for Linux) 环境下调用Windows中的SACS程序而设计。
    """

    def __init__(self, project_path: Optional[str] = None, sacs_install_path: Optional[str] = None):
        """
        初始化SACS运行器。
        - 设定 SACS 项目在 WSL 中的路径。
        - 设定 SACS 在 Windows 中的安装路径。
        - 预先处理好所有需要的 WSL 路径和 Windows 路径。
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        wsl_project_path_str = project_path or "/mnt/d/Python project/sacs_llm/demo06_project/Demo06"
        win_sacs_install_path_str = sacs_install_path or r"C:\Program Files (x86)\Bentley\Engineering\SACS CONNECT Edition V16 Update 1"

        self.project_path: Path = Path(wsl_project_path_str)
        self.win_sacs_install_path_str: str = win_sacs_install_path_str

        try:
            self.win_project_path_str: str = self._wsl_to_windows_path(self.project_path)
            self.win_runx_path_str: str = self._wsl_to_windows_path(self.project_path / "demo06.runx")
            self.wsl_engine_path_str: str = self._windows_to_wsl_path(
                f"{self.win_sacs_install_path_str}\\AnalysisEngine.exe")
        except Exception as e:
            self.logger.error(f"路径转换失败: {e}")
            raise RuntimeError("路径转换失败，请确保在WSL环境中运行，并且`wslpath`命令可用。") from e

        self.input_file: Path = self.project_path / "sacinp.demo06"
        self.db_file: Path = self.project_path / "sacsdb.db"
        self.backup_dir: Path = self.project_path / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self._verify_wsl_environment()

        self.logger.info(f"SacsRunner 初始化成功。项目路径 (WSL): {self.project_path}")
        self.logger.info(f"SACS 引擎路径 (WSL): {self.wsl_engine_path_str}")

    def _wsl_to_windows_path(self, wsl_path: Path) -> str:
        result = subprocess.run(['wslpath', '-w', str(wsl_path)], capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def _windows_to_wsl_path(self, win_path: str) -> str:
        result = subprocess.run(['wslpath', '-u', win_path], capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def _verify_wsl_environment(self):
        if not self.project_path.exists():
            raise FileNotFoundError(f"项目目录在WSL中不存在: {self.project_path}")
        if not self.input_file.exists():
            raise FileNotFoundError(f"SACS输入文件在WSL中不存在: {self.input_file}")
        if not Path(self.wsl_engine_path_str).exists():
            raise FileNotFoundError(f"SACS引擎在WSL中不可见或路径有误: {self.wsl_engine_path_str}")
        self.logger.info("SACS 运行环境 (WSL侧) 验证通过。")

    def run_analysis(self, timeout: int = 300, cleanup_old_results: bool = True) -> Dict[str, Any]:
        self.logger.info("=" * 20 + " 开始SACS分析 " + "=" * 20)
        start_time = time.time()
        try:
            if cleanup_old_results: self._cleanup_old_results()
            backup_path = self._create_backup("before_run")
            run_result = self._execute_sacs_on_windows(timeout)

            if not run_result['success']:
                run_result.update({'execution_time': time.time() - start_time,
                                   'backup_path': str(backup_path) if backup_path else None})
                return run_result

            if not self._wait_for_database():
                return {'success': False, 'error': '数据库文件生成超时或无效',
                        'execution_time': time.time() - start_time,
                        'backup_path': str(backup_path) if backup_path else None}

            validation_result = self._validate_results()
            execution_time = time.time() - start_time
            result = {'success': True, 'execution_time': execution_time,
                      'backup_path': str(backup_path) if backup_path else None, 'database_path': str(self.db_file),
                      'validation': validation_result, 'output_files': self._get_output_files(),
                      'timestamp': datetime.now().isoformat()}
            self.logger.info(f"SACS分析成功完成，耗时: {execution_time:.2f}秒。")
            return result
        except Exception as e:
            self.logger.critical(f"SACS分析过程中发生未捕获的异常: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'execution_time': time.time() - start_time}
        finally:
            self.logger.info("=" * 20 + " SACS分析结束 " + "=" * 20)

    def _execute_sacs_on_windows(self, timeout: int) -> Dict[str, Any]:
        command_args = [
            self.wsl_engine_path_str,
            self.win_runx_path_str,
            self.win_sacs_install_path_str
        ]

        working_directory_wsl = str(self.project_path)

        self.logger.info(f"即将直接执行Windows程序: {command_args}")
        self.logger.info(f"在WSL工作目录中运行 (将自动映射到Windows): {working_directory_wsl}")

        try:
            process = subprocess.run(
                command_args,
                cwd=working_directory_wsl,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
                encoding='gbk',
                errors='ignore'
            )

            self.logger.info("SACS引擎成功执行。")
            return {'success': True, 'process': process}

        except FileNotFoundError:
            msg = f"执行失败：在WSL中找不到 '{self.wsl_engine_path_str}'。请检查路径中的空格、权限，并确保wslpath转换正确。"
            self.logger.critical(msg)
            return {'success': False, 'error': msg}
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SACS分析失败，返回码非零: {e.returncode}")
            self.logger.error(f"SACS 输出:\n{e.stderr or e.stdout}")
            return {'success': False, 'error': e.stderr or e.stdout, 'process': e}
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"SACS分析超时（超过 {timeout} 秒）。")
            return {'success': False, 'error': 'TimeoutExpired', 'process': e}
        except Exception as e:
            self.logger.critical(f"执行SACS时发生未知严重错误: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _cleanup_old_results(self):
        self.logger.info("正在清理旧的分析结果...")
        patterns = ['*.db', '*.lst', '*.out', '*.log', '*.err', '*.tmp', '*.ftg*']
        for p in patterns:
            for f in self.project_path.glob(p):
                try:
                    f.unlink()
                except OSError as e:
                    self.logger.warning(f"删除文件失败 {f.name}: {e}")

    def _create_backup(self, desc: str) -> Optional[Path]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"sacinp_{desc}_{ts}.demo06"
        try:
            shutil.copy2(self.input_file, backup_path)
            self.logger.info(f"已创建输入文件备份: {backup_path.name}")
            return backup_path
        except Exception as e:
            self.logger.error(f"创建备份失败: {e}")
            return None

    def _wait_for_database(self, timeout: int = 60) -> bool:
        self.logger.info(f"等待数据库文件 '{self.db_file.name}' 生成...")
        end_time = time.time() + timeout
        while time.time() < end_time:
            if self.db_file.exists() and self.db_file.stat().st_size > 0:
                time.sleep(2)
                if self._check_database_integrity():
                    self.logger.info("数据库文件已生成并可访问。")
                    return True
            time.sleep(1)
        self.logger.error(f"等待数据库文件超时 ({timeout}秒)。")
        return False

    def _check_database_integrity(self) -> bool:
        try:
            with sqlite3.connect(f"file:{self.db_file}?mode=ro", uri=True) as conn:
                conn.cursor().execute("SELECT name FROM sqlite_master WHERE type='table'")
                return True
        except sqlite3.Error as e:
            self.logger.warning(f"数据库完整性检查失败: {e}")
            return False

    def _validate_results(self) -> Dict[str, Any]:
        val = {'db_exists': self.db_file.exists(), 'db_accessible': False, 'tables': [], 'errors': []}
        if not val['db_exists']: return val
        try:
            with sqlite3.connect(f"file:{self.db_file}?mode=ro", uri=True) as conn:
                val['db_accessible'] = True
                val['tables'] = [r[0] for r in
                                 conn.cursor().execute("SELECT name FROM sqlite_master WHERE type='table'")]
        except sqlite3.Error as e:
            val['errors'].append(f"数据库验证失败: {e}")
        return val

    # ####################################################################
    #
    # 最终修正点：修正文件搜索通配符
    #
    # ####################################################################
    def _get_output_files(self) -> List[str]:
        # 修正前: ['.db', '.lst', '.out', '.log', '.err', '*.ftg']
        # 修正后:
        exts = ['.db', '.lst', '.out', '.log', '.err', '.ftg']  # <--- 移除了多余的 '*'
        return sorted([str(p) for ext in exts for p in self.project_path.glob(f"*{ext}")])


class SacsAnalysisManager:
    def __init__(self, project_path: str = None, sacs_install_path: str = None):
        self.runner = SacsRunner(project_path, sacs_install_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_with_retry(self, max_retries: int = 2, timeout: int = 300) -> Dict[str, Any]:
        for attempt in range(max_retries):
            self.logger.info(f"运行SACS分析 (尝试 {attempt + 1}/{max_retries})")
            result = self.runner.run_analysis(timeout=timeout)
            if result.get('success'): return result
            if attempt < max_retries - 1:
                self.logger.warning(f"尝试 {attempt + 1} 失败，5秒后重试...")
                time.sleep(5)
        self.logger.error("所有重试均失败。")
        return result


def run_sacs_analysis(project_path: str = None, sacs_install_path: str = None, timeout: int = 300):
    return SacsAnalysisManager(project_path, sacs_install_path).run_with_retry(max_retries=2, timeout=timeout)


def quick_run_sacs(project_path: str = None, sacs_install_path: str = None, timeout: int = 120):
    return run_sacs_analysis(project_path, sacs_install_path, timeout).get('success', False)


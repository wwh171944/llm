# E:\llm_project\utils\log_utils.py
import os
import sys
import json
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


class Tee:
    """同时输出到控制台和文件"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


def start_log(script_name: str):
    """重定向 stdout/stderr 到 logs 文件，返回日志文件路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.txt")
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee
    print(f"[LOG] 日志文件：{log_path}")
    print(f"[LOG] 开始时间：{datetime.now().isoformat()}")
    return log_path


def save_json(script_name: str, data: dict):
    """保存结构化指标到 JSON，方便 Dashboard 读取"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[LOG] 指标 JSON：{json_path}")
    return json_path


def timer():
    """返回一个计时上下文，用于测量时延"""
    class Timer:
        def __init__(self):
            self.start = None
            self.end = None

        def __enter__(self):
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start = time.time()
            return self

        def __exit__(self, *args):
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.end = time.time()

        @property
        def ms(self):
            return (self.end - self.start) * 1000

    return Timer()
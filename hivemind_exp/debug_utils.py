import platform
import subprocess
import sys
import logging
from pathlib import Path

import psutil
import colorlog

DIVIDER = "[========= 系统信息 =========]"


def print_system_info():
    lines = ['\n']
    lines.append(DIVIDER)
    lines.append("")
    lines.append("🧠 Python 版本:")
    lines.append(f"  {sys.version}")

    lines.append("\n🖥️ 平台信息:")
    lines.append(f"  系统: {platform.system()}")
    lines.append(f"  版本: {platform.release()} - {platform.version()}")
    lines.append(f"  架构: {platform.machine()}")
    lines.append(f"  处理器: {platform.processor()}")

    lines.append("\n🧩 CPU 信息:")
    lines.append(f"  实体核心数: {psutil.cpu_count(logical=False)}")
    lines.append(f"  总线程数:   {psutil.cpu_count(logical=True)}")
    cpu_freq = psutil.cpu_freq()
    lines.append(f"  最大频率:   {cpu_freq.max:.2f} MHz")
    lines.append(f"  当前频率:   {cpu_freq.current:.2f} MHz")

    lines.append("\n💾 内存信息:")
    vm = psutil.virtual_memory()
    lines.append(f"  总内存: {vm.total / (1024**3):.2f} GB")
    lines.append(f"  可用:   {vm.available / (1024**3):.2f} GB")
    lines.append(f"  已用:   {vm.used / (1024**3):.2f} GB")

    lines.append("\n🍎 Apple 芯片:")
    try:
        cpu_brand = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
        lines.append(f"  芯片型号: {cpu_brand}")
        import torch
        if torch.backends.mps.is_available():
            lines.append("  MPS 加速: ✅ 可用")
            lines.append(f"  MPS 设备: {torch.device('mps')}")
        else:
            lines.append("  MPS 加速: ❌ 不可用")
    except Exception as e:
        lines.append(f"  MPS 检测失败: {e}")

    lines.append("")
    lines.append(DIVIDER)
    return "\n".join(lines)


class TeeHandler(logging.Handler):
    def __init__(self, filename, mode='a', console_level=logging.INFO, file_level=logging.DEBUG):
        super().__init__()
        from colorlog import ColoredFormatter, StreamHandler

        self.console_handler = StreamHandler()
        self.console_handler.setLevel(console_level)
        self.console_handler.setFormatter(
            ColoredFormatter("%(green)s%(levelname)s:%(name)s:%(message)s")
        )

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self.file_handler = logging.FileHandler(filename, mode=mode)
        self.file_handler.setLevel(file_level)
        self.file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s")
        )

    def emit(self, record):
        if record.levelno >= self.console_handler.level:
            self.console_handler.emit(record)
        if record.levelno >= self.file_handler.level:
            self.file_handler.emit(record)


class PrintCapture:
    def __init__(self, logger):
        self.logger = logger
        self.original_stdout = sys.stdout

    def write(self, buf):
        self.original_stdout.write(buf)
        for line in buf.rstrip().splitlines():
            if line.strip():
                self.logger.debug(f"[PRINT] {line.rstrip()}")

    def flush(self):
        self.original_stdout.flush()

    def __getattr__(self, attr):
        return getattr(self.original_stdout, attr)


def log_memory_usage(stage=""):
    vm = psutil.virtual_memory()
    print(f"\n🧠 \033[95m[内存监控] —— {stage}\033[0m")
    print(f"  • 总内存容量:    \033[96m{vm.total / (1024 ** 3):6.2f} GB\033[0m")
    print(f"  • 已使用内存:    \033[91m{vm.used / (1024 ** 3):6.2f} GB\033[0m")
    print(f"  • 可用内存:      \033[92m{vm.available / (1024 ** 3):6.2f} GB\033[0m")
    print(f"  • 使用比例:      \033[93m{vm.percent:5.1f}%\033[0m")

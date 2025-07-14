from typing import List
import functools
import time
import torch


def get_total_gpu_memory_allocated_mb() -> float:
    """计算并返回所有可用CUDA设备上已分配显存的总和（单位MB）。"""
    if not torch.cuda.is_available():
        return 0.0
    total_mem = 0
    for i in range(torch.cuda.device_count()):
        total_mem += torch.cuda.memory_allocated(device=i)
    return total_mem / 1024**2


# --- 新增：性能指标装饰器（已升级为多GPU感知） ---
class with_metrics:
    """
    一个用于测量函数性能指标（如执行时间、GPU显存消耗）的装饰器。
    """

    def __init__(self, metrics: List[str]):
        valid_metrics = ["time", "gpu_memory"]
        if not all(m in valid_metrics for m in metrics):
            raise ValueError(f"指定的指标无效。请从 {valid_metrics} 中选择。")
        self.metrics = metrics

    def __call__(self, func):
        decorator_self = self

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = None
            initial_gpu_mems = []
            use_time = "time" in decorator_self.metrics
            use_gpu = (
                "gpu_memory" in decorator_self.metrics and torch.cuda.is_available()
            )

            # --- 准备指标收集 ---
            if use_time:
                start_time = time.perf_counter()

            if use_gpu:
                # 遍历所有设备，重置峰值统计并记录初始显存
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device=i)
                    initial_gpu_mems.append(torch.cuda.memory_allocated(device=i))

            # --- 执行原始函数 ---
            result = func(self, *args, **kwargs)

            # --- 收集并打印指标 ---
            print(f"\n--- 📊 Metrics for '{func.__name__}' ---")

            if use_time and start_time is not None:
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                print(f"⏱️  Execution Time: {elapsed:.4f} seconds")

            if use_gpu:
                peak_gpu_mems = []
                # 遍历所有设备，获取峰值显存
                for i in range(torch.cuda.device_count()):
                    peak_gpu_mems.append(torch.cuda.max_memory_allocated(device=i))

                total_initial_mem = sum(initial_gpu_mems)
                total_peak_mem = sum(peak_gpu_mems)
                gpu_mem_consumed = total_peak_mem - total_initial_mem

                print(
                    f"💾  Total GPU Memory Consumed (Delta): {gpu_mem_consumed / 1024**2:.2f} MB"
                )
                print(
                    f"    (Initial Total: {total_initial_mem / 1024**2:.2f} MB -> Peak Total: {total_peak_mem / 1024**2:.2f} MB)"
                )

                if torch.cuda.device_count() > 1:
                    print("    Per-GPU Breakdown (Initial -> Peak) MB:")
                    for i in range(torch.cuda.device_count()):
                        initial_mb = initial_gpu_mems[i] / 1024**2
                        peak_mb = peak_gpu_mems[i] / 1024**2
                        delta_mb = peak_mb - initial_mb
                        print(
                            f"      GPU {i}: {initial_mb:.2f} -> {peak_mb:.2f} (Delta: {delta_mb:+.2f})"
                        )

            print("---------------------------------")

            return result

        return wrapper

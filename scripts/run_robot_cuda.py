import torch
import time
import signal
import sys
import argparse  # 引入参数解析模块


def print_gpu_memory(device_id=0):
    """打印指定GPU的显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        print(
            f"GPU {device_id} 显存占用: 已分配 {allocated:.2f} GB | 已保留 {reserved:.2f} GB"
        )
        return allocated
    else:
        print("没有可用的CUDA设备")
        return 0


def signal_handler(sig, frame):
    """处理 Ctrl+C 退出"""
    print("\n\n程序被用户中断，正在退出...")
    sys.exit(0)


def occupy_gpu_memory(max_gb_target=None, device_id=0):
    """
    逐渐占用GPU显存，直到达到指定目标或耗尽内存。

    Args:
        max_gb_target (float, optional): 最大尝试占用的显存（GB）。如果为None，则会一直分配直到内存耗尽。Defaults to None.
        device_id (int): 要操作的GPU设备ID。
    """
    device = f"cuda:{device_id}"
    if not torch.cuda.is_available():
        print("错误: 没有可用的CUDA设备")
        return

    print(f"开始占用 {device} 显存")
    print(f"设备名称: {torch.cuda.get_device_name(device_id)}")
    if max_gb_target is not None:
        print(f"目标显存占用: {max_gb_target:.2f} GB")
    else:
        print("目标: 耗尽所有可用显存")

    print("\n初始显存状态:")
    print_gpu_memory(device_id)

    # 用于存储创建的张量，防止被垃圾回收
    tensors = []

    # 设置张量大小 (MB)
    chunk_size_mb = 256  # 每次分配256MB
    fail_attempts = 0  # 记录失败尝试次数

    try:
        while True:
            try:
                # 创建指定大小的张量 (单精度浮点数，每个占4字节)
                tensor_size_elements = chunk_size_mb * 1024 * 1024 // 4
                new_tensor = torch.rand(
                    tensor_size_elements, dtype=torch.float32, device=device
                )
                tensors.append(new_tensor)

                # 显示当前显存使用情况
                allocated_gb = print_gpu_memory(device_id)

                # 如果之前有失败，重置计数器并通知恢复
                if fail_attempts > 0:
                    print(f"恢复成功分配! 之前已连续失败 {fail_attempts} 次")
                    fail_attempts = 0

                # 检查是否达到目标
                if max_gb_target is not None and allocated_gb >= max_gb_target:
                    print(f"\n已达到或超过目标显存 {max_gb_target:.2f} GB。停止分配。")
                    while True:
                        time.sleep(100)
                    continue
                    break  # 退出循环

            except torch.cuda.OutOfMemoryError:
                fail_attempts += 1
                print(f"显存已耗尽，无法分配，这是第 {fail_attempts} 次失败尝试")
                # 如果是奔着耗尽内存去的，失败一次就可以停了
                if max_gb_target is None:
                    print("已无法分配更多显存，程序结束。")
                    break

            except Exception as e:
                print(f"发生未知错误: {e}")
                break

            finally:
                # 短暂休眠，便于观察和减轻系统负担
                time.sleep(0.5)
    finally:
        print("\n" + "=" * 20)
        print("最终显存状态:")
        print_gpu_memory(device_id)
        print("程序结束。")
        print("=" * 20)


def main():
    """主函数，用于解析参数和启动程序"""
    # 注册信号处理器，以便可以用Ctrl+C优雅地退出
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(
        description="逐渐占用GPU显存，可设置最大占用目标。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=20,
        help="最大尝试占用的GPU显存（单位：GB）。\n如果不设置此参数，程序将持续分配直到显存耗尽。",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="指定要操作的GPU设备ID (例如: 0, 1, ...)。默认为0。",
    )

    args = parser.parse_args()

    # 注意：原始代码中的 "CUDA:2" 可能是个笔误，实际代码用的是 "cuda:0"
    # 这里我们使用 --device 参数来控制，默认为0
    occupy_gpu_memory(max_gb_target=args.max_gb, device_id=args.device)


if __name__ == "__main__":
    main()

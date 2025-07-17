# run_brain_system.py

import time
import traceback
import multiprocessing
import matplotlib
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="../robot_brain_system/conf", config_name='config')
def main(cfg: DictConfig):
    """
    Main execution function for the Robot Brain System.
    This function initializes and runs the entire application.
    """

    # 1. 导入你的系统和配置
    # 假设 run_brain_system.py 在项目根目录，并且 src 在 python path 中
    # 如果不在，你可能需要先 `import sys; sys.path.append('src')`
    from robot_brain_system.core.system import RobotBrainSystem
    from robot_brain_system.utils.logging_utils import setup_logging
    
    print(OmegaConf.to_yaml(cfg))
    
    logger = setup_logging(
        log_level="INFO", log_file=f"{cfg['monitoring']['log_file']}"
    )

    print("--- TEST ROBOT BRAIN SYSTEM ---")

    # 2. 创建系统实例
    system = RobotBrainSystem(cfg)

    # 3. 初始化系统
    print("\n--- Initializing System Components... ---")
    if not system.initialize():
        print("\nFATAL: System initialization failed. Exiting.")
        return

    # 4. 启动系统的主循环
    print("\n--- Starting System Main Loop... ---")
    if not system.start():
        print("\nFATAL: System failed to start. Exiting.")
        system.shutdown()
        return

    # 5. 执行一个高级任务指令
    print("\n--- System is running, executing high-level task... ---")
    task_instruction = "grasp the spanner in the red box, home position is [1.1283, -3.8319,  3.6731, -0.6167,  0.3308, -0.3199, -0.6386]"
    task_instruction = "move to the red box"
    if not system.execute_task(task_instruction):
        print(
            f"\nFailed to start task: '{task_instruction}'. System might be busy or an error occurred."
        )
        # 根据情况决定是否需要关闭系统
        # system.shutdown()
        # return

    # 6. 监控系统状态，直到任务完成或被中断
    print("\n--- Monitoring task execution... (Press Ctrl+C to interrupt) ---")
    try:
        while system.state.is_running:
            time.sleep(2)  # 降低打印频率，让日志更清晰
            status = system.get_status()

            system_op = status.get("system", {}).get("status", "unknown")
            brain_op = status.get("brain", {}).get("status", "unknown")
            sim_skill_op = (
                status.get("simulator", {})
                .get("skill_executor", {})
                .get("status", "unknown")
            )

            print(
                f"STATUS | System: {system_op} | Brain: {brain_op} | Sim Skill: {sim_skill_op}"
            )

            # 改进的退出条件：当系统和大脑都空闲时，任务才算完成
            if system_op == "idle" and brain_op == "idle":
                print("\n--- Task completed: System and Brain are both idle. ---")
                break

            # 增加一个错误状态的退出条件
            if "error" in system_op.lower() or "error" in brain_op.lower():
                error_msg = status.get("system", {}).get("error_message") or status.get(
                    "brain", {}
                ).get("error_message")
                print(
                    f"\n--- System entered ERROR state: {error_msg}. Shutting down. ---"
                )
                break

    except KeyboardInterrupt:
        print("\n--- Keyboard interrupt received. Shutting down system... ---")
    except Exception as e:
        print(
            f"\n--- An unexpected error occurred in the main monitoring loop: {e} ---"
        )
        traceback.print_exc()
    finally:
        # 7. 确保系统被优雅地关闭
        print("\n--- Initiating graceful shutdown... ---")
        system.interrupt_task("Test run finished or was interrupted.")
        system.shutdown()
        print("\n--- Robot Brain System test completed. ---")


# --- Python 程序入口点 ---
if __name__ == "__main__":
    # 在这里进行所有一次性的、必须在任何其他操作之前完成的全局配置

    # 配置 Matplotlib 后端，避免 GUI 问题
    matplotlib.use("Agg")

    # 配置多进程启动方法。这对于 macOS 和 Windows 是必需的，
    # 并且对于所有平台上的 Isaac Sim 都是推荐的最佳实践。
    # 必须在创建任何进程或使用任何 multiprocessing 功能之前调用。
    try:
        multiprocessing.set_start_method("spawn", force=True)
        print("[Launcher] Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        # 如果已经设置，这可能会抛出 RuntimeError，可以安全地忽略
        print("[Launcher] Multiprocessing start method was already set.")

    main()

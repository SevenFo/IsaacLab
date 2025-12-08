# ui_run_robot_brain_system.py

import time
import traceback
import multiprocessing
import matplotlib
from omegaconf import DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra  # [新增]

# 导入全局控制台
from robot_brain_system.ui.console import global_console


def backend_worker(cfg: DictConfig):
    """
    后台业务逻辑线程。
    """
    # 延迟导入，避免在主线程导入过重资源
    from robot_brain_system.core.system import RobotBrainSystem
    from robot_brain_system.utils.logging_utils import setup_logging

    # 1. 设置文件日志
    # 注意：ConsoleUI 已经接管了 stdout/stderr，这里的 logger 主要用于写文件
    # 我们不需要它打印到控制台，否则会和 UI 冲突
    logger = setup_logging(
        log_level="INFO",
        log_file=cfg["monitoring"]["log_file"],
        redirect_print=False,  # 确保不重复接管 print
    )

    global_console.log("system", "--- ROBOT BRAIN SYSTEM STARTED ---")
    global_console.log("info", f"Log file: {cfg['monitoring']['log_file']}")

    # 2. 创建系统实例
    try:
        system = RobotBrainSystem(cfg)
    except Exception as e:
        global_console.log("error", f"System instantiation failed: {e}")
        traceback.print_exc()
        return

    # Ensure UI exit triggers system shutdown to stop subprocesses
    def _on_ui_exit():
        try:
            if system.state.is_running:
                global_console.log("system", "UI exit: interrupting running task...")
                try:
                    system.interrupt_task("UI exit")
                except Exception as e_int:
                    global_console.log("error", f"Interrupt on UI exit failed: {e_int}")
            global_console.log("system", "UI exit: shutting down system...")
            system.shutdown()
        except Exception as e:
            global_console.log("error", f"UI exit shutdown failed: {e}")

    global_console.set_shutdown_callback(_on_ui_exit)

    # 3. 初始化系统
    global_console.log("system", "Initializing System Components...")
    if not system.initialize():
        global_console.log("error", "FATAL: System initialization failed. Exiting.")
        return

    # 4. 启动系统主循环
    global_console.log("system", "Starting System Main Loop...")
    if not system.start():
        global_console.log("error", "FATAL: System failed to start. Exiting.")
        system.shutdown()
        return

    # 5. 执行任务
    task_instruction = "先调整箱子的位置和方向，然后 grasp the spanner in the red box, then move the spanner to the white hand palm, and release it there."
    # task_instruction = "move to the red box"

    global_console.log("brain", f"Executing Task: {task_instruction}")

    if not system.execute_task(task_instruction):
        global_console.log("error", f"Failed to start task: {task_instruction}")

    # 6. 监控循环
    global_console.log(
        "info", "Monitoring task execution... (Press Ctrl+C or type /exit to quit)"
    )

    try:
        running_times = 30
        success_times = 0
        failed_times = 0

        while system.state.is_running:
            time.sleep(2)

            status = system.get_status()
            system_op = status.get("system", {}).get("status", "unknown")
            brain_op = status.get("brain", {}).get("status", "unknown")
            sim_skill_op = (
                status.get("simulator", {})
                .get("skill_executor", {})
                .get("status", "unknown")
            )

            # # 在 UI 中打印状态
            # global_console.log(
            #     "info",
            #     f"STATUS | Sys: {system_op} | Brain: {brain_op} | Skill: {sim_skill_op}",
            # )

            # 成功重启逻辑
            if system_op == "idle" and brain_op == "idle":
                global_console.log(
                    "success", "Task completed: System and Brain are both idle."
                )
                success_times += 1

                if success_times + failed_times >= running_times:
                    global_console.log("system", "Target run count reached.")
                    break

                global_console.log(
                    "brain",
                    f"Restarting Task (Run {success_times + failed_times + 1})...",
                )
                system.reset()
                if not system.execute_task(task_instruction):
                    global_console.log("error", "Failed to restart task.")

            # 错误重启逻辑
            if "error" in str(system_op).lower() or "error" in str(brain_op).lower():
                error_msg = status.get("system", {}).get("error_message") or status.get(
                    "brain", {}
                ).get("error_message")
                global_console.log("error", f"System entered ERROR state: {error_msg}")
                failed_times += 1

                if success_times + failed_times >= running_times:
                    break

                global_console.log("brain", "System reset after error. Retrying...")
                system.reset()
                if not system.execute_task(task_instruction):
                    global_console.log("error", "Failed to retry task.")

    except Exception as e:
        global_console.log("error", f"Unexpected error in backend loop: {e}")
        traceback.print_exc()
    finally:
        global_console.log("system", "Initiating graceful shutdown...")
        try:
            if system.state.is_running:
                system.interrupt_task("Test run finished.")
        except Exception as e:
            global_console.log("error", f"Interrupt during shutdown failed: {e}")
        finally:
            try:
                system.shutdown()
            except Exception as e:
                global_console.log("error", f"System shutdown failed: {e}")
        global_console.log("system", "Backend worker finished.")
        # 可选：任务结束后通知 UI 退出，或者保持 UI 开启查看日志
        # global_console.stop()


@hydra.main(
    version_base=None, config_path="../robot_brain_system/conf", config_name="config"
)
def main(cfg: DictConfig):
    """
    Main entry point.
    Runs the UI on the main thread and the Logic on a background thread.
    """
    try:
        # 将 cfg 传给后台任务
        # run() 会阻塞主线程，直到 UI 退出
        global_console.run(lambda: backend_worker(cfg))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    # 1. 配置 Matplotlib
    matplotlib.use("Agg")

    # 2. 配置多进程
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    GlobalHydra.instance().clear()

    # 4. 运行
    main()

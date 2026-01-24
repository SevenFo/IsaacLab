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

# 导入状态枚举用于比较
from robot_brain_system.core.types import SystemStatus


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
    task_instruction = "请将桌面的红色工具箱旋转至按钮朝向机械臂，然后按下黄色按钮打开工具箱，从工具箱中取出黄色扳手并递到手掌上。"
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

            # 直接从 SystemState 获取状态枚举，而非从 get_status() 字符串判断
            system_status = system.state.status  # SystemStatus 枚举
            status_dict = system.get_status()

            # Brain 状态：从 status 字典获取布尔标志
            brain_has_task = status_dict.get("brain", {}).get("has_task", False)
            brain_has_pending_skills = status_dict.get("brain", {}).get(
                "has_pending_skills", False
            )

            # Skill Executor 状态
            skill_executor_status = (
                status_dict.get("simulator", {})
                .get("skill_executor", {})
                .get("status", "unknown")
            )

            # 调试日志（可选）
            # global_console.log(
            #     "info",
            #     f"STATUS | Sys: {system_status.name} | Brain has_task: {brain_has_task} | Skill: {skill_executor_status}",
            # )

            # === 成功完成判断 ===
            # 系统处于 IDLE 且 Brain 没有待执行任务
            if system_status == SystemStatus.IDLE and not brain_has_task:
                global_console.log(
                    "success", "Task completed: System is IDLE and Brain has no task."
                )
                success_times += 1

                if success_times + failed_times >= running_times:
                    global_console.log("system", "Target run count reached.")
                    break

                global_console.log(
                    "brain",
                    f"Restarting Task (Run {success_times, failed_times + 1}/{success_times}:{failed_times}/{running_times})...",
                )
                if not system.reset():
                    global_console.log("error", "System reset failed.")
                    failed_times += 1
                    continue

                if not system.execute_task(task_instruction):
                    global_console.log("error", "Failed to restart task.")
                    failed_times += 1

            # === 错误处理 ===

            elif system_status == SystemStatus.ERROR:
                error_msg = system.state.error_message or "Unknown error"
                global_console.log("error", f"System entered ERROR state: {error_msg}")
                failed_times += 1

                if success_times + failed_times >= running_times:
                    global_console.log(
                        "system", "Target run count reached (with errors)."
                    )
                    break

                global_console.log(
                    "brain",
                    f"Resetting after error (Attempt {success_times + failed_times}/{running_times})...",
                )
                if not system.reset():
                    global_console.log("error", "System reset failed after error.")
                    continue

                if not system.execute_task(task_instruction):
                    global_console.log("error", "Failed to retry task after error.")

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

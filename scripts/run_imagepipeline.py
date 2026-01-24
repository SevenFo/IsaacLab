#!/usr/bin/env python3
"""
ImagePipeline 测试脚本
用于测试 ImagePipeline.initialize_with_instruction 方法

功能：
- 启动仿真环境获取相机输入
- 提供交互式终端 UI 进行测试
- 支持切换相机、重置环境、切换场景模式等

用法：
    ./isaaclab.sh -p scripts/run_imagepipeline.py
"""

import time
import traceback
import multiprocessing
import matplotlib
from omegaconf import DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
import torch
import numpy as np
from PIL import Image

from robot_brain_system.ui.console import global_console
from robot_brain_system.skills.imagepipleline import ImagePipeline
from robot_brain_system.core.model_adapters_v2 import OpenAIAdapter
from robot_brain_system.utils.config_utils import hydra_context_base
from cutie.utils.get_default_model import get_default_model


def get_default_cutie_model():
    """加载 Cutie 默认模型"""
    with hydra_context_base():
        return get_default_model()


class ImagePipelineTester:
    """ImagePipeline 测试器"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.simulator = None
        self.env_proxy = None
        self.pipeline: ImagePipeline | None = None
        self.available_cameras: list[str] = []
        self.current_camera: str = ""
        self.device = "cuda:0"
        self.vlm = None
        self.cutie_model = None
        self.is_running = False

        # SAM2 配置
        self.sam_checkpoint = "thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt"
        self.sam_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

        # 场景模式
        self.scene_modes = ["default", "random"]
        self.current_scene_mode_idx = 0

    def initialize(self) -> bool:
        """初始化测试环境"""
        try:
            global_console.log("system", "[Tester] Initializing simulator...")

            # 初始化 Isaac 仿真器
            from robot_brain_system.core.isaac_simulator import IsaacSimulator
            from robot_brain_system.core.env_proxy import create_env_proxy
            from omegaconf import OmegaConf

            # 转换为普通 dict 以避免 OmegaConf struct 模式限制
            sim_config_dict = OmegaConf.to_container(
                self.cfg["simulator"], resolve=True
            )
            # 使用视觉测试环境
            sim_config_dict["task"] = "Isaac-Move-Box-UR5-IK-Rel-rot-0923-visual-test"
            # ImagePipeline 测试不需要 skills 配置
            # 但 IsaacSimulator 可能期望这个字段存在
            if "skills" not in sim_config_dict:
                sim_config_dict["skills"] = {}

            self.simulator = IsaacSimulator(sim_config=sim_config_dict)

            if not self.simulator.initialize():
                global_console.log("error", "[Tester] Failed to initialize simulator")
                return False

            global_console.log("system", "[Tester] Simulator initialized successfully")

            # 创建环境代理
            self.env_proxy = create_env_proxy(self.simulator, scene_mode="default")
            global_console.log("system", "[Tester] EnvProxy created successfully")

            # 获取可用相机列表
            obs = self.env_proxy.update(return_obs=True)
            if obs is None:
                global_console.log(
                    "error", "[Tester] Failed to get initial observation"
                )
                return False

            obs_dict = obs.data.get("policy", {})
            self.available_cameras = [
                k
                for k in obs_dict.keys()
                if k.startswith("camera_") and isinstance(obs_dict[k], torch.Tensor)
            ]

            if not self.available_cameras:
                global_console.log("error", "[Tester] No cameras found in observation")
                return False

            self.current_camera = self.available_cameras[0]
            global_console.log(
                "system", f"[Tester] Available cameras: {self.available_cameras}"
            )
            global_console.log(
                "system", f"[Tester] Current camera: {self.current_camera}"
            )

            # 初始化 VLM 适配器
            global_console.log("system", "[Tester] Initializing VLM adapter...")
            self.vlm = OpenAIAdapter(
                "qwen2.5-vl-32b-awq",
                api_key="123",
                base_url="http://127.0.0.1:8000/v1",
            )
            global_console.log("system", "[Tester] VLM adapter initialized")

            # 加载 Cutie 模型
            global_console.log("system", "[Tester] Loading Cutie model...")
            self.cutie_model = get_default_cutie_model()
            global_console.log("system", "[Tester] Cutie model loaded")

            self.is_running = True
            return True

        except Exception as e:
            global_console.log("error", f"[Tester] Initialization failed: {e}")
            traceback.print_exc()
            return False

    def create_pipeline(self) -> ImagePipeline:
        """创建新的 ImagePipeline 实例"""
        return ImagePipeline(
            self.device,
            self.sam_checkpoint,
            self.sam_model_config,
            self.cutie_model,
            self.vlm,
        )

    def test_initialize_with_instruction(
        self, instruction: str, camera_name: str | None = None, visualize: bool = True
    ) -> bool:
        """
        测试 ImagePipeline.initialize_with_instruction 方法

        Args:
            instruction: 目标对象描述
            camera_name: 相机名称（None 则使用当前相机）
            visualize: 是否可视化
        """
        camera = camera_name or self.current_camera

        if camera not in self.available_cameras:
            global_console.log(
                "error",
                f"[Tester] Camera '{camera}' not found. Available: {self.available_cameras}",
            )
            return False

        global_console.log("skill", "[Tester] Testing initialize_with_instruction:")
        global_console.log("skill", f"  - Target: {instruction}")
        global_console.log("skill", f"  - Camera: {camera}")
        global_console.log("skill", f"  - Visualize: {visualize}")

        try:
            # 获取当前相机图像
            obs = self.env_proxy.update(return_obs=True)
            if obs is None:
                global_console.log("error", "[Tester] Failed to get observation")
                return False

            obs_dict = obs.data.get("policy", {})
            frame_tensor = obs_dict.get(camera)

            if frame_tensor is None:
                global_console.log("error", f"[Tester] No data for camera: {camera}")
                return False

            frame = frame_tensor[0].cpu().numpy()

            # 保存输入图像
            timestamp = int(time.time() * 1000)
            input_path = f"test_input_{camera}_{timestamp}.png"
            Image.fromarray(frame).save(input_path)
            global_console.log("info", f"[Tester] Input image saved: {input_path}")

            # 创建新的 pipeline 实例
            global_console.log(
                "system", "[Tester] Creating new ImagePipeline instance..."
            )
            self.pipeline = self.create_pipeline()

            # 执行测试
            global_console.log(
                "system", "[Tester] Calling initialize_with_instruction..."
            )
            start_time = time.time()

            mask, bboxes = self.pipeline.initialize_with_instruction(
                frame, instruction, return_bbox=True, visualize=visualize
            )

            elapsed = time.time() - start_time
            global_console.log(
                "skill",
                f"[Tester] initialize_with_instruction completed in {elapsed:.2f}s",
            )

            # 报告结果
            if mask is not None:
                unique_ids = np.unique(mask)
                global_console.log(
                    "skill",
                    f"[Tester] Result: Found {len(unique_ids) - 1} objects (IDs: {unique_ids[unique_ids > 0].tolist()})",
                )
                global_console.log("skill", f"[Tester] Bounding boxes: {bboxes}")
            else:
                global_console.log("skill", "[Tester] Result: No mask generated")

            return True

        except ValueError as e:
            global_console.log("error", f"[Tester] Target not found: {e}")
            return False
        except Exception as e:
            global_console.log("error", f"[Tester] Test failed: {e}")
            traceback.print_exc()
            return False

    def test_update_masks(self, visualize: bool = True) -> bool:
        """测试 update_masks 方法（需要先调用 initialize_with_instruction）"""
        if self.pipeline is None or not self.pipeline.is_initialized:
            global_console.log(
                "error", "[Tester] Pipeline not initialized. Run 'test <target>' first."
            )
            return False

        try:
            obs = self.env_proxy.update(return_obs=True)
            if obs is None:
                global_console.log("error", "[Tester] Failed to get observation")
                return False

            obs_dict = obs.data.get("policy", {})
            frame_tensor = obs_dict.get(self.current_camera)

            if frame_tensor is None:
                global_console.log(
                    "error", f"[Tester] No data for camera: {self.current_camera}"
                )
                return False

            frame = frame_tensor[0].cpu().numpy()

            global_console.log("system", "[Tester] Calling update_masks...")
            start_time = time.time()

            all_mask, mask_list = self.pipeline.update_masks(frame, visualize=visualize)

            elapsed = time.time() - start_time
            global_console.log(
                "skill", f"[Tester] update_masks completed in {elapsed:.3f}s"
            )

            if all_mask is not None and all_mask.any():
                unique_ids = np.unique(all_mask)
                global_console.log(
                    "skill", f"[Tester] Tracking result: {len(unique_ids) - 1} objects"
                )
            else:
                global_console.log("skill", "[Tester] Tracking result: Lost target")

            return True

        except Exception as e:
            global_console.log("error", f"[Tester] update_masks failed: {e}")
            traceback.print_exc()
            return False

    def switch_camera(self, camera_name: str) -> bool:
        """切换相机"""
        if camera_name not in self.available_cameras:
            global_console.log(
                "error",
                f"[Tester] Camera '{camera_name}' not found. Available: {self.available_cameras}",
            )
            return False

        self.current_camera = camera_name
        global_console.log("system", f"[Tester] Switched to camera: {camera_name}")
        return True

    def list_cameras(self):
        """列出所有可用相机"""
        global_console.log("info", "[Tester] Available cameras:")
        for i, cam in enumerate(self.available_cameras):
            marker = " (current)" if cam == self.current_camera else ""
            global_console.log("info", f"  [{i}] {cam}{marker}")

    def reset_environment(self):
        """重置仿真环境"""
        global_console.log("system", "[Tester] Resetting environment...")
        try:
            self.env_proxy.reset()
            # 清理 pipeline 状态
            if self.pipeline:
                self.pipeline.cleanup()
                self.pipeline = None
            global_console.log("system", "[Tester] Environment reset complete")
        except Exception as e:
            global_console.log("error", f"[Tester] Reset failed: {e}")

    def reset_box_and_spanner(self, mode: str = "normal"):
        """重置箱子和扳手位置"""
        global_console.log(
            "system", f"[Tester] Resetting box/spanner to '{mode}' mode..."
        )
        try:
            self.env_proxy.reset_box_and_spanner(mode)
            global_console.log("system", f"[Tester] Box/spanner reset to '{mode}' mode")
        except Exception as e:
            global_console.log("error", f"[Tester] Reset box/spanner failed: {e}")

    def capture_frame(self, camera_name: str | None = None):
        """保存当前帧"""
        camera = camera_name or self.current_camera
        try:
            obs = self.env_proxy.update(return_obs=True)
            if obs is None:
                global_console.log("error", "[Tester] Failed to get observation")
                return

            obs_dict = obs.data.get("policy", {})
            frame_tensor = obs_dict.get(camera)

            if frame_tensor is None:
                global_console.log("error", f"[Tester] No data for camera: {camera}")
                return

            frame = frame_tensor[0].cpu().numpy()
            timestamp = int(time.time() * 1000)
            path = f"capture_{camera}_{timestamp}.png"
            Image.fromarray(frame).save(path)
            global_console.log("info", f"[Tester] Frame saved: {path}")
        except Exception as e:
            global_console.log("error", f"[Tester] Capture failed: {e}")

    def shutdown(self):
        """关闭测试器"""
        global_console.log("system", "[Tester] Shutting down...")
        self.is_running = False

        if self.pipeline:
            try:
                self.pipeline.cleanup()
            except Exception:
                pass

        if self.simulator and self.simulator.is_initialized:
            self.simulator.shutdown()

        global_console.log("system", "[Tester] Shutdown complete")


def print_help():
    """打印帮助信息"""
    help_text = """
╔══════════════════════════════════════════════════════════════════╗
║             ImagePipeline 测试工具 - 命令帮助                      ║
╠══════════════════════════════════════════════════════════════════╣
║ 测试命令：                                                        ║
║   test <target>        测试 initialize_with_instruction            ║
║                        例：test red box / test spanner            ║
║   test <target> <cam>  指定相机测试                                ║
║                        例：test red box camera_left               ║
║   update               测试 update_masks (需先 test)              ║
║   track [N]            连续追踪 N 帧 (默认 10)                     ║
║                                                                   ║
║ 相机命令：                                                        ║
║   cameras              列出所有可用相机                            ║
║   cam <name>           切换当前相机                                ║
║   capture [cam]        保存当前帧图像                              ║
║                                                                   ║
║ 环境命令：                                                        ║
║   reset                重置整个环境                                ║
║   normal               重置 box/spanner 到正常位置                 ║
║   far                  移走 box/spanner (测试缺失场景)             ║
║                                                                   ║
║ 其他命令：                                                        ║
║   help, h, ?           显示此帮助                                  ║
║   status               显示当前状态                                ║
║   clear                清理 pipeline 状态                          ║
║   /exit, /quit         退出程序                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""
    global_console.log("info", help_text)


def backend_worker(cfg: DictConfig):
    """后台工作线程"""
    from robot_brain_system.utils.logging_utils import setup_logging

    setup_logging(
        log_level="INFO",
        log_file=cfg["monitoring"]["log_file"],
        redirect_print=False,
    )

    global_console.log("system", "=== ImagePipeline 测试工具启动 ===")

    tester = ImagePipelineTester(cfg)

    # 设置 UI 退出回调
    def on_ui_exit():
        try:
            tester.shutdown()
        except Exception as e:
            global_console.log("error", f"Shutdown error: {e}")

    global_console.set_shutdown_callback(on_ui_exit)

    # 初始化
    if not tester.initialize():
        global_console.log("error", "Initialization failed. Exiting.")
        return

    print_help()
    global_console.log("info", "Ready for commands. Type 'help' for usage.")

    # 主循环 - 处理用户输入
    while tester.is_running:
        try:
            # 非阻塞检查输入队列
            try:
                cmd = global_console.input_queue.get(timeout=0.5)
            except Exception:
                continue

            if not cmd:
                continue

            parts = cmd.strip().split()
            if not parts:
                continue

            command = parts[0].lower()
            args = parts[1:]

            # 退出命令
            if command in ["/exit", "/quit", "exit", "quit"]:
                global_console.log("system", "Exiting...")
                break

            # 帮助命令
            elif command in ["help", "h", "?"]:
                print_help()

            # 测试 initialize_with_instruction
            elif command == "test":
                if not args:
                    global_console.log("error", "Usage: test <target> [camera]")
                    continue

                # 解析参数：可能是 "test red box" 或 "test red box camera_left"
                # 检查最后一个参数是否是相机名
                camera = None
                target_parts = args

                if args[-1] in tester.available_cameras:
                    camera = args[-1]
                    target_parts = args[:-1]

                target = " ".join(target_parts)
                if not target:
                    global_console.log("error", "Usage: test <target> [camera]")
                    continue

                tester.test_initialize_with_instruction(target, camera, visualize=True)

            # 测试 update_masks
            elif command == "update":
                tester.test_update_masks(visualize=True)

            # 连续追踪
            elif command == "track":
                n_frames = int(args[0]) if args else 10
                global_console.log(
                    "system", f"[Tester] Tracking for {n_frames} frames..."
                )
                for i in range(n_frames):
                    if not tester.test_update_masks(visualize=(i % 5 == 0)):
                        break
                    time.sleep(0.1)
                global_console.log("system", "[Tester] Tracking complete")

            # 列出相机
            elif command in ["cameras", "cams"]:
                tester.list_cameras()

            # 切换相机
            elif command == "cam":
                if not args:
                    global_console.log("error", "Usage: cam <camera_name>")
                    tester.list_cameras()
                    continue
                tester.switch_camera(args[0])

            # 保存帧
            elif command == "capture":
                camera = args[0] if args else None
                tester.capture_frame(camera)

            # 重置环境
            elif command == "reset":
                tester.reset_environment()

            # 重置 box/spanner 位置
            elif command == "normal":
                tester.reset_box_and_spanner("normal")

            elif command == "far":
                tester.reset_box_and_spanner("far")

            # 状态
            elif command == "status":
                global_console.log(
                    "info", f"[Status] Current camera: {tester.current_camera}"
                )
                global_console.log(
                    "info", f"[Status] Available cameras: {tester.available_cameras}"
                )
                global_console.log(
                    "info",
                    f"[Status] Pipeline initialized: {tester.pipeline is not None and tester.pipeline.is_initialized}",
                )
                if tester.pipeline and tester.pipeline.is_initialized:
                    global_console.log(
                        "info",
                        f"[Status] Tracking instruction: {tester.pipeline.instruction}",
                    )
                    global_console.log(
                        "info",
                        f"[Status] Tracked objects: {tester.pipeline.current_objects}",
                    )

            # 清理 pipeline
            elif command == "clear":
                if tester.pipeline:
                    tester.pipeline.cleanup()
                    tester.pipeline = None
                    global_console.log("system", "[Tester] Pipeline cleared")
                else:
                    global_console.log("info", "[Tester] No pipeline to clear")

            else:
                global_console.log(
                    "error", f"Unknown command: {command}. Type 'help' for usage."
                )

        except KeyboardInterrupt:
            global_console.log("system", "Interrupted. Exiting...")
            break
        except Exception as e:
            global_console.log("error", f"Command error: {e}")
            traceback.print_exc()

    # 清理
    tester.shutdown()
    global_console.log("system", "Backend worker finished.")


@hydra.main(
    version_base=None, config_path="../robot_brain_system/conf", config_name="config"
)
def main(cfg: DictConfig):
    """主入口"""
    try:
        global_console.run(lambda: backend_worker(cfg))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    matplotlib.use("Agg")

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    GlobalHydra.instance().clear()
    main()

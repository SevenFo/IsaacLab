# mocap_sender.py (Corrected Version)
# ----------------------------------------------------------------
# 已修正，不再使用不存在的 MocapBone 类。
# 直接使用 mocap_api.py 中定义的 MCPAvatar 和 MCPJoint 类来提取数据。
# ----------------------------------------------------------------

import time
import json
import socket
import traceback


# 从 mocap_api 导入实际存在的类
from mocap_api import (
    MCPApplication,
    MCPSettings,
    MCPBvhRotation,
    MCPEventType,
    MCPAvatar,  # 虽然不直接在参数中使用，但了解其存在是关键
)


def extract_bone_data_from_avatar(avatar: MCPAvatar) -> dict:
    """
    直接从 MCPAvatar 对象中提取所有关节（骨骼）的名称、
    局部位置和局部旋转，并构建一个字典。
    """
    bone_data_dict = {}

    # MCPAvatar 对象有一个 get_joints() 方法，返回 MCPJoint 对象列表
    joints = avatar.get_joints()

    for joint in joints:
        try:
            # MCPJoint 对象有我们需要的所有方法
            joint_name = joint.get_name()
            local_pos = joint.get_local_position()
            local_rot = joint.get_local_rotation()  # (w, x, y, z)

            # 检查 get_local_position 是否返回了有效数据
            if local_pos[0] is not None:
                bone_data_dict[joint_name] = {
                    "local_position": list(local_pos),
                    "local_rotation": list(local_rot),
                }
        except Exception as e:
            # 在处理单个关节时可能会出现错误，打印并继续
            print(f"Warning: Could not process joint. Error: {e}")

    return bone_data_dict


class MotionCaptureSender:
    """一个简单的TCP客户端，用于将数据发送给Isaac Sim。(代码保持不变)"""

    def __init__(self, host="127.0.0.1", port=12345):
        self.host = host
        self.port = port
        self.sock = None
        self.is_connected = False

    def connect(self):
        if self.is_connected:
            return True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3.0)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(None)
            self.is_connected = True
            print(f"成功连接到 Isaac Sim -> {self.host}:{self.port}")
            return True
        except (ConnectionRefusedError, socket.timeout):
            print("连接失败。请确保 Isaac Sim 中的接收器正在运行。")
            self.close()
            return False
        except Exception as e:
            print(f"连接时发生未知错误: {e}")
            self.close()
            return False

    def send_data(self, data_dict: dict):
        if not self.is_connected:
            if not self.connect():
                time.sleep(2)
                return
        if not data_dict:
            return
        try:
            message = json.dumps(data_dict).encode("utf-8")
            self.sock.sendall(message + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            print("连接已断开。将在下次发送时尝试重连。")
            self.close()
        except Exception as e:
            print(f"发送数据失败: {e}")
            self.close()

    def close(self):
        if self.sock:
            self.sock.close()
        self.sock = None
        self.is_connected = False


def main():
    """主函数：初始化动捕，循环获取数据，并发送出去。"""
    print("Mocap 发送程序启动...")
    app = MCPApplication()
    settings = MCPSettings()
    settings.set_udp(7013)
    settings.set_bvh_rotation(MCPBvhRotation.XYZ)
    app.set_settings(settings)
    sender = MotionCaptureSender()

    try:
        # 您提供的API中，open()返回一个元组 (bool, str)
        is_ok, msg = app.open()
        if not is_ok:
            print(f"打开 Mocap 应用失败: {msg}")
            return
        print("动捕系统初始化成功。正在等待数据...")

        while True:
            evts = app.poll_next_event()
            if not evts:
                time.sleep(0.005)
                continue

            latest_avatar_evt = None
            for evt in evts:
                if evt.event_type == MCPEventType.AvatarUpdated:
                    latest_avatar_evt = evt

            if latest_avatar_evt:
                # 1. 从事件中获取 MCPAvatar 对象
                avatar = MCPAvatar(latest_avatar_evt.event_data.avatar_handle)

                # 2. 直接从 avatar 对象提取数据为字典
                bone_data_dict = extract_bone_data_from_avatar(avatar)
                # breakpoint()
                # 3. 发送字典
                sender.send_data(bone_data_dict)

    except KeyboardInterrupt:
        print("接收到手动停止信号 (Ctrl+C)。")
    except Exception as e:
        print(f"主循环发生未处理的错误: {e}")
        traceback.print_exc()
    finally:
        print("正在关闭程序...")
        app.close()
        sender.close()
        print("程序已退出。")


if __name__ == "__main__":
    main()

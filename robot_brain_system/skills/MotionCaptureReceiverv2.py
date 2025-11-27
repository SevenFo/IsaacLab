# MotionCaptureReceiver.py
# ----------------------------------------------------------------
# 这个类在 Isaac Sim 内部运行，创建一个异步TCP服务器。
# 它接收来自外部发送器的数据，并通过回调函数将最新的数据传递出去。
# 它的设计目标是低延迟，通过只处理最新的消息来避免数据积压。
# ----------------------------------------------------------------

import asyncio
import json
from ..utils.logging_utils import get_logger


class MotionCaptureReceiver:
    def __init__(self, data_handler_callback, host="127.0.0.1", port=12345):
        self.host = host
        self.port = port
        self.data_handler = data_handler_callback
        self._logger = get_logger("skills.motion_capture_receiver")
        self._logger.info("MotionCaptureReceiver initialized.")

    async def start_server(self):
        try:
            server = await asyncio.start_server(
                self._handle_client, self.host, self.port
            )
            self._logger.info(
                f"Mocap server started on {self.host}:{self.port}. Waiting for connection..."
            )
            async with server:
                await server.serve_forever()
        except OSError as e:
            self._logger.error(
                f"Failed to start server on port {self.port}. Is it already in use? Error: {e}"
            )
        except Exception as e:
            self._logger.error(f"An unexpected error occurred in start_server: {e}")

    async def _handle_client(self, reader, writer):
        addr = writer.get_extra_info("peername")
        self._logger.info(f"Client connected from {addr}")
        buffer = ""
        try:
            while True:
                data = await reader.read(8192)  # 读取一大块数据
                if not data:
                    break

                buffer += data.decode("utf-8")

                # 找出最后一个完整的消息以保证低延迟
                messages = buffer.split("\n")
                if len(messages) > 1:
                    # 保留最后一个可能不完整的消息片段在缓冲区
                    buffer = messages[-1]
                    # 在所有已收到的完整消息中，只处理最后一个
                    last_full_message = messages[-2]
                    if last_full_message:
                        try:
                            bone_data = json.loads(last_full_message)
                            if self.data_handler:
                                # 只把最新的数据通过回调函数传递出去
                                self.data_handler(bone_data)
                        except json.JSONDecodeError:
                            self._logger.warning(
                                "Received invalid JSON data, skipping."
                            )
                        except Exception as e:
                            self._logger.error(f"Error processing client data: {e}")
        finally:
            self._logger.info(f"Client {addr} disconnected.")
            writer.close()
            await writer.wait_closed()

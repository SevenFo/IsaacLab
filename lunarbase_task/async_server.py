# rpyc_server.py

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Async HTTP Server for Isaac Sim Environment"
)
parser.add_argument(
    "--mode", type=int, default=2,
    choices=[1, 2],
    help="Running mode: 1=Auto, 2=Manual (default: Manual)"
)
parser.add_argument(
    "--task", type=str, required=True,
    help="Name of the task/environment to create"
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--env_config_file", type=str, default=None,
    help="Env Config YAML file Path, use to update default env config"
)
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="Hostname to bind the server.",
)
parser.add_argument(
    "--port", type=int, default=18861, help="Port to bind the server."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments for the server's env instance.",
)
parser.add_argument(
    "--max_episode_length",
    type=int,
    default=250,
    help="Max steps per episode on server.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import numpy as np
import gymnasium as gym
import yaml
from functools import partial
from typing import Optional, Any, Dict, Tuple
from isaaclab_tasks.utils import parse_env_cfg
from aiohttp import web
import threading
from queue import Queue
import json
import time
import pyarrow as pa
import pickle

def dynamic_set_attr(object: object, kwargs: dict, path: list[str]):
    for k, v in kwargs.items():
        if k in object.__dict__:
            if isinstance(v, dict):
                next_path = path.copy()
                next_path.append(k)
                dynamic_set_attr(
                    object.__getattribute__(k), v, next_path
                )
            else:
                print(
                    f"set {'.'.join(path + [k])} from {object.__getattribute__(k)} to {v}"
                )
                object.__setattr__(k, v)


class SimulationManager:
    def __init__(self, args):
        self.args = args
        self.env = None
        self.mode = args.mode
        self.request_queue = Queue()
        self.result_queue = Queue()
        self.running = True
        self.current_action = None
        self._init_env()

    def _init_env(self):
        """Initialize environment in main thread"""
        env_cfg = parse_env_cfg(
            self.args.task,
            device="cuda:0",
            num_envs=self.args.num_envs,
            use_fabric=not self.args.disable_fabric
        )
        if self.args.env_config_file:
            with open(self.args.env_config_file, "r") as f:
                env_new_cfg = yaml.safe_load(f)
                dynamic_set_attr(env_cfg, env_new_cfg, path=["env_cfg"])
        self.env = gym.make(self.args.task, cfg=env_cfg)
        self.env.reset()

    def run(self):
        """Main simulation loop"""
        while self.running:
            # Handle incoming requests
            if not self.request_queue.empty():
                request = self.request_queue.get()
                if request['type'] == 'reset':
                    obs, info = self.env.reset()
                    obs = obs['policy']
                    self.result_queue.put({'obs': obs.cpu().numpy(), 'info': info})
                elif request['type'] == 'step':
                    action = request['action']
                    obs, reward, done, trunc, info = self.env.step(action)
                    obs = obs['policy']
                    self.result_queue.put({
                        'obs': obs.cpu().numpy(),
                        'reward': reward.cpu().numpy(),
                        'done': done.cpu().numpy(),
                        'trunc': trunc.cpu().numpy(),
                        'info': info
                    })
            
            # Auto-step logic
            if self.mode == 1:
                if self.current_action is not None:
                    self.env.step(self.current_action)
                    self.current_action = None
                else:
                    self.env.step(None)
            
            time.sleep(0.001)  # Adjust as needed

    def close(self):
        self.running = False
        if self.env:
            self.env.close()

async def handle_reset(request: web.Request) -> web.Response:
    sim_manager: SimulationManager = request.app['sim_manager']
    sim_manager.request_queue.put({'type': 'reset'})
    while True:
        if not sim_manager.result_queue.empty():
            result = sim_manager.result_queue.get()
            return web.json_response({
                'obs': result['obs'].tolist(),
                'info': result['info']
            })
        await asyncio.sleep(0.01)

async def handle_step(request: web.Request) -> web.Response:
    sim_manager: SimulationManager = request.app['sim_manager']
    
    # 反序列化
    raw_bytes = await request.read()
    action = pickle.loads(raw_bytes)  # 自动还原numpy数组
    
    if sim_manager.mode == 1:
        sim_manager.current_action = action
        return web.json_response({'status': 'action_queued'})
    else:
        sim_manager.request_queue.put({
            'type': 'step',
            'action': action
        })

    while True:
        if not sim_manager.result_queue.empty():
            result = sim_manager.result_queue.get()
            
            # 序列化返回数据
            return web.Response(
                body=pickle.dumps(result['obs']),  # 自动处理numpy
                content_type='application/python-pickle'
            )
        await asyncio.sleep(0.01)

async def on_shutdown(app: web.Application) -> None:
    sim_manager: SimulationManager = app['sim_manager']
    sim_manager.close()

def run_web_server(sim_manager: SimulationManager, host: str, port: int):
    app = web.Application()
    app['sim_manager'] = sim_manager
    app.add_routes([
        web.post('/reset', handle_reset),
        web.post('/step', handle_step)
    ])
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host=host, port=port)
if __name__ == "__main__":
    # 主线程初始化必须保留
    sim_manager = SimulationManager(args)
    
    # 正确做法：Web服务器放子线程
    server_thread = threading.Thread(
        target=run_web_server,
        args=(sim_manager, args.host, args.port),
        daemon=True
    )
    server_thread.start()

    # 主线程运行模拟循环
    try:
        sim_manager.run()  # 这里包含_app.update()
    except KeyboardInterrupt:
        sim_manager.close()
    finally:
        server_thread.join()

# if __name__ == "__main__":
#     sim_manager = SimulationManager(args)
    
#     # 主线程运行web服务器
#     # 注意：此时需要将sim_manager.run()移到子线程
#     simulation_thread = threading.Thread(
#         target=sim_manager.run,
#         daemon=True
#     )
#     simulation_thread.start()
    
#     # 主线程运行web服务器
#     run_web_server(sim_manager, args.host, args.port)
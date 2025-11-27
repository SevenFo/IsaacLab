
# Robot Brain System 技术文档


## 项目定位与核心原理

本系统采用 **Server/Client 分离架构**，实现高性能、低延迟的具身智能闭环：

- **Server**（`isaac_lab_server_shm.py`）：纯 Isaac Sim 仿真进程  
  - 不加载任何大模型、技能或 Brain  
  - 通过 **共享内存**（Shared Memory）写入最新 observation  
  - 通过 **Unix Domain Socket** 接收控制命令  
  - 仅提供环境服务与 success 判断支持

- **Client**（`scripts/run_robot_brain_system.py`）：主控进程  
  - 运行 Qwen-VL 大模型（Brain）进行任务解析与技能规划  
  - 维护 SkillRegistry，所有技能在此注册并执行  
  - 通过 `EnvProxy` 提供本地化 env 接口，屏蔽通信细节  
  - 50Hz 主循环：观测 → 技能推理 → 动作下发 → 监控 → 重规划

**通信机制**  
- 观测：共享内存（~10MB，pickle 序列化）→ 零拷贝、高频率  
- 动作 & 命令：共享内存 + Unix Socket（轻量可靠）

## 环境准备（完整可执行）

```bash
# 1. 克隆仓库并切换分支
git clone https://github.com/SevenFo/IsaacLab.git
cd IsaacLab
git checkout dev_502_bg_sm_align_demo_v0

# 2. 安装 Isaac Sim 4.5（推荐 pip 方式）
参考 https://isaacsim.github.io/IsaacLab/v2.0.0/source/setup/installation/pip_installation.html#installingisaac-sim 
使用 pip 安装 isaacsim 4.5

# 3. 仓库依赖
./isaaclab.sh --install

# 4. thirdparty 子模块同步
git submodule update --init --recursive

# 5. 安装关键 thirdparty（必须）
pip install -e thirdparty/lerobot
pip install -e thirdparty/MocapApi
pip install -e thirdparty/robot-retargeting

# SAM2 + Cutie（object_tracking 技能必需）
cd thirdparty
git clone https://github.com/facebookresearch/sam2.git sam2
cd sam2 && pip install -e . && cd ..

git clone https://github.com/hkchengrex/Cutie.git Cutie
cd Cutie && pip install -e . && python cutie/utils/download_models.py && cd ../..

pip install -r ImagePipeline/requirements.txt
```

## 数字资产准备（必须步骤）

```bash
# 创建资产目录
mkdir -p env_assets

下载并解压环境资产（包含所有 URDF、场景、纹理等）env.zip

unzip env.zip -d env_assets
```

**重要：路径替换**  
编辑 `configs/ur5_lunar_base.yaml`（或其他使用的任务 config），将文件中所有出现的老路径：

```
/home/ps/Projects/isaac-lab-workspace/IsaacLabLatest/IsaacLab/
```

替换为你本地的实际路径，例如：

```
$(pwd)/env_assets/
```

保存后确保无残留旧路径。

## 启动方式（两种模式）

### 模式一：推荐（Client 自动启动 Server）—— 日常使用

```bash
python scripts/run_robot_brain_system.py
```

Client 会自动通过 `./isaaclab.sh -p ...` 启动 Server 子进程并完成连接。  
这是最简单、最稳定的方式，无需手动管理进程。

### 模式二：手动启动 Server + Client —— 仅用于调试 Server 本身

如果你需要单独调试 `isaac_lab_server_shm.py`（例如加 print、查看仿真窗口、修改 warmup 逻辑等），请严格按以下两步操作：

**终端 1：手动启动纯仿真 Server**
```bash
./isaaclab.sh -p robot_brain_system/launcher/isaac_lab_server_shm.py
```
Server 会创建 `/tmp/isaac.sock` 和共享内存 `isaac_shm`，然后卡在 `accept()` 等待 Client 连接。

**终端 2：启动 Client 并关闭自动拉起功能**
```bash
python scripts/run_robot_brain_system.py simulator.auto_start=false
```
或在配置文件中强制覆盖：
```bash
python scripts/run_robot_brain_system.py +simulator.auto_start=false
```

- 手动模式下 **必须** 加 `simulator.auto_start=false`  
- 否则会出现 “Address already in use” 或共享内存重复创建导致 Client 直接崩溃  
- 两个进程使用的 socket 路径和共享内存名前缀必须一致（默认都是 `/tmp/isaac.sock` 和 `isaac_shm`）


## 核心组件路径说明

| 组件            | 文件路径                                              |
| --------------- | ----------------------------------------------------- |
| 主入口          | `scripts/run_robot_brain_system.py`                   |
| 纯仿真 Server   | `robot_brain_system/launcher/isaac_lab_server_shm.py` |
| 系统状态机      | `robot_brain_system/core/system.py`                   |
| Brain 实现      | `robot_brain_system/core/brain.py`                    |
| 技能执行器      | `robot_brain_system/core/skill_executor_client.py`    |
| 本地化 Env 接口 | `robot_brain_system/core/env_proxy.py`                |
| 技能注册表      | `robot_brain_system/core/skill_manager.py`            |
| 所有技能实现    | `robot_brain_system/skills/`                          |
| 主配置文件      | `robot_brain_system/conf/config.yaml`                 |

## 已实现技能列表

| 技能名                          | ExecutionMode | 说明                                     |
| ------------------------------- | ------------- | ---------------------------------------- |
| `object_tracking`               | PREACTION     | SAM2+Cutie 实时目标检测，写入 `xxx_aabb` |
| `move_to_target_object`         | STEPACTION    | 移动到目标物体中心                       |
| `move_to_target_pose`           | STEPACTION    | 移动到指定位姿                           |
| `grasp_spanner`                 | STEPACTION    | 专用抓取策略                             |
| `open_box`                      | STEPACTION    | 按按钮打开箱子                           |
| `move_box_to_suitable_position` | STEPACTION    | 调整箱子位置                             |

## 调试常用命令

```python
# 在任意位置（技能、断点）快速查看
env_proxy.scene.keys()                    # 当前场景物体
env_proxy.peek_observation_buffer(n=10)   # 最近10帧观测
system.state.status                       # 系统状态
skill_executor.status                     # 当前技能状态
```

## 常见问题

| 问题                      | 原因与解决方案                               |
| ------------------------- | -------------------------------------------- |
| 场景加载失败 / 物体找不到 | 未正确解压 env.zip 或 config 中路径未替换    |
| `object_tracking` 无输出  | 未下载 Cutie 权重或 `target_object` 拼写错误 |
| Client 连接不到 Server    | Server 未启动，或 socket 路径不一致          |
| 动作无响应                | 共享内存名称不匹配，或 Server 已崩溃         |



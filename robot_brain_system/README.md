# Robot Brain System

一个简洁、高效的机器人控制架构，集成Qwen VL大脑、Isaac Lab仿真器和基于生成器的技能执行系统。

## 核心特性

- **🧠 Qwen VL大脑**：用于任务解析和技能编排
- **🔄 非阻塞执行**：基于生成器的技能执行模式
- **⚡ 实时监控**：支持中途打断和干预
- **🛡️ 安全保障**：内置安全限制和紧急停止
- **🎯 快速迭代**：简单的架构便于快速开发

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Qwen VL Brain │    │  Isaac Simulator│    │  Skill Manager  │
│                 │────│   (subprocess)  │────│                 │
│ - Task parsing  │    │ - Gym env       │    │ - Skill registry│
│ - Planning      │    │ - Physics sim   │    │ - Execution     │
│ - Monitoring    │    │ - Rendering     │    │ - Generator mode│
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────────┐
                    │ Robot Brain     │
                    │ System          │
                    │ (Orchestrator)  │
                    └─────────────────┘
```

## 快速开始

### 1. 基本使用

```python
from robot_brain_system import RobotBrainSystem
from robot_brain_system.configs.config import DEVELOPMENT_CONFIG

# 初始化系统
system = RobotBrainSystem(DEVELOPMENT_CONFIG)
system.initialize()
system.start()

# 执行任务
system.execute_task("Reset the robot to home position")

# 监控状态
status = system.get_status()
print(f"System status: {status['system']['status']}")

# 关闭系统
system.shutdown()
```

### 2. 运行示例

```bash
# 运行组件测试
python robot_brain_system/examples/test_components.py

# 运行完整演示
python robot_brain_system/examples/simple_demo.py
```

## 技能系统

### 技能类型

- **FUNCTION**: 固定逻辑的Python函数
- **POLICY**: 训练好的强化学习策略

### 执行模式

- **DIRECT**: 直接执行，无需与环境交互
- **GENERATOR**: 生成器模式，支持env.step()调用和中途打断

### 创建自定义技能

```python
from robot_brain_system.core.skill_manager import skill_register
from robot_brain_system.core.types import SkillType, ExecutionMode, Action, Observation

@skill_register(
    name="my_skill",
    skill_type=SkillType.FUNCTION,
    execution_mode=ExecutionMode.GENERATOR,
    description="My custom skill",
    timeout=10.0,
    requires_env=True,
)
def my_skill(params: Dict[str, Any]) -> Generator[Action, Observation, bool]:
    # 创建动作
    action = Action(data=np.array([0.0, 0.0, 0.1]), metadata={"skill": "my_skill"})
    
    # 发送动作并接收观察
    observation = yield action
    
    # 返回结果
    return True
```

### 已实现的技能

#### 基础技能 (`basic_skills.py`)
- `reset_to_home`: 回到起始位置
- `wait`: 等待指定时间
- `emergency_stop`: 紧急停止
- `get_current_state`: 获取当前状态

#### 操作技能 (`manipulation_skills.py`)
- `reach_position`: 移动到目标位置
- `grasp_object`: 抓取物体
- `release_object`: 释放物体
- `move_trajectory`: 沿轨迹移动
- `pick_and_place`: 抓取和放置（复合技能）

## 配置说明

### 配置文件：`configs/config.py`

```python
SYSTEM_CONFIG = {
    "simulator": {
        "env_name": "Isaac-Reach-Franka-v0",
        "device": "cuda:0",
        "headless": True,
        # ... 其他仿真器配置
    },
    "brain": {
        "qwen": {
            "model": "qwen-vl",
            "api_key": "your_api_key",
        },
        "monitoring_interval": 1.0,
        # ... 其他大脑配置
    },
    "safety": {
        "emergency_stop_enabled": True,
        "position_limits": {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]},
        # ... 其他安全配置
    }
}
```

## Isaac Lab集成

系统在子进程中运行Isaac Lab，正确的启动流程为：

1. 启动AppLauncher
2. 导入isaaclab包
3. 使用gym.make()创建环境
4. 进入主循环

```python
from isaaclab.app import AppLauncher
import gymnasium as gym

# 启动Isaac Sim
app_launcher = AppLauncher(params)
app = app_launcher.app

# 创建环境
env = gym.make(task_name, cfg=env_cfg)
```

## API参考

### 系统类

#### `RobotBrainSystem`
- `initialize()`: 初始化系统组件
- `start()`: 启动主循环
- `execute_task(instruction, image_data)`: 执行任务
- `interrupt_task(reason)`: 中断当前任务
- `get_status()`: 获取系统状态
- `shutdown()`: 关闭系统

#### `QwenVLBrain`
- `parse_task(instruction, image_data)`: 解析自然语言任务
- `plan_task(task)`: 创建技能执行计划
- `monitor_execution(observation)`: 监控执行状态

#### `SkillRegistry` & `SkillExecutor`
- `register_skill()`: 注册技能
- `execute_skill()`: 执行技能
- `list_skills()`: 列出可用技能

### 数据类型

#### `Action`
```python
@dataclass
class Action:
    data: Union[np.ndarray, Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
```

#### `Observation`
```python
@dataclass
class Observation:
    data: Union[np.ndarray, Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
```

## 开发指南

### 添加新技能

1. 在`skills/`目录下创建Python文件
2. 使用`@skill_register`装饰器注册技能
3. 实现技能函数（直接模式或生成器模式）
4. 重新启动系统以加载新技能

### 调试技巧

1. 使用`DEVELOPMENT_CONFIG`获得详细日志
2. 检查`system.get_status()`了解系统状态
3. 运行`test_components.py`验证组件功能
4. 使用技能的`metadata`字段传递调试信息

### 性能优化

1. 对于不需要环境交互的技能使用`DIRECT`模式
2. 合理设置技能超时时间
3. 在生成器技能中添加适当的`time.sleep()`
4. 使用GPU加速Isaac Lab仿真

## 故障排除

### 常见问题

1. **导入错误**: 确保所有依赖包正确安装
2. **Isaac启动失败**: 检查CUDA驱动和Isaac Lab安装
3. **技能未找到**: 确保技能正确注册并导入
4. **系统卡住**: 检查技能是否正确使用yield

### 日志检查

系统日志包含详细的执行信息：
```python
from robot_brain_system.utils.logging_utils import setup_logging

logger = setup_logging("DEBUG", "logs/debug.log")
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 添加测试
4. 提交Pull Request

## 许可证

[您的许可证信息]

---

**注意**: 这是一个重构后的简化版本，专注于快速迭代开发。如需完整功能，请参考原始`intelligent_robot_system`。

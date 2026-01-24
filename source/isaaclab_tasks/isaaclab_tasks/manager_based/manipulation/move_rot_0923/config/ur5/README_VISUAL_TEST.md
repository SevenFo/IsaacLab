# Visual Test Environment 使用说明

## 概述

`Isaac-Move-Box-UR5-IK-Rel-rot-0923-visual-test` 是专为测试 ImagePipeline 目标检测能力而设计的仿真环境。

## 三种测试场景

### 场景1：红色工具箱检测测试
**目标**: 测试模型检测红色工具箱的能力

**场景配置**:
- 目标物体：1个红色工具箱 (heavy_box)
- 干扰物体：0-3个随机颜色的工具箱（绿色/蓝色/橙色）
- 随机化：位置、朝向

**测试指令示例**:
```python
test red toolbox
test red box
```

### 场景2：扳手检测测试
**目标**: 测试模型检测扳手的能力

**场景配置**:
- 目标物体：1个扳手 (spanner)
- 干扰物体：0-2个其他工具（螺丝刀/锤子）
- 随机化：位置、全方位旋转（roll/pitch/yaw）
- 场景切换：自动隐藏场景1的所有箱子

**测试指令示例**:
```python
test spanner
test wrench
```

### 场景3：人手检测测试
**目标**: 测试模型检测人手的能力

**场景配置**:
- 目标物体：Alice手掌（朝上姿态）
- 干扰物体：无
- 随机化：在操作位置附近小幅度移动（±5-10cm）
- 场景切换：自动隐藏场景1和场景2的所有物体

**测试指令示例**:
```python
test hand
test palm
```

## 环境使用

### 1. 在 run_imagepipeline.py 中使用

修改 simulator 配置：
```python
# 在测试脚本中
sim_config_dict = OmegaConf.to_container(
    self.cfg["simulator"], resolve=True
)
# 切换到 visual test 环境
sim_config_dict["task"] = "Isaac-Move-Box-UR5-IK-Rel-rot-0923-visual-test"
```

### 2. 场景切换

环境会根据每次 reset 自动随机选择场景，或者通过环境参数指定：

```python
# 在事件配置中，可以通过修改 visual_test_mode 来控制
# 0 = 红色工具箱测试
# 1 = 扳手测试
# 2 = 人手测试
```

### 3. 测试流程

```bash
# 启动测试环境
./isaaclab.sh -p scripts/run_imagepipeline.py

# 在 UI 中执行测试
> test red box camera_left
> test spanner camera_right
> test hand camera_wrist
```

## 技术实现

### 对象池机制
- 预先创建所有可能的资产（最大数量）
- 通过位置隐藏实现"动态出现/消失"
- 隐藏位置：(100, 100, -50) 米（地平面以下）

### 事件管理
- `red_box_test_randomize`: 场景1的随机化
- `spanner_test_randomize`: 场景2的随机化
- `hand_test_randomize`: 场景3的随机化
- `hide_*`: 场景切换时的资产隐藏

### 资产列表
```
目标资产：
- heavy_box (红色工具箱)
- spanner (扳手)
- alice (人手模型)

干扰资产池：
- distractor_box_1/2/3 (干扰箱子)
- screwdriver (螺丝刀)
- hammer (锤子)
```

## 配置文件

- `move_joint_pos_env_cfg_visual_test.py`: 场景和资产配置
- `move_ik_rel_env_cfg_visual_test.py`: IK控制配置
- `mdp/visual_test_events.py`: 事件处理函数

## 调试建议

1. **检查资产路径**: 确保 USD 文件路径正确
2. **观察隐藏位置**: 如果物体"消失"，检查是否移到了 (100, 100, -50)
3. **材质随机化**: 干扰箱子的材质通过 USD 资产变体实现
4. **相机视角**: 使用多个相机角度测试鲁棒性

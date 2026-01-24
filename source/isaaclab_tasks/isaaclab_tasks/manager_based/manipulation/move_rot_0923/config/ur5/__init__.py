import gymnasium as gym
import os

from . import agents, move_ik_rel_env_cfg, move_ik_rel_env_cfg_visual_test

# 逆运动学 + 相对位姿控制
print("Registering UR5 Move Box IK Rel Env")
gym.register(
    id="Isaac-Move-Box-UR5-IK-Rel-rot-0923",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        # 提供的配置是类
        "env_cfg_entry_point": move_ik_rel_env_cfg.UR5BoxMoveEnvCfg,
    },
    disable_env_checker=True,
)

# 逆运动学 + 相对位姿控制 + 用于视觉 pipeline 的测试，需要准备随机化的 assets
# （1）红色工具箱检测测试
#   在仿真场景的实验桌上放置目标“红色工具箱”。
#   混入干扰项：同构型的绿色、蓝色、橙色工具箱。(这些通过普通 box 随机化 materials 实现，其中 box 就是由 self.scene.box + self.scene.button + self.scene.bar 三个资产构成的东西（不要管 heavy box)，但是随机化材质不需要随机化 bar 和 button，位置就围绕着 self.scene.box 虽在的位置周围不要太大随机摆放)
#   每次测试随机改变所有箱子的摆放位置和顺序。
#   下发识别指令，记录模型是否能准确框选红色工具箱。
# （2）扳手检测测试
#   在工具箱内或桌面上放置目标“扳手”。
#   混入干扰项：同等大小的螺丝刀 (/data/shared_folder/IssacAsserts/Projects/tools/Screwdriver.usd)、锤子 (/data/shared_folder/IssacAsserts/Projects/tools/Hammer.usd) 异形工具。
#   切换到这个场景的时候，要把上面的红色工具箱检测测试用到的四个箱子都移到别的位置避免干扰，让后他们的位置也是参考 self.scene.box 的位置随机摆放。
#   每次测试随机改变工具的摆放位置和旋转角度。
#   下发识别指令，记录模型是否能准确框选扳手。
# （3）人手检测测试
#   这里的人手实际上就是 alice, 他的位置就是 docker/export/workspace/repos/IsaacLab/robot_brain_system/skills/alice_control_skills.py 中的 move_to_operation_position 要移动到的位置
#   然后在这个位置进行小幅度的左右，前后，上下移动，就可以，不要幅度太大
#   同样这个场景在测试的时候也要将测试（1）（2）用到的资产都移开避免干扰。
#   在场景中引入协作人员模型，设置手掌朝上姿态。
#   在机械臂工作空间内随机变换人手的位置和角度。
#   下发识别指令，记录模型是否能准确框选人手部位。
gym.register(
    id="Isaac-Move-Box-UR5-IK-Rel-rot-0923-visual-test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        # 提供的配置是类
        "env_cfg_entry_point": move_ik_rel_env_cfg_visual_test.UR5BoxMoveEnvCfg,
    },
    disable_env_checker=True,
)

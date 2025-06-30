import torch

from isaaclab.utils.math import quat_inv, quat_mul, matrix_from_quat
from isaaclab.utils import configclass

from .quadcopter_controller import QuadcopterController

@configclass
class DeltaPIDControllerCfg:
    position_gain: float = 2.0
    attitude_gain: float = 0.5
    derivative_gain: float = 0.1
    max_position_delta: float = 0.5  # 米
    max_rotation_delta: float = 0.1  # 弧度

class DeltaPIDController(QuadcopterController):
    def __init__(self, config: DeltaPIDControllerCfg, device="cuda:0"):
        super().__init__(device)
        self.pos_gain = config.position_gain
        self.att_gain = config.attitude_gain
        self.deriv_gain = config.derivative_gain
        self.max_pos_delta = config.max_position_delta
        self.max_rot_delta = config.max_rotation_delta
        
    def compute_control(self, current_pos, current_rot, action):
        """
        参数:
            action: [B, 7] 动作向量 [delta_x, delta_y, delta_z, delta_qx, delta_qy, delta_qz, delta_qw]
        """
        batch_size = current_pos.shape[0]
        
        # 动作解析和缩放
        delta_pos = action[:, :3] * self.max_pos_delta
        delta_rot = action[:, 3:]
        delta_rot_norm = torch.norm(delta_rot, dim=1, keepdim=True)
        delta_rot = torch.where(delta_rot_norm > 0, 
                               delta_rot / delta_rot_norm * self.max_rot_delta,
                               delta_rot)
        
        # 计算目标状态
        target_pos = current_pos + delta_pos
        target_rot = quat_mul(current_rot, delta_rot)
        
        # 计算位置误差
        pos_error = target_pos - current_pos
        
        # 计算姿态误差
        rot_diff = self._quaternion_difference(current_rot, target_rot)
        rot_mat_diff = matrix_from_quat(rot_diff)
        att_error = self._rotation_matrix_to_euler_angles(rot_mat_diff)
        
        # 计算微分项
        dt = 1/100  # 假设固定时间步长
        pos_deriv = (current_pos - self._prev_pos) / dt
        rot_deriv = (current_rot - self._prev_rot) / dt
        
        # 更新历史状态
        self._prev_pos = current_pos.clone()
        self._prev_rot = current_rot.clone()
        
        # PID控制计算
        thrust_global = (
            self.pos_gain * pos_error +
            self.deriv_gain * pos_deriv
        )
        
        moment_global = (
            self.att_gain * att_error +
            self.deriv_gain * rot_deriv[:, :3]  # 只取前3个分量
        )
        
        # 转换到机体坐标系
        rot_mat = matrix_from_quat(current_rot)
        thrust_body = torch.bmm(rot_mat.transpose(1, 2), thrust_global.unsqueeze(-1)).squeeze(-1)
        moment_body = torch.bmm(rot_mat.transpose(1, 2), moment_global.unsqueeze(-1)).squeeze(-1)
        
        return thrust_body, moment_body

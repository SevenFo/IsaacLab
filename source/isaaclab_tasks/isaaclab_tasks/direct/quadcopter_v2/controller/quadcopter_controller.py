import torch
import numpy as np
from isaaclab.utils.math import quat_inv, quat_mul, matrix_from_quat
from abc import ABC, abstractmethod

class QuadcopterController(ABC):
    def __init__(self, device="cuda:0"):
        self.device = device
        self.reset()
    
    def reset(self):
        """重置控制器状态"""
        self._prev_pos = torch.zeros(1, 3, device=self.device)
        self._prev_rot = torch.zeros(1, 4, device=self.device)
        self._prev_rot[:, 0] = 1.0  # 初始化为单位四元数
    
    @abstractmethod
    def compute_control(self, current_pos, current_rot, action):
        """
        计算控制输出
        参数:
            current_pos: [B, 3] 当前位置 (x,y,z)
            current_rot: [B, 4] 当前姿态 (qx,qy,qz,qw)
            action: [B, A] 动作向量
            
        返回:
            thrust: [B, 3] 机体坐标系下的推力向量
            moment: [B, 3] 机体坐标系下的力矩向量
        """
        pass



    def _quaternion_difference(self, q1, q2):
        q1_inv = quat_inv(q1)
        return quat_mul(q2, q1_inv)


    def _rotation_matrix_to_euler_angles(self, R):
        """extrinsic rotations xyz"""
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(R)
        return r.as_euler(seq="xyx") 

        # pytorch implement extrinsic zyx
        sy = torch.sqrt(R[:, 0, 0]*R[:, 0, 0] + R[:, 1, 0]*R[:, 1, 0])
        singular = sy < 1e-6
        x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        y = torch.atan2(-R[:, 2, 0], sy)
        z = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        x[singular] = torch.atan2(-R[singular, 1, 2], R[singular, 1, 1])
        y[singular] = torch.atan2(-R[singular, 2, 0], sy[singular])
        z[singular] = 0
        return torch.stack([x, y, z], dim=-1)

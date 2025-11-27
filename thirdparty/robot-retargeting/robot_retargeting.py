import ctypes
from platform import *
import os
import sys
if sys.version_info < (3, 9):
    from typing import List  # 按需导入
else:
    # Python 3.9+ 可以直接使用标准容器类型
    pass

# 加载动态链接库
architecture = machine()
retargeting_file_path = None
retargeting_file_name = None

if architecture == "arm64" or architecture == "aarch64":
    retargeting_file_path = "lib/arm64"
    retargeting_file_name = "librobot-retargeting.so"
elif architecture == "x86_64":
    retargeting_file_path = "lib/x86_64"
    retargeting_file_name = "librobot-retargeting.so.1.6.0"
elif architecture == "AMD64":
    retargeting_file_path = "lib/amd64"
    retargeting_file_name = "robot-retargeting.dll"
else:
    raise Exception("Unsupported architecture")

retargeting_lib_file = os.path.join(os.path.dirname(
    __file__), retargeting_file_path, retargeting_file_name)
print(f'retargeting_lib_file = {retargeting_lib_file}')

retargeting_lib = ctypes.CDLL(retargeting_lib_file)

# 定义常量
MAX_BONE_NAME_LEN = 64
MAX_CHILDREN_BONES = 10
MAX_JOINT_NUM = 128


class MocapBone_c(ctypes.Structure):
    """MocapBone_c结构体"""
    _fields_ = [
        ("mName", ctypes.c_char * MAX_BONE_NAME_LEN),
        ("mTag", ctypes.c_int),
        ("len", ctypes.c_size_t),
        ("msChildrenBoneName", ctypes.c_char *
         MAX_BONE_NAME_LEN * MAX_CHILDREN_BONES),
        ("mLocalPosition", ctypes.c_float * 3),
        ("mLocalRotation", ctypes.c_float * 4)
    ]


class MocapBone:
    """MocapBone_c 结构体的高级封装类"""

    def __init__(self, c_struct=None):
        """初始化封装类，可传入原生结构体实例"""
        self._c_struct = c_struct or MocapBone_c()

    @property
    def name(self) -> str:
        """获取骨骼名称"""
        return self._c_struct.mName.decode('utf-8').rstrip('\x00')

    @name.setter
    def name(self, value: str):
        """设置骨骼名称"""
        name_bytes = value.encode('utf-8')[:MAX_BONE_NAME_LEN-1]
        self._c_struct.mName = name_bytes + b'\0' * \
            (MAX_BONE_NAME_LEN - len(name_bytes))

    @property
    def tag(self) -> int:
        """获取骨骼标签"""
        return self._c_struct.mTag

    @tag.setter
    def tag(self, value: int):
        """设置骨骼标签"""
        self._c_struct.mTag = value

    @property
    def local_position(self) -> list:
        """获取局部位置 (x, y, z)"""
        return list(self._c_struct.mLocalPosition)

    @local_position.setter
    def local_position(self, value: list):
        """设置局部位置 (x, y, z)"""
        if len(value) != 3:
            raise ValueError("Local position must be a 3-element list")
        self._c_struct.mLocalPosition[:] = value

    @property
    def local_rotation(self) -> list:
        """获取局部旋转 (四元数 x, y, z, w)"""
        return list(self._c_struct.mLocalRotation)

    @local_rotation.setter
    def local_rotation(self, value: list):
        """设置局部旋转 (四元数 x, y, z, w)"""
        if len(value) != 4:
            raise ValueError("Local rotation must be a 4-element list")
        self._c_struct.mLocalRotation[:] = value

    def get_children_count(self) -> int:
        """获取子骨骼数量"""
        return self._c_struct.len

    def set_children_count(self, count: int):
        """设置子骨骼数量"""
        if count < 0 or count > MAX_CHILDREN_BONES:
            raise ValueError(
                f"Children count must be between 0 and {MAX_CHILDREN_BONES}")
        self._c_struct.len = count

    def get_child_name(self, index: int) -> str:
        """获取指定索引的子骨骼名称"""
        if index < 0 or index >= self.get_children_count():
            raise IndexError("Child index out of range")
        return self._c_struct.msChildrenBoneName[index].value.decode('utf-8').rstrip('\x00')

    def set_child_name(self, index: int, name: str):
        """设置指定索引的子骨骼名称"""
        if index < 0 or index >= self.get_children_count():
            raise IndexError("Child index out of range")
        name_bytes = name.encode('utf-8')[:MAX_BONE_NAME_LEN-1]
        padded_name = name_bytes.ljust(MAX_BONE_NAME_LEN, b'\0')
        padded_name_ctypes = (
            ctypes.c_char * MAX_BONE_NAME_LEN).from_buffer_copy(padded_name)
        self._c_struct.msChildrenBoneName[index] = padded_name_ctypes

    def get_children_names(self) -> list:
        """获取所有子骨骼名称列表"""
        return [self.get_child_name(i) for i in range(self.get_children_count())]

    def to_dict(self) -> dict:
        """将骨骼数据转换为字典格式"""
        return {
            'name': self.name,
            'tag': self.tag,
            'local_position': self.local_position,
            'local_rotation': self.local_rotation,
            'children': self.get_children_names()
        }

    def __str__(self) -> str:
        """返回骨骼的简要字符串表示"""
        return (f"MocapBone(name='{self.name}', tag={self.tag}, "
                f"children={self.get_children_count()})")

    def print_details(self):
        """打印骨骼的详细信息"""
        print(f"=== MocapBone: {self.name} (Tag: {self.tag}) ===")
        print(f"Local Position: {self.local_position}")
        print(f"Local Rotation: {self.local_rotation}")
        print(f"Children ({self.get_children_count()}):")
        for i, child_name in enumerate(self.get_children_names()):
            print(f"  {i+1}. {child_name}")

    def get_native_struct(self) -> MocapBone_c:
        """获取原生结构体实例，用于与 C 代码交互"""
        return self._c_struct


class RetargetingResult_c(ctypes.Structure):
    """描述RetargetingResult_c结构体"""
    _fields_ = [
        ("mTimestamp", ctypes.c_uint64),
        ("mRootPosition", ctypes.c_float * 3),
        ("mRootRotation", ctypes.c_float * 4),
        ("mRootVelocity", ctypes.c_float * 3),
        ("mRootAngularVelocity", ctypes.c_float * 3),
        ("len", ctypes.c_size_t),
        ("mName", ctypes.c_char * MAX_BONE_NAME_LEN * MAX_JOINT_NUM),
        ("mPosition", ctypes.c_double * MAX_JOINT_NUM),
        ("mVelocity", ctypes.c_double * MAX_JOINT_NUM),
        ("mEffort", ctypes.c_double * MAX_JOINT_NUM)
    ]


class RetargetingResult:
    """RetargetingResult_c 结构体的高级封装类"""

    def __init__(self, c_struct=None):
        """初始化封装类，可传入原生结构体实例"""
        self._c_struct = c_struct or RetargetingResult_c()

    @property
    def timestamp(self) -> int:
        """获取时间戳"""
        return self._c_struct.mTimestamp

    @timestamp.setter
    def timestamp(self, value: int):
        """设置时间戳"""
        self._c_struct.mTimestamp = value

    @property
    def root_position(self) -> list:
        """获取根位置 (x, y, z)"""
        return list(self._c_struct.mRootPosition)

    @root_position.setter
    def root_position(self, value: list):
        """设置根位置 (x, y, z)"""
        if len(value) != 3:
            raise ValueError("Root position must be a 3-element list")
        self._c_struct.mRootPosition[:] = value

    @property
    def root_rotation(self) -> list:
        """获取根旋转 (四元数 x, y, z, w)"""
        return list(self._c_struct.mRootRotation)

    @root_rotation.setter
    def root_rotation(self, value: list):
        """设置根旋转 (四元数 x, y, z, w)"""
        if len(value) != 4:
            raise ValueError("Root rotation must be a 4-element list")
        self._c_struct.mRootRotation[:] = value

    @property
    def root_velocity(self) -> list:
        """获取根速度 (x, y, z)"""
        return list(self._c_struct.mRootVelocity)

    @root_velocity.setter
    def root_velocity(self, value: list):
        """设置根速度 (x, y, z)"""
        if len(value) != 4:
            raise ValueError("Root rotation must be a 4-element list")
        self._c_struct.mRootRotation[:] = value

    @property
    def root_angular_velocity(self) -> list:
        """获取根角速度 (wx, wy, wz)"""
        return list(self._c_struct.mRootAngularVelocity)

    @root_angular_velocity.setter
    def root_angular_velocity(self, value: list):
        """设置根角速度 (wx, wy, wz)"""
        if len(value) != 3:
            raise ValueError("Root angular velocity must be a 3-element list")
        self._c_struct.mRootAngularVelocity[:] = value

    @property
    def joint_count(self) -> int:
        """获取时间戳"""
        return self._c_struct.len

    @joint_count.setter
    def joint_count(self, value: int):
        """设置时间戳"""
        self._c_struct.len = value

    def get_joint_name(self, index: int) -> str:
        """获取指定索引关节的名称"""
        if index < 0 or index >= self.joint_count:
            raise IndexError("Joint index out of range")
        return self._c_struct.mName[index].value.decode('utf-8').rstrip('\x00')

    def set_joint_name(self, index: int, name: str):
        """设置指定索引关节的名称"""
        if index < 0 or index >= self.joint_count:
            raise IndexError("Joint index out of range")
        name_bytes = name.encode('utf-8')[:MAX_BONE_NAME_LEN-1]
        padded_name = name_bytes.ljust(MAX_BONE_NAME_LEN, b'\0')
        padded_name_ctypes = (
            ctypes.c_char * MAX_BONE_NAME_LEN).from_buffer_copy(padded_name)
        self._c_struct.mName[index] = padded_name_ctypes

    def get_joint_position(self, index: int) -> float:
        """获取指定索引关节的位置"""
        if index < 0 or index >= self.joint_count:
            raise IndexError("Joint index out of range")
        return self._c_struct.mPosition[index]

    def set_joint_position(self, index: int, position: float):
        """设置指定索引关节的位置"""
        if index < 0 or index >= self.joint_count:
            raise IndexError("Joint index out of range")
        self._c_struct.mPosition[index] = position

    def get_joint_velocity(self, index: int) -> float:
        """获取指定索引关节的速度"""
        if index < 0 or index >= self.joint_count:
            raise IndexError("Joint index out of range")
        return self._c_struct.mVelocity[index]

    def set_joint_velocity(self, index: int, velocity: float):
        """设置指定索引关节的速度"""
        if index < 0 or index >= self.joint_count:
            raise IndexError("Joint index out of range")
        self._c_struct.mVelocity[index] = velocity

    def get_joint_effort(self, index: int) -> float:
        """获取指定索引关节的力度"""
        if index < 0 or index >= self.joint_count:
            raise IndexError("Joint index out of range")
        return self._c_struct.mEffort[index]

    def set_joint_effort(self, index: int, effort: float):
        """设置指定索引关节的力度"""
        if index < 0 or index >= self.joint_count:
            raise IndexError("Joint index out of range")
        self._c_struct.mEffort[index] = effort

    if sys.version_info < (3, 9):
        @property
        def joint_names(self) -> List[str]:
            """获取所有关节名称的列表（按索引顺序）"""
            return [self.get_joint_name(i) for i in range(self.joint_count)]

        @property
        def positions(self) -> List[float]:
            """获取所有关节位置的列表（按索引顺序）"""
            return [self.get_joint_position(i) for i in range(self.joint_count)]

        @property
        def velocities(self) -> List[float]:
            """获取所有关节速度的列表（按索引顺序）"""
            return [self.get_joint_velocity(i) for i in range(self.joint_count)]

        @property
        def efforts(self) -> List[float]:
            """获取所有关节力度的列表（按索引顺序）"""
            return [self.get_joint_effort(i) for i in range(self.joint_count)]
    
    else:
        @property
        def joint_names(self) -> list[str]:
            """获取所有关节名称的列表（按索引顺序）"""
            return [self.get_joint_name(i) for i in range(self.joint_count)]

        @property
        def positions(self) -> list[float]:
            """获取所有关节位置的列表（按索引顺序）"""
            return [self.get_joint_position(i) for i in range(self.joint_count)]

        @property
        def velocities(self) -> list[float]:
            """获取所有关节速度的列表（按索引顺序）"""
            return [self.get_joint_velocity(i) for i in range(self.joint_count)]

        @property
        def efforts(self) -> list[float]:
            """获取所有关节力度的列表（按索引顺序）"""
            return [self.get_joint_effort(i) for i in range(self.joint_count)]
        
    def to_dict(self) -> dict:
        """将所有数据转换为字典格式"""
        data = {
            'timestamp': self.timestamp,
            'root_position': self.root_position.tolist(),
            'root_rotation': self.root_rotation.tolist(),
            'root_angular_velocity': self.root_angular_velocity,
            'joints': []
        }

        for i in range(self.joint_count):
            joint_data = {
                'name': self.get_joint_name(i),
                'position': self.get_joint_position(i),
                'velocity': self.get_joint_velocity(i),
                'effort': self.get_joint_effort(i)
            }
            data['joints'].append(joint_data)

        return data

    def __str__(self) -> str:
        """返回对象的字符串表示形式，用于打印"""
        return f"RetargetingResult(timestamp={self.timestamp}, root_pos={self.root_position}, joint_count={self.joint_count})"

    def print_details(self):
        """打印详细信息，包括所有关节数据"""
        print(str(self))
        print("Joints:")
        for i in range(self.joint_count):
            print(f"  {i}: {self.get_joint_name(i)} - pos={self.get_joint_position(i):.4f}, "
                  f"vel={self.get_joint_velocity(i):.4f}, effort={self.get_joint_effort(i):.4f}")

    def print_brief(self):
        """打印精简信息"""
        print(str(self))

    def get_native_struct(self) -> RetargetingResult_c:
        """获取原生结构体实例，用于与 C 代码交互"""
        return self._c_struct

class Vector3:
    """三维向量类，支持通过 x, y, z 属性访问"""
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self) -> str:
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"
    
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

class Quaternion:
    """四元数类，支持通过 w, x, y, z 属性访问"""
    __slots__ = ['w', 'x', 'y', 'z']
    
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self) -> str:
        return f"Quaternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"
    
    def __iter__(self):
        yield self.w
        yield self.x
        yield self.y
        yield self.z

class RetargetingAvatarState_c(ctypes.Structure):
    """C结构体: 骨骼在全局坐标系下的状态信息"""
    _fields_ = [
        ("mTimestamp", ctypes.c_uint64),  # 时间戳（毫秒）
        ("len", ctypes.c_size_t),
        ("mName", ctypes.c_char * MAX_BONE_NAME_LEN *
         MAX_JOINT_NUM),  # 骨骼名称（与后续数组顺序对应）
        ("mParentBoneName", ctypes.c_char * MAX_BONE_NAME_LEN *
         MAX_JOINT_NUM),  # 父骨骼名称 （根节点的父骨骼为"World"）
        ("mBonePosition", ctypes.c_float * 3 *
         MAX_JOINT_NUM),  # 骨骼全局位置 [x, y, z]（单位：米）
        ("mBoneRotation", ctypes.c_float * 4 *
         MAX_JOINT_NUM),  # 骨骼全局旋转 [w, x, y, z]（单位：无）
    ]


class RetargetingAvatarState:
    """RetargetingAvatarState_c 结构体的高级封装类"""

    def __init__(self, c_struct=None):
        """初始化封装类，可传入原生结构体实例"""
        self._c_struct = c_struct or RetargetingAvatarState_c()
        # 初始化根骨骼的父骨骼为"World"
        if c_struct is None:
            self.set_parent_bone_name(0, "World")

    @property
    def timestamp(self) -> int:
        """获取时间戳（毫秒）"""
        return self._c_struct.mTimestamp

    @timestamp.setter
    def timestamp(self, value: int):
        """设置时间戳（毫秒）"""
        self._c_struct.mTimestamp = value

    @property
    def bone_count(self) -> int:
        """获取骨骼数量"""
        return self._c_struct.len

    @bone_count.setter
    def bone_count(self, value: int):
        """设置骨骼数量"""
        if value < 0 or value > MAX_JOINT_NUM:
            raise ValueError(f"骨骼数量必须在0到{MAX_JOINT_NUM}之间")
        self._c_struct.len = value

    def get_bone_name(self, index: int) -> str:
        """获取指定索引的骨骼名称"""
        if index < 0 or index >= self.bone_count:
            raise IndexError(f"骨骼索引 {index} 超出范围")
        
        return self._c_struct.mName[index].value.decode('utf-8').rstrip('\x00')

    def set_bone_name(self, index: int, name: str):
        """设置指定索引的骨骼名称"""
        if index < 0 or index >= self.bone_count:
            raise IndexError(f"骨骼索引 {index} 超出范围")

        name_bytes = name.encode('utf-8')[:MAX_BONE_NAME_LEN-1]
        padded_name = name_bytes.ljust(MAX_BONE_NAME_LEN, b'\x00')
        padded_name_ctypes = (
            ctypes.c_char * MAX_BONE_NAME_LEN).from_buffer_copy(padded_name)
        self._c_struct.mName[index] = padded_name_ctypes

    def get_parent_bone_name(self, index: int) -> str:
        """获取指定索引的父骨骼名称"""
        if index < 0 or index >= self.bone_count:
            raise IndexError(f"骨骼索引 {index} 超出范围")

        return self._c_struct.mParentBoneName[index].value.decode('utf-8').rstrip('\x00')

    def set_parent_bone_name(self, index: int, name: str):
        """设置指定索引的父骨骼名称"""
        if index < 0 or index >= self.bone_count:
            raise IndexError(f"骨骼索引 {index} 超出范围")

        name_bytes = name.encode('utf-8')[:MAX_BONE_NAME_LEN-1]
        padded_name = name_bytes.ljust(MAX_BONE_NAME_LEN, b'\x00')
        padded_name_ctypes = (
            ctypes.c_char * MAX_BONE_NAME_LEN).from_buffer_copy(padded_name)
        self._c_struct.mParentBoneName[index] = padded_name_ctypes

    def get_bone_position(self, index: int) -> list:
        """获取指定索引的骨骼全局位置 [x, y, z]"""
        if index < 0 or index >= self.bone_count:
            raise IndexError(f"骨骼索引 {index} 超出范围")

        pos_ptr = ctypes.addressof(self._c_struct.mBonePosition)
        pos_offset = index * 3 * ctypes.sizeof(ctypes.c_float)
        pos_array = (ctypes.c_float * 3).from_address(pos_ptr + pos_offset)
        return list(pos_array)

    def set_bone_position(self, index: int, position: list):
        """设置指定索引的骨骼全局位置 [x, y, z]"""
        if index < 0 or index >= self.bone_count:
            raise IndexError(f"骨骼索引 {index} 超出范围")
        if len(position) != 3:
            raise ValueError("位置必须是三维向量 [x, y, z]")

        pos_ptr = ctypes.addressof(self._c_struct.mBonePosition)
        pos_offset = index * 3 * ctypes.sizeof(ctypes.c_float)
        ctypes.memmove(pos_ptr + pos_offset, (ctypes.c_float * 3)
                       (*position), 3 * ctypes.sizeof(ctypes.c_float))

    def get_bone_rotation(self, index: int) -> list:
        """获取指定索引的骨骼全局旋转 [w, x, y, z]"""
        if index < 0 or index >= self.bone_count:
            raise IndexError(f"骨骼索引 {index} 超出范围")

        rot_ptr = ctypes.addressof(self._c_struct.mBoneRotation)
        rot_offset = index * 4 * ctypes.sizeof(ctypes.c_float)
        rot_array = (ctypes.c_float * 4).from_address(rot_ptr + rot_offset)
        return list(rot_array)

    def set_bone_rotation(self, index: int, rotation: list):
        """设置指定索引的骨骼全局旋转 [w, x, y, z]"""
        if index < 0 or index >= self.bone_count:
            raise IndexError(f"骨骼索引 {index} 超出范围")
        if len(rotation) != 4:
            raise ValueError("旋转必须是四元数 [w, x, y, z]")

        rot_ptr = ctypes.addressof(self._c_struct.mBoneRotation)
        rot_offset = index * 4 * ctypes.sizeof(ctypes.c_float)
        ctypes.memmove(rot_ptr + rot_offset, (ctypes.c_float * 4)
                       (*rotation), 4 * ctypes.sizeof(ctypes.c_float))


    if sys.version_info < (3, 9):
        @property
        def bone_names(self) -> List[str]:
            """获取所有关节名称的列表（按索引顺序）"""
            return [self.get_bone_name(i) for i in range(self.bone_count)]

        @property
        def parent_bone_names(self) -> List[str]:
            """获取所有关节名称的列表（按索引顺序）"""
            return [self.get_parent_bone_name(i) for i in range(self.bone_count)]
        
        @property
        def bone_positions(self) -> List[Vector3]:
            """获取所有骨骼的全局位置列表，每个位置为 Vector3 对象，支持 .x/.y/.z 访问"""
            return [Vector3(*self.get_bone_position(i)) for i in range(self.bone_count)]

        @property
        def bone_rotations(self) -> List[Quaternion]:
            """获取所有骨骼的全局旋转列表，每个旋转为 Quaternion 对象，支持 .w/.x/.y/.z 访问"""
            return [Quaternion(*self.get_bone_rotation(i)) for i in range(self.bone_count)]
    else:
        @property
        def bone_names(self) -> list[str]:
            """获取所有关节名称的列表（按索引顺序）"""
            return [self.get_bone_name(i) for i in range(self.bone_count)]

        @property
        def parent_bone_names(self) -> list[str]:
            """获取所有关节名称的列表（按索引顺序）"""
            return [self.get_parent_bone_name(i) for i in range(self.bone_count)]
        
        @property
        def bone_positions(self) -> list[Vector3]:
            """获取所有骨骼的全局位置列表，每个位置为 Vector3 对象，支持 .x/.y/.z 访问"""
            return [Vector3(*self.get_bone_position(i)) for i in range(self.bone_count)]

        @property
        def bone_rotations(self) -> list[Quaternion]:
            """获取所有骨骼的全局旋转列表，每个旋转为 Quaternion 对象，支持 .w/.x/.y/.z 访问"""
            return [Quaternion(*self.get_bone_rotation(i)) for i in range(self.bone_count)]

    def to_dict(self) -> dict:
        """将所有骨骼数据转换为字典格式"""
        return {
            'timestamp': self.timestamp,
            'bone_count': self.bone_count,
            'bones': [
                {
                    'name': self.get_bone_name(i),
                    'parent': self.get_parent_bone_name(i),
                    'position': self.get_bone_position(i),
                    'rotation': self.get_bone_rotation(i)
                }
                for i in range(self.bone_count)
            ]
        }

    def __str__(self) -> str:
        """返回简要字符串表示"""
        return (f"RetargetingAvatarState(timestamp={self.timestamp}, "
                f"bone_count={self.bone_count})")

    def print_details(self):
        """打印详细信息"""
        print(
            f"=== RetargetingAvatarState (Timestamp: {self.timestamp}ms) ===")
        print(f"Bones: {self.bone_count}")
        for i in range(min(self.bone_count, 10)):  # 只打印前10个骨骼避免过多输出
            bone_name = self.get_bone_name(i)
            parent_name = self.get_parent_bone_name(i)
            position = self.get_bone_position(i)
            rotation = self.get_bone_rotation(i)
            print(f"\nBone #{i}: {bone_name} (Parent: {parent_name})")
            print(
                f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
            print(
                f"  Rotation: [{rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f}, {rotation[3]:.3f}]")

        if self.bone_count > 10:
            print(f"\n... 和其他 {self.bone_count-10} 个骨骼")

    def get_native_struct(self) -> RetargetingAvatarState_c:
        """获取原生结构体实例，用于与C代码交互"""
        return self._c_struct


# 定义函数原型
retargeting_lib.RetargetingCreateHandle_c.argtypes = [ctypes.c_char_p]
retargeting_lib.RetargetingCreateHandle_c.restype = ctypes.c_uint64

retargeting_lib.RetargetingDestroyHandle_c.argtypes = [ctypes.c_uint64]
retargeting_lib.RetargetingDestroyHandle_c.restype = None

retargeting_lib.RetargetingUpdateMocapData_c.argtypes = [
    ctypes.c_uint64, ctypes.POINTER(MocapBone_c), ctypes.c_size_t]
retargeting_lib.RetargetingUpdateMocapData_c.restype = None

retargeting_lib.RetargetingGetResult_c.argtypes = [
    ctypes.c_uint64, ctypes.POINTER(RetargetingResult_c)]
retargeting_lib.RetargetingGetResult_c.restype = ctypes.c_bool

retargeting_lib.RetargetingMocapData_c.argtypes = [ctypes.c_uint64, ctypes.POINTER(
    MocapBone_c), ctypes.c_size_t, ctypes.POINTER(RetargetingResult_c)]
retargeting_lib.RetargetingMocapData_c.restype = ctypes.c_bool

retargeting_lib.RetargetingGetAvatarState_c.argtypes = [
    ctypes.c_uint64, ctypes.POINTER(RetargetingAvatarState_c)]
retargeting_lib.RetargetingGetAvatarState_c.restype = ctypes.c_bool

# 封装的Python接口函数


def RetargetingCreateHandle(config_file):
    '''
    基于配置文件，创建retargeting句柄
    '''
    return retargeting_lib.RetargetingCreateHandle_c(config_file.encode())


def RetargetingDestroyHandle(handle):
    '''
    销毁句柄
    '''
    retargeting_lib.RetargetingDestroyHandle_c(handle)


def RetargetingUpdateMocapData(handle, mocapBones):
    '''
    异步接口，更新骨骼数据，mocapBones是个list，每个元素类型是MocapBone，而不是MocapBone_c
    '''
    count = len(mocapBones)
    bones_c = [bone.get_native_struct() for bone in mocapBones]
    arr = (MocapBone_c * count)(*bones_c)
    retargeting_lib.RetargetingUpdateMocapData_c(handle, arr, count)


def RetargetingGetResult(handle):
    '''
    异步接口，获取retargeting结果，返回success和result。其中result类型是RetargetingResult，而不是RetargetingResult_c
    '''
    result = RetargetingResult_c()
    success = retargeting_lib.RetargetingGetResult_c(
        handle, ctypes.byref(result))
    return success, RetargetingResult(result)


def RetargetingMocapData(handle, mocapBones):
    '''
    同步接口，获取retargeting结果，返回success和result。其中result类型是RetargetingResult，而不是RetargetingResult_c
    '''
    count = len(mocapBones)
    bones_c = [bone.get_native_struct() for bone in mocapBones]
    arr = (MocapBone_c * count)(*bones_c)
    result = RetargetingResult_c()
    success = retargeting_lib.RetargetingMocapData_c(
        handle, arr, count, ctypes.byref(result))
    return success, RetargetingResult(result)

def RetargetingGetAvatarState(handle):
    '''
    同步接口，获取骨骼位姿数据，返回success和result。其中result类型是RetargetingAvatarState，而不是RetargetingAvatarState_c
    '''
    result = RetargetingAvatarState_c()
    success = retargeting_lib.RetargetingGetAvatarState_c(
        handle, ctypes.byref(result))
    return success, RetargetingAvatarState(result)

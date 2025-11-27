/**
 * @file robot_retargeting.h
 * @brief 机器人运动重定向接口头文件
 *
 * 该文件定义了运动重定向系统的核心数据结构和接口函数，用于将动作捕捉数据转换为机器人控制指令。
 */

#ifndef _ROBOT_RETARGETING_H_
#define _ROBOT_RETARGETING_H_

#include <string>
#include <vector>

// 跨平台导出宏定义
#ifdef _WIN32
    #ifdef NR_RETARGETING_LIB_EXPORTS
        #define NR_RETARGETING_API __declspec(dllexport)
    #else
        #define NR_RETARGETING_API __declspec(dllimport)
    #endif
#else
    // Linux/macOS等平台使用visibility属性
    #ifdef NR_RETARGETING_LIB_EXPORTS
        #define NR_RETARGETING_API __attribute__((visibility("default")))
    #else
        #define NR_RETARGETING_API
    #endif
#endif

/**
 * @struct MocapBone
 * @brief 动作捕捉骨骼数据结构
 *
 * 描述单个骨骼在动作捕捉系统中的位置、旋转及其层级关系。
 */
struct NR_RETARGETING_API MocapBone {
    std::string mName;  ///< 骨骼名称标识符
    int mTag;           ///< 自定义标签（可用于特殊标记或分类）

    std::vector<std::string> msChildrenBoneName;  ///< 子骨骼名称列表（描述骨骼层级关系）

    float mLocalPosition[3];  ///< 局部坐标系中的位置 [x, y, z]（单位：米）
    float mLocalRotation[4];  ///< 局部坐标系中的四元数旋转 [w, x, y, z]（单位：无，需满足||q||=1）
};

/**
 * @struct RetargetingResult
 * @brief 重定向计算结果结构体
 *
 * 包含机器人控制所需的完整状态信息（位置、速度、力矩等）。
 */
struct NR_RETARGETING_API RetargetingResult {
    uint64_t mTimestamp;  ///< 时间戳（毫秒单位，系统参考时间）

    float mRootPosition[3];  ///< 机器人根关节全局位置 [x, y, z]（单位：米）
    float mRootRotation[4];  ///< 机器人根关节全局旋转 [w, x, y, z]（单位：无）

    float mRootVelocity[3];         ///< 根关节坐标系下线速度 [x, y, z]（单位：米/秒）
    float mRootAngularVelocity[3];  ///< 根关节坐标系下角速度 [x, y, z]（单位：弧度/秒）

    std::vector<std::string> mName;  ///< 关节名称列表（与mPosition等数组顺序对应）
    std::vector<double> mPosition;   ///< 关节目标位置（单位：弧度或米）
    std::vector<double> mVelocity;   ///< 关节目标速度（单位：弧度/秒或米/秒）
    std::vector<double> mEffort;     ///< 关节目标力矩（单位：牛顿·米）
};

struct NR_RETARGETING_API Rotation {
    float w;
    float x;
    float y;
    float z;
};

struct NR_RETARGETING_API Position {
    float x;
    float y;
    float z;
};

struct NR_RETARGETING_API AvatarState {
    uint64_t mTimestamp;                       ///< 时间戳（毫秒单位，系统参考时间）
    std::vector<std::string> mBoneName;        ///< 骨骼名称列表（与后续数组顺序对应）
    std::vector<std::string> mParentBoneName;  ///< 父骨骼名称列表（根节点的父骨骼为"World"）
    std::vector<Position> mBonePosition;       ///< 骨骼全局位置 [x, y, z]（单位：米）
    std::vector<Rotation> mBoneRotation;       ///< 骨骼全局旋转 [w, x, y, z]（单位：无）
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建重定向处理器实例
 *
 * @param config_file 配置文件路径（JSON/XML格式，包含机器人模型参数和重定向规则）
 * @return uint64_t 处理器句柄（0表示创建失败）
 */
NR_RETARGETING_API uint64_t RetargetingCreateHandle(const char* config_file);

/**
 * @brief 销毁重定向处理器并释放资源
 *
 * @param handle 由RetargetingCreateHandle创建的处理器句柄
 */
NR_RETARGETING_API void RetargetingDestroyHandle(uint64_t handle);

/**
 * @brief 更新动作捕捉数据（非阻塞式）
 *
 * 将新的骨骼数据推送到处理队列，内部进行异步处理。
 *
 * @param handle 处理器句柄
 * @param mocapBones 动作捕捉骨骼数据数组
 */
NR_RETARGETING_API void RetargetingUpdateMocapData(uint64_t handle, const std::vector<MocapBone>& mocapBones);

/**
 * @brief 获取最新重定向结果（非阻塞式）
 *
 * @param handle 处理器句柄
 * @param[out] result 输出结果结构体（需预先分配内存）
 * @return bool true: 成功获取新数据, false: 无新数据或出错
 */
NR_RETARGETING_API bool RetargetingGetResult(uint64_t handle, RetargetingResult& result);

/**
 * @brief 同步处理动作捕捉数据（阻塞式）
 *
 * 单次处理输入数据并立即返回结果（适用于实时性要求高的场景）。
 *
 * @param handle 处理器句柄
 * @param mocapBones 动作捕捉骨骼数据数组
 * @param[out] result 输出结果结构体
 * @return bool true: 处理成功, false: 处理失败
 */
NR_RETARGETING_API bool RetargetingMocapData(uint64_t handle, const std::vector<MocapBone>& mocapBones, RetargetingResult& result);

/**
 * @brief 获取最新骨骼全局位姿结果（阻塞式）
 *
 * 调用RetargetingMocapData后才可调用本函数。
 *
 * @param handle 处理器句柄
 * @param[out] state 输出骨骼全局位姿结构体
 * @return bool true: 处理成功, false: 处理失败
 */
NR_RETARGETING_API bool RetargetingGetAvatarState(uint64_t handle, AvatarState& state);

#ifdef __cplusplus
}
#endif

#endif  // _ROBOT_RETARGETING_H_

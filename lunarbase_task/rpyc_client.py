import requests
import pickle
import numpy as np

# Reset环境
response = requests.post("http://127.0.0.1:18861/reset")
print(response.json())

        #     # delat pose: pos, rot, axis-angle
        #     "end_effector": gym.spaces.Box(
        #         low=-1.0,
        #         high=1.0,
        #         shape=(6,),
        #         dtype=np.float32,
        #     ),
        #     "gripper": gym.spaces.Box(
        #         low=0.0,
        #         high=1.0,
        #         shape=(2,),
        #         dtype=np.float32,
        #     ),
        # }

while True:
    import pdb
    pdb.set_trace()
    # 执行step（手动模式）
    # 构造动作数据（保持原始numpy格式）
    action = {
        'end_effector': np.random.random(size=(1,6)).astype(np.float32),
        'gripper': np.random.random(size=(1,2)).astype(np.float32)
    }
    action['end_effector'][:,3:] = 0  # 保持原有逻辑

    # 序列化
    serialized = pickle.dumps(action)

    # 发送请求（注意header修改）
    response = requests.post(
        "http://127.0.0.1:18861/step",
        data=serialized,
        headers={'Content-Type': 'application/python-pickle'}
    )
    print(response.json())
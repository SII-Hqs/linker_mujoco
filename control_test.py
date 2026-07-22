import math
import time
import mujoco
from mujoco import viewer

MODEL_PATH = r"C:\Users\wqf\Desktop\linker_mujoco\src\linker_hand_mujoco_ros2\linker_hand_mujoco_ros2\urdf\L20\linker_hand_l20_left\linker_hand_l20_left.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("执行器数量:", model.nu)
print("当前控制第 0 个执行器:", model.actuator(0).name)

with viewer.launch_passive(model, data) as v:
    t = 0.0

    while v.is_running():
        # 其他执行器保持 0
        data.ctrl[:] = 0

        # 第 0 个执行器来回运动
        data.ctrl[0] = 0.3 * math.sin(t)

        mujoco.mj_step(model, data)
        v.sync()

        t += model.opt.timestep
        time.sleep(model.opt.timestep)

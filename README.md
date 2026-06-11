# linker_hand_sim
[Isaac-Gym by Python3](https://github.com/linkerbotai/linker_hand_sim/blob/main/linker_hand_isaac_gym_urdf/README.md)

[Mujoco by ros noetic](linker_hand_mujoco_ros/README_CN.md)

[Mujoco by ros2 jazzy](linker_hand_mujoco_ros2/README_CN.MD)

[PyBullet by ros noetic](linker_hand_pybullet_ros/README_PyBullet_CN.md)

[PyBullet by ros2 jazzy](linker_hand_pybullet_ros2/README_CN.MD)

python -m linker_hand_mujoco_ros2.contact.contact_control_demo_esn \  --solref 0.04 1.0 --solimp 0.85 0.99 0.001 --headless

已添加。现在可以通过命令行参数调节接触刚度：
> 
    1 # 默认（硬接触，solref=0.005）— 力很大
    2 python -m linker_hand_mujoco_ros2.contact.contact_control_demo_esn --headless
    3 
    4 # 软接触 — 增大 solref 时间常数到 0.02s，力会明显变小
    python -m linker_hand_mujoco_ros2.contact.contact_control_demo_esn \
    --solref 0.02 1.0 --headless
    7 
    8 # 更软（0.04s），适合 ESN 输出力矩较小的情况
    9 python -m linker_hand_mujoco_ros2.contact.contact_control_demo_esn \
   10     --solref 0.04 1.0 --solimp 0.85 0.99 0.001 --headless

  参数含义：


  ┌──────────────────────┬──────────────┬──────────────────┐
  │ 参数                  │ 作用          │ 调大效果         │
  ├──────────────────────┼──────────────┼──────────────────┤
  │ solref[0] (时间常数)  │ 约束恢复速度    │ 接触变软，力变小 │
  │ solref[1] (阻尼比)    │ 约束阻尼       │ 减少振荡         │
  │ solimp[0] (dmin)     │ 最小阻抗       │ 接近1=更硬       │
  │ solimp[1] (dmax)     │ 最大阻抗       │ 接近1=更硬       │
  │ solimp[2] (width)    │ 过渡宽度       │ 越大过渡越平滑   │
  └──────────────────────┴──────────────┴──────────────────┘

python src/linker_hand_mujoco_ros2/linker_hand_mujoco_ros2/contact/contact_control_demo_esn_visualized.py  --high-z-offset 0.02 --low-z-offset 0.01 --solref 0.02 1.0

* 控制平台运动+esn可视化
python -m linker_hand_mujoco_ros2.contact.contact_control_gui_esn

* 控制平台运动+esn可视化+目标位置
* conda activate linker_mujoco
python src/linker_hand_mujoco_ros2/linker_hand_mujoco_ros2/contact/contact_control_gui_esn_target.py --solref 0.02 1.0 --target-qpos 0.332 0.254 0.271

* esn可视化+目标位置+注入力
python src/linker_hand_mujoco_ros2/linker_hand_mujoco_ros2/contact/contact_control_gui_esn_target_contact_patch.py --solref 0.02 1.0 --target-qpos 0.332 0.254 0.271
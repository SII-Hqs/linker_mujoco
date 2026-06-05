# L20a 接触力节点说明

本节点针对 `L20a/linker_hand_l20a_right` MuJoCo 模型，使用 `finger_l20_vec.pkl` 中的 21 维关节角格式作为输入，并额外接收一个接触力数值，单位为 g。

## 启动

```bash
colcon build --symlink-install
source ./install/setup.bash
ros2 launch linker_hand_mujoco_ros2 linker_hand_l20a_contact.launch.py
```

## 输入 Topic

```text
/cb_right_hand_l20a_contact_cmd
std_msgs/msg/Float32MultiArray
```

`data` 长度为 22：

- `data[0:21]`：21 个关节角，单位 rad，顺序来自 `finger_l20_vec.pkl`
- `data[21]`：食指接触力，单位 g

21 个关节顺序为：

```text
index_joint0, index_joint1, index_joint2, index_joint3,
little_joint0, little_joint1, little_joint2, little_joint3,
middle_joint0, middle_joint1, middle_joint2, middle_joint3,
ring_joint0, ring_joint1, ring_joint2, ring_joint3,
thumb_joint0, thumb_joint1, thumb_joint2, thumb_joint3, thumb_joint4
```

示例：

```bash
ros2 topic pub /cb_right_hand_l20a_contact_cmd std_msgs/msg/Float32MultiArray '
{
  data: [0.0, 0.33163944, 0.25424874, 0.27051812,
         0.0, 1.01525867, 0.90827024, 0.96639045,
         0.0, 1.1932292, 1.08099997, 1.15017316,
         0.0, 1.20462227, 1.07738614, 1.14632808,
         0.0, 0.79711825, 0.38298517, 0.70328254, 0.70328254,
         50.0]
}'
```

## 可视化

MuJoCo 界面中：

- 红色点：`index_link3` 当前最低点附近的接触面采样点
- 蓝色点：当前最低点
- 绿色箭头：接触力按高斯分布分配到各采样点后的力，方向沿世界坐标 `+Z`

## 输出 Topic

```text
/cb_right_hand_l20a_index_contact_torque
sensor_msgs/msg/JointState
```

其中：

- `name`: `index_joint0` 到 `index_joint3`
- `effort`: 当前帧食指四个关节的接触力矩，单位 N*m

查看：

```bash
ros2 topic echo /cb_right_hand_l20a_index_contact_torque
```

## 回放 retargeting_data

`retargeting_data` 中每个 session 包含：

- `*_joints.pkl`：21 维 L20a 关节角
- `frame_log.csv`：每帧同步后的 `matched_pressure`
- `metadata.json`：帧率等信息

`matched_pressure` 已按 g 使用，回放脚本会合成 `/cb_right_hand_l20a_contact_cmd` 需要的 22 维数组：

```text
[21 个关节角, matched_pressure]
```

启动 L20a MuJoCo 节点后，在另一个终端播放指定 session：

```bash
ros2 run linker_hand_mujoco_ros2 replay_l20a_contact_cmd \
  src/linker_hand_mujoco_ros2/retargeting_data/linker_l20a_052001
```

也可以直接使用最新 session：

```bash
ros2 run linker_hand_mujoco_ros2 replay_l20a_contact_cmd --latest
```

常用参数：

```bash
ros2 run linker_hand_mujoco_ros2 replay_l20a_contact_cmd \
  src/linker_hand_mujoco_ros2/retargeting_data/linker_l20a_052001 \
  --rate 30 \
  --start-frame 0 \
  --end-frame 300 \
  --loop
```

如果需要临时调整压力数值：

```bash
ros2 run linker_hand_mujoco_ros2 replay_l20a_contact_cmd \
  src/linker_hand_mujoco_ros2/retargeting_data/linker_l20a_052001 \
  --pressure-scale 1.0 \
  --pressure-offset 0.0
```

## 计算方式

节点将接触力 `g` 转成牛顿：

```text
F_N = contact_force_g * 9.80665 / 1000
```

然后以接触面最低点为中心做高斯分配。对每个采样点计算 MuJoCo 平移 Jacobian，并用：

```text
tau_i = J_i^T f_i
```

最后对所有采样点的力矩求和，并提取食指四个关节对应的力矩。

## 离线动力学力矩计算

如果需要使用真实实验采集到的手指姿态和 FSR 压力数据逐帧计算力矩，可以运行：

```bash
conda run -n linker_mujoco --no-capture-output python -m \
  linker_hand_mujoco_ros2.calculate_l20a_dynamics \
  src/linker_hand_mujoco_ros2/retargeting_data/linker_l20a_052001
```

默认输出到 session 目录下：

```text
l20a_dynamics_torque.csv
```

也可以指定帧范围和输出路径：

```bash
conda run -n linker_mujoco --no-capture-output python -m \
  linker_hand_mujoco_ros2.calculate_l20a_dynamics \
  src/linker_hand_mujoco_ros2/retargeting_data/linker_l20a_052001 \
  --start-frame 0 \
  --end-frame 300 \
  --output /tmp/l20a_dynamics_torque.csv
```

CSV 中包含三组食指关节力矩：

- `tau_motion_no_contact_*`：无 FSR 接触力时，复现实验轨迹所需的 MuJoCo 逆动力学力矩
- `tau_contact_fsr_*`：FSR 外力通过 `J^T F` 映射到食指关节的接触力矩
- `tau_required_with_fsr_*`：注入 FSR 外力后，为保持同一实验轨迹所需的驱动力矩，计算方式为 `tau_motion_no_contact - tau_contact_fsr`

脚本会先对关节角做简单移动平均平滑，再由时间戳差分得到 `qvel` 和 `qacc`，然后调用 MuJoCo `mj_inverse()` 计算逆动力学。

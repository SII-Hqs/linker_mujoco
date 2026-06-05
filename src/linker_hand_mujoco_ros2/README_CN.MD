# 1. **概述**

灵心巧手，创造万物。

LinkerHand 灵巧手 ROS2 SDK 是由灵心巧手（北京）科技有限公司开发的一款软件工具，用于驱动其灵巧手系列产品，并提供功能示例。它支持多种设备（如笔记本、台式机、树莓派、Jetson 等），主要服务于人型机器人、工业自动化和科研院所等领域，适用于人型机器人、柔性化生产线、具身大模型训练和数据采集等场景。

# 1.1 **说明**
本程序为LinkerHand制作系列灵巧手Mujoco仿真环境，便于使用者熟悉LinkerHand灵巧手系列产品的使用方式方法，以及进行仿真环境下的模型训练和数据采集

# 2. **使用说明**
```bash
$ mkdir -p linker_hand_mujoco_ros2/src    #创建目录
$ cd linker_hand_mujoco_ros2/src    #进入目录
$ # 1. 克隆仓库（使用 sparse 模式 + blob 过滤，节省空间）
$ git clone --filter=blob:none --sparse https://github.com/linker-bot/linkerhand-sim.git
$ # 2. 进入仓库目录
$ cd linkerhand-sim
$ # 3. 设置 sparse-checkout 目录
$ git sparse-checkout set linker_hand_mujoco_ros2
$ cd linker_hand_mujoco_ros2/src/linker_hand_sim/
$ pip install -r requirements.txt
$ /usr/bin/python3 -m pip install mujoco # 使用ROS2调用mujoco必须使用系统环境下的python3安装
```
- 修改linker_hand_mujoco_ros2/launch/linker_hand_mujoco_ros2.launch.py
根据文件内参数说明修改即可
```bash
$ cd linker_hand_mujoco_ros2/
$ colcon build --symlink-install
$ source ./install/setup.bash
$ ros2 launch linker_hand_mujoco_ros2 linker_hand_mujoco_ros2.launch.py
```

# 3. **topic说明**
- /cb_right_hand_control_cmd or /cb_left_hand_control_cmd
```bash
ros2 topic pub /joint_states sensor_msgs/msg/JointState '
{
  header: { stamp: { sec: 0, nanosec: 0 }, frame_id: "" },
  name: [],
  position: [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
  velocity: [],
  effort: []
}'

  L20: ["拇指根部", "食指根部", "中指根部", "无名指根部","小指根部",
        "拇指侧摆","食指侧摆","中指侧摆","无名指侧摆","小指侧摆",
        "拇指横摆","预留","预留","预留","预留",
        "拇指尖部","食指末端","中指末端","无名指末端","小指末端"]

ros2 topic pub /cb_right_hand_control_cmd sensor_msgs/msg/JointState '
{
  header: { stamp: { sec: 0, nanosec: 0 }, frame_id: "" },
  name: [],
  position: [200.0, 200.0, 200.0, 200.0, 200.0, 
              0, 0, 0, 0, 0, 
              200.0, 200.0, 200.0, 200.0, 200.0, 
              200.0, 200.0, 0, 0, 0],
  velocity: [],
  effort: []
}'

ros2 topic pub /cb_right_hand_contact_cmd std_msgs/msg/Float32MultiArray '
{
  data: [200.0, 200.0, 200.0, 200.0, 200.0,
         0.0, 0.0, 0.0, 0.0, 0.0,
         200.0, 200.0, 200.0, 200.0, 200.0,
         200.0, 200.0, 0.0, 0.0, 0.0,
         50.0]
}'

```
- /cb_right_hand_contact_cmd or /cb_left_hand_contact_cmd
  - `data` 前 20 个数为 L20 控制位置，最后 1 个数为食指接触力，单位 g。
  - 节点会将接触力按高斯分布分配到 `index_link3` 接触面采样点，方向沿世界坐标 `+Z`，并发布食指接触力矩到 `/cb_right_hand_index_contact_torque`。

- position 说明
  L6:  ["大拇指弯曲", "大拇指横摆","食指弯曲", "中指弯曲", "无名指弯曲", "小拇指弯曲"]
  
  L7:  ["大拇指弯曲", "大拇指横摆","食指弯曲", "中指弯曲", "无名指弯曲","小拇指弯曲","拇指旋转"]

  L10: ["拇指根部", "拇指侧摆","食指根部", "中指根部", "无名指根部","小指根部","食指侧摆","无名指侧摆","小指侧摆","拇指旋转"]

  L20: ["拇指根部", "食指根部", "中指根部", "无名指根部","小指根部","拇指侧摆","食指侧摆","中指侧摆","无名指侧摆","小指侧摆","拇指横摆","预留","预留","预留","预留","拇指尖部","食指末端","中指末端","无名指末端","小指末端"]

  L21: ["大拇指根部", "食指根部", "中指根部","无名指根部","小拇指根部","大拇指侧摆","食指侧摆","中指侧摆","无名指侧摆","小拇指侧摆","大拇指横滚","预留","预留","预留","预留","大拇指中部","预留","预留","预留","预留","大拇指指尖","食指指尖","中指指尖","无名指指尖","小拇指指尖"]

  L25: ["大拇指根部", "食指根部", "中指根部","无名指根部","小拇指根部","大拇指侧摆","食指侧摆","中指侧摆","无名指侧摆","小拇指侧摆","大拇指横滚","预留","预留","预留","预留","大拇指中部","食指中部","中指中部","无名指中部","小拇指中部","大拇指指尖","食指指尖","中指指尖","无名指指尖","小拇指指尖"]

# 3.1 **GUI控制**
可以使用 linker_hand_ros2_sdk的[gui_control](https://github.com/linkerbotai/linker_hand_ros2_sdk/blob/main/README_CN.md)控制仿真环境


# 末端指尖接触面估计
python3 src/linker_hand_mujoco_ros2/scripts/visualize_index_contact_patch.py


colcon build --symlink-install
source ./install/setup.bash
ros2 launch linker_hand_mujoco_ros2 linker_hand_mujoco_ros2.launch.py
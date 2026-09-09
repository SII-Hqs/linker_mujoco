# Joint Log 数据处理

以下命令从项目根目录执行：

```bash
cd /home/maker4090/Code/linker_mujoco
conda activate linker_mujoco
```

## 1. 插值到固定频率

```bash
python src/linker_hand_mujoco_ros2/scripts/resample_joint_logs.py \
  results/joint_logs/movement_without_thumb_joint0 \
  --frequency 100 --method pchip --plot \
  -o results/joint_logs/movement_without_thumb_joint0_100hz_pchip
```

`--method` 支持 `linear`、`nearest`、`cubic`、`pchip`、`akima`。脚本插值
`qpos_*`，随后重新计算 `qvel_*` 和 `qacc_*`。对比图保存在输出目录的
`plots/` 中。

## 2. 滤波

```bash
python src/linker_hand_mujoco_ros2/scripts/filter_joint_logs.py \
  results/joint_logs/movement_without_thumb_joint0_100hz_pchip \
  --method butterworth --cutoff-hz 10 --order 4 --plot \
  -o results/joint_logs/movement_without_thumb_joint0_100hz_pchip_filtered
```

`--method` 支持 `butterworth`、`savgol`、`moving-average`、`median`。滤波后
会从新 `qpos_*` 重新计算 `qvel_*` 和 `qacc_*`。Savitzky-Golay、移动平均和
中值滤波使用 `--window`，Savitzky-Golay 还可使用 `--polyorder`。

## 3. 计算逆动力学力矩并绘图

```bash
PYTHONPATH=src/linker_hand_mujoco_ros2 \
python src/linker_hand_mujoco_ros2/linker_hand_mujoco_ros2/tools/calculate_joint_log_torque.py \
  results/joint_logs/movement_without_thumb_joint0_100hz_pchip_filtered \
  --damping 0.8 --plot \
  -o results/joint_torques/movement_without_thumb_joint0_100hz_pchip_filtered
```

输出 CSV 增加 `qacc_*` 和 `tau_inverse_*`。每条 CSV 对应一张 `5x5` 总图，
各小图使用双纵轴显示该关节的角度和力矩，图片位于输出目录的 `plots/`。

力矩脚本默认从 `qpos_*` 重新计算速度和加速度。只有明确需要沿用输入导数时
才使用 `--use-input-derivatives`。

## 4. 平滑力矩并重建关节状态

```bash
PYTHONPATH=src/linker_hand_mujoco_ros2 \
python src/linker_hand_mujoco_ros2/scripts/reconstruct_joint_state_from_torque.py \
  results/joint_torques/movement_without_thumb_joint0_100hz_pchip_filtered \
  --method butterworth --cutoff-hz 2 --order 4 --damping 0.8 --plot \
  -o results/joint_state_reconstruction/movement_without_thumb_joint0_flexion_2hz
```

脚本平滑 `tau_inverse_*`，使用原日志首帧 `qpos/qvel` 作为初始状态，将平滑
力矩作为 MuJoCo 广义力进行前向动力学积分。输出增加 `tau_smoothed_*`、
`qpos_reconstructed_*`、`qvel_reconstructed_*` 和 `qacc_reconstructed_*`。
每条日志生成一张原始/平滑力矩及原始/重建角度联合图。

默认只平滑并绘制 `index/middle/ring/little_joint1-3`，大拇指关节和各手指
`joint0` 的力矩保持不变。使用 `--joint-scope all` 可平滑全部关节，使用
`--plot-all-joints` 可在图中显示全部关节。对于当前 100 Hz 数据，推荐截止
频率为 2 Hz；10 Hz 对已经过角度滤波的数据作用很弱。

这是依赖模型和初始状态的开环动力学重建，并非仅由力矩唯一确定角度的代数
求逆；长时间数据可能因模型误差和力矩平滑产生累计漂移。

## 5. 在末端保持姿态并延长数据

```bash
PYTHONPATH=src/linker_hand_mujoco_ros2 \
python src/linker_hand_mujoco_ros2/scripts/extend_joint_logs.py \
  results/joint_state_reconstruction/movement_without_thumb_joint0_flexion_2hz \
  --duration 2 --plot \
  -o results/joint_state_reconstruction/movement_without_thumb_joint0_flexion_2hz_extended_2s
```

脚本按照输入采样率追加指定时长。新增段保持末帧角度和力矩，`sim_time` 和
`wall_time` 继续递增，所有 `qvel_*`、`qacc_*`（包括重建列）设为零。使用
`--plot` 会在输出目录的 `plots/` 中生成包含延长段的角度/力矩联合图。

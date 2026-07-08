# Task: 在 contact_control_gui_esn.py 中添加食指末端 3D 轨迹实时可视化

## 目标

在 `contact_control_gui_esn.py` 的 `ESNContactControlWindow` GUI 中，新增一个面板，**实时绘制食指末端（`index_link3` body）在世界坐标系下的 3D 运动轨迹**。

## 背景

- 食指运动链为 `index_link0 → index_link1 → index_link2 → index_link3`，其中 **`index_link3` 是最末端 body**（`index_link4` 只是挂在 `index_link3` 上的指尖 geom，不是独立 body）。
- 末端世界坐标通过 MuJoCo 的 `data.xpos[body_id]` 获取，其中 `body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "index_link3")`。
- 仿真对象在 `self.worker.simulation` 中，类型为 `ESNContactSimulation`，包含 `model` 和 `data` 属性。
- GUI 已有一个 `QTimer` 驱动的 `refresh_status()` 方法，每 `refresh_ms`（默认 100ms）刷新一次状态。

## 技术要求

### 使用 pyqtgraph（项目已有依赖，见 requirements.txt）

- 使用 `pyqtgraph.opengl.GLViewWidget` + `GLLinePlotItem` 绘制 3D 轨迹曲线。
- **不要用 matplotlib**，项目不依赖它，且 pyqtgraph 性能更好、与 PyQt5 集成更自然。

### 新增 widget 类：`TrajectoryWidget(QGroupBox)`

- 标题："末端轨迹 (index_link3)"
- 内部包含一个 `GLViewWidget` 用于 3D 显示。
- 维护一个 `collections.deque`（`maxlen=2000`）存储历史位置点。
- 提供 `update_position(pos: np.ndarray)` 方法，接收一个 3D 坐标，追加到 deque 并刷新曲线。
- 轨迹曲线用颜色渐变表示时间先后：旧的点用暗色（如深蓝），新的点用亮色（如亮青色/白色）。
- 同时绘制一个醒目的小球标记当前末端位置。
- 提供 `clear_trajectory()` 方法清空轨迹。

### 集成到 ESNContactControlWindow

1. 在 `__init__` 中：
   - 用 `mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "index_link3")` 获取 body id，存为 `self.tip_body_id`。
2. 在 `_build_content()` 中：
   - 在 `ReservoirHeatmapWidget` 下方（`root_layout.addWidget(self.reservoir_widget)` 之后）插入 `self.trajectory_widget = TrajectoryWidget()`。
   - 在按钮行中新增一个 "清除轨迹" 按钮，连接到 `self.trajectory_widget.clear_trajectory()`。
3. 在 `refresh_status()` 中：
   - 读取 `sim.data.xpos[self.tip_body_id].copy()`，调用 `self.trajectory_widget.update_position(pos)`。

### 代码风格

- 严格遵循现有文件的代码风格：类命名用 PascalCase，方法用 snake_case，docstring 用三引号单行。
- import 按标准库 / 第三方 / 本地分组，与文件现有 import 风格一致。
- 新增的 import：`import collections`、`import pyqtgraph.opengl as gl`。
- 不要修改任何现有功能的行为，只做纯增量添加。

## 不要做的事

- 不要修改 `contact_control_demo_esn.py`、`contact_control_gui.py` 或其他文件。
- 不要添加新的命令行参数。
- 不要引入 matplotlib 或任何不在 requirements.txt 中的新依赖。

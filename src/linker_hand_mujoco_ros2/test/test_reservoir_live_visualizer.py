import importlib.util
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "deployment" / "reservoir_live_visualizer.py"


def load_visualizer_module():
    spec = importlib.util.spec_from_file_location(
        "test_reservoir_live_visualizer_module", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    def is_alive(self):
        return False


def test_put_latest_replaces_stale_snapshot():
    module = load_visualizer_module()
    snapshots = queue.Queue(maxsize=1)
    snapshots.put_nowait("old")

    module._put_latest(snapshots, "new")

    assert snapshots.get_nowait() == "new"


def test_update_keeps_100_hz_rolling_window_without_starting_gui():
    module = load_visualizer_module()
    visualizer = module.LiveReservoirVisualizer.__new__(
        module.LiveReservoirVisualizer
    )
    visualizer.visualization_fps = 15.0
    visualizer.window_seconds = 0.05
    visualizer.publish_period = 1.0 / visualizer.visualization_fps
    visualizer.last_publish_wall = 0.0
    visualizer.times = module.deque(maxlen=5)
    visualizer.current_qpos = module.deque(maxlen=5)
    visualizer.future_target = module.deque(maxlen=5)
    visualizer.states = module.deque(maxlen=5)
    visualizer.saturation = module.deque(maxlen=5)
    visualizer.process = FakeProcess()
    visualizer.queue = queue.Queue(maxsize=2)

    for frame in range(8):
        visualizer.update(
            frame / 100.0,
            np.full(12, frame),
            np.full(12, frame + 1),
            np.full(7, frame / 10.0),
            np.asarray([0.0, 0.96, -0.97, 0.3, 0.2, 0.1, 0.0]),
            1.2,
            np.full(12, 0.1),
        )

    assert np.allclose(visualizer.times, [0.03, 0.04, 0.05, 0.06, 0.07])
    assert len(visualizer.states) == 5
    assert np.isclose(visualizer.saturation[-1], 2.0 / 7.0)


def test_visualization_constructor_rejects_non_positive_values():
    module = load_visualizer_module()
    controller = SimpleNamespace(sample_rate_hz=100.0)

    for fps, window in ((0.0, 2.5), (15.0, 0.0)):
        try:
            module.LiveReservoirVisualizer(
                controller, None, visualization_fps=fps, window_seconds=window
            )
        except ValueError as exc:
            assert "must be positive" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_deployment_parser_accepts_live_visualization_options():
    deployment_path = PACKAGE_ROOT / "deployment" / "future_pose_torque_mujoco.py"
    spec = importlib.util.spec_from_file_location(
        "test_live_visualization_deployment_parser", deployment_path
    )
    deployment = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = deployment
    spec.loader.exec_module(deployment)

    args = deployment.parse_args(
        [
            "--live-visualization",
            "--visualization-fps",
            "20",
            "--visualization-window-seconds",
            "3.5",
            "--visualization-backend",
            "QtAgg",
        ]
    )

    assert args.live_visualization is True
    assert args.visualization_fps == 20.0
    assert args.visualization_window_seconds == 3.5
    assert args.visualization_backend == "QtAgg"

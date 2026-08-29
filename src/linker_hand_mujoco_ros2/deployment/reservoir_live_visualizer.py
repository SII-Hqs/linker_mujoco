#!/usr/bin/env python3
"""Non-blocking live visualization for a future-pose torque reservoir."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from collections import deque

import numpy as np


JOINT_COLORS = ("#1677b8", "#dc7b24", "#258d5b")
TIME_CURSOR_COLOR = "#202020"


def _scalar(params, name, default):
    return np.asarray(params[name]).item() if name in params else default


def _put_latest(target_queue, payload) -> None:
    try:
        target_queue.put_nowait(payload)
        return
    except queue.Full:
        pass
    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        target_queue.put_nowait(payload)
    except queue.Full:
        pass


class LiveReservoirVisualizer:
    """Buffer 100 Hz controller state and publish snapshots to a GUI process."""

    def __init__(
        self,
        controller,
        reference,
        visualization_fps=15.0,
        window_seconds=2.5,
        backend="QtAgg",
    ):
        self.visualization_fps = float(visualization_fps)
        self.window_seconds = float(window_seconds)
        if self.visualization_fps <= 0.0 or self.window_seconds <= 0.0:
            raise ValueError("visualization FPS and window seconds must be positive")

        self.publish_period = 1.0 / self.visualization_fps
        self.last_publish_wall = -self.publish_period
        sample_count = max(
            2, int(round(self.window_seconds * controller.sample_rate_hz))
        )
        self.times = deque(maxlen=sample_count)
        self.current_qpos = deque(maxlen=sample_count)
        self.future_target = deque(maxlen=sample_count)
        self.states = deque(maxlen=sample_count)
        self.saturation = deque(maxlen=sample_count)

        params = controller.params
        config = {
            "candidate_id": int(_scalar(params, "candidate_id", -1)),
            "screening_status": str(
                _scalar(params, "screening_status", "unclassified")
            ),
            "reservoir_size": int(controller.state.size),
            "spectral_radius": float(_scalar(params, "spectral_radius", np.nan)),
            "leak_rate": float(controller.dt),
            "input_scaling": float(_scalar(params, "input_scaling", np.nan)),
            "joint_names": list(controller.joint_names),
            "sample_rate_hz": float(controller.sample_rate_hz),
            "future_horizon": float(controller.future_horizon),
            "visualization_fps": self.visualization_fps,
            "window_seconds": self.window_seconds,
            "backend": str(backend),
            "parent_pid": os.getpid(),
            "reference_time": (
                None if reference is None else np.asarray(reference.time, dtype=float)
            ),
            "reference_qpos": (
                None if reference is None else np.asarray(reference.qpos, dtype=float)
            ),
        }
        self.context = mp.get_context("spawn")
        self.queue = self.context.Queue(maxsize=2)
        self.process = self.context.Process(
            target=_visualization_process,
            args=(self.queue, config),
            name="reservoir-live-visualization",
        )
        self.process.start()

    def update(
        self,
        sim_time,
        current_qpos,
        future_target,
        state,
        activation,
        max_abs_input_zscore,
        raw_torque,
    ) -> None:
        current = np.asarray(current_qpos, dtype=float).reshape(-1).copy()
        target = np.asarray(future_target, dtype=float).reshape(-1).copy()
        reservoir_state = np.asarray(state, dtype=float).reshape(-1).copy()
        reservoir_activation = np.asarray(activation, dtype=float).reshape(-1)
        self.times.append(float(sim_time))
        self.current_qpos.append(current)
        self.future_target.append(target)
        self.states.append(reservoir_state)
        self.saturation.append(float(np.mean(np.abs(reservoir_activation) > 0.95)))

        if not self.process.is_alive():
            return
        now = time.monotonic()
        if now - self.last_publish_wall < self.publish_period:
            return
        self.last_publish_wall = now
        snapshot = {
            "time": np.asarray(self.times, dtype=float),
            "current_qpos": np.asarray(self.current_qpos, dtype=float),
            "future_target": np.asarray(self.future_target, dtype=float),
            "states": np.asarray(self.states, dtype=float),
            "saturation": np.asarray(self.saturation, dtype=float),
            "max_abs_input_zscore": float(max_abs_input_zscore),
            "max_abs_raw_torque": float(
                np.max(np.abs(np.asarray(raw_torque, dtype=float)))
            ),
        }
        _put_latest(self.queue, snapshot)

    def close(self) -> None:
        if self.process.is_alive():
            _put_latest(self.queue, None)
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)
        self.queue.close()
        self.queue.join_thread()


def _visualization_process(state_queue, config) -> None:
    try:
        import matplotlib

        matplotlib.use(config["backend"], force=True)
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import LineCollection
    except Exception as exc:
        print(f"Live reservoir visualization failed to start: {exc}", flush=True)
        return

    if str(plt.get_backend()).lower() == "agg":
        print(
            "Live reservoir visualization requires an interactive backend; "
            "use QtAgg or TkAgg.",
            flush=True,
        )
        return

    joint_names = config["joint_names"]
    reservoir_size = int(config["reservoir_size"])
    window_seconds = float(config["window_seconds"])
    reference_time = config["reference_time"]
    reference_qpos = config["reference_qpos"]

    figure = plt.figure(figsize=(14, 12), constrained_layout=True)
    grid = figure.add_gridspec(4, 4, height_ratios=(1.0, 0.82, 0.65, 0.20))
    joint_axes = [figure.add_subplot(grid[0, column]) for column in range(4)]
    state_axis = figure.add_subplot(grid[1, :])
    trace_axis = figure.add_subplot(grid[2, :], sharex=state_axis)
    saturation_axis = figure.add_subplot(grid[3, :], sharex=state_axis)

    actual_lines = []
    current_markers = []
    target_markers = []
    cursor_lines = []
    finger_names = ("index", "middle", "ring", "little")
    for finger_index, (finger, axis) in enumerate(zip(finger_names, joint_axes)):
        base = finger_index * 3
        for local_index in range(3):
            joint_index = base + local_index
            color = JOINT_COLORS[local_index]
            if reference_time is not None:
                axis.plot(
                    reference_time,
                    reference_qpos[:, joint_index],
                    color=color,
                    linewidth=1.0,
                    alpha=0.35,
                )
            actual_line, = axis.plot(
                [], [], color=color, linewidth=1.5, label=f"joint{local_index + 1}"
            )
            current, = axis.plot(
                [], [], marker="o", color=color, markersize=5, linestyle="none"
            )
            target, = axis.plot(
                [], [], marker="x", color=color, markersize=6,
                linestyle="none", markeredgewidth=1.4
            )
            actual_lines.append(actual_line)
            current_markers.append(current)
            target_markers.append(target)
        cursor_lines.append(
            axis.axvline(0.0, color=TIME_CURSOR_COLOR, linewidth=1.2, alpha=0.9)
        )
        axis.set_title(finger)
        axis.set_xlabel("Time [s]")
        if finger_index == 0:
            axis.set_ylabel("Joint angle [rad]")
        axis.grid(alpha=0.18)
        axis.legend(loc="upper right", fontsize=8)
        if reference_time is not None:
            axis.set_xlim(float(reference_time[0]), float(reference_time[-1]))

    candidate = config["candidate_id"]
    candidate_label = f"Candidate {candidate}" if candidate >= 0 else "ESN"
    figure.suptitle(
        f"{candidate_label} | {config['screening_status'].upper()} | "
        f"N={reservoir_size}, SR={config['spectral_radius']:.3f}, "
        f"leak={config['leak_rate']:.3f}, input scale={config['input_scaling']:.3f}\n"
        f"circle: actual q(t)   cross: target q(t+{config['future_horizon']:g}s)",
        fontsize=13,
    )

    image = state_axis.imshow(
        np.zeros((reservoir_size, 2)),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        extent=(-window_seconds, 0.0, 0, reservoir_size),
    )
    state_axis.set_ylabel("Reservoir neuron")
    state_axis.set_title("Rolling reservoir state x_i(t)")
    colorbar = figure.colorbar(image, ax=state_axis, pad=0.01, fraction=0.025)
    colorbar.set_label("Neuron state")

    trace_collection = LineCollection(
        [],
        colors=plt.colormaps["turbo"](np.linspace(0.04, 0.96, reservoir_size)),
        linewidths=0.55,
        alpha=0.34,
        rasterized=True,
    )
    trace_axis.add_collection(trace_collection)
    trace_axis.set_ylim(-1.05, 1.05)
    trace_axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    trace_axis.axhline(0.0, color="0.35", linewidth=0.7, alpha=0.6)
    trace_axis.set_ylabel("Neuron state")
    trace_axis.set_title("All reservoir neuron trajectories x_i(t)")
    trace_axis.grid(alpha=0.15)

    saturation_line, = saturation_axis.plot([], [], color="#c43c39", linewidth=1.2)
    saturation_axis.set_ylim(-0.02, 1.02)
    saturation_axis.axhline(0.30, color="black", linestyle="--", linewidth=0.8)
    saturation_axis.set_ylabel("Saturation")
    saturation_axis.set_xlabel("Rolling physical time [s]")
    saturation_axis.grid(alpha=0.15)
    state_cursor = state_axis.axvline(0.0, color=TIME_CURSOR_COLOR, linewidth=1.4)
    trace_cursor = trace_axis.axvline(0.0, color=TIME_CURSOR_COLOR, linewidth=1.4)
    saturation_cursor = saturation_axis.axvline(
        0.0, color=TIME_CURSOR_COLOR, linewidth=1.4
    )
    info = state_axis.text(
        0.01,
        0.98,
        "Waiting for controller data...",
        transform=state_axis.transAxes,
        ha="left",
        va="top",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.88, "pad": 4},
    )
    latest = None
    close_timer = None

    def request_close():
        nonlocal close_timer
        if close_timer is not None:
            return
        close_timer = figure.canvas.new_timer(interval=1)
        close_timer.single_shot = True
        close_timer.add_callback(plt.close, figure)
        close_timer.start()

    def drain_queue():
        nonlocal latest
        while True:
            try:
                payload = state_queue.get_nowait()
            except queue.Empty:
                return True
            if payload is None:
                request_close()
                return False
            latest = payload

    def update(_frame):
        if os.getppid() != config["parent_pid"]:
            request_close()
            return ()
        if not drain_queue() or latest is None:
            return ()
        times = latest["time"]
        current = latest["current_qpos"]
        target = latest["future_target"]
        states = latest["states"]
        saturation = latest["saturation"]
        now = float(times[-1])
        start = float(times[0])
        left = now - window_seconds
        right = now + 0.06 * window_seconds

        for joint_index, line in enumerate(actual_lines):
            line.set_data(times, current[:, joint_index])
            current_markers[joint_index].set_data(
                [now], [current[-1, joint_index]]
            )
            target_markers[joint_index].set_data([now], [target[-1, joint_index]])
        for cursor in cursor_lines:
            cursor.set_xdata([now, now])

        image_left = start if now > start else now - 1.0 / config["sample_rate_hz"]
        image.set_data(states.T)
        image.set_extent((image_left, now, 0, reservoir_size))
        trace_collection.set_segments(
            [
                np.column_stack((times, states[:, neuron_index]))
                for neuron_index in range(reservoir_size)
            ]
        )
        saturation_line.set_data(times, saturation)
        state_axis.set_xlim(left, right)
        saturation_axis.set_xlim(left, right)
        state_cursor.set_xdata([now, now])
        trace_cursor.set_xdata([now, now])
        saturation_cursor.set_xdata([now, now])
        info.set_text(
            f"t={now:6.2f} s\n"
            f"instant saturation={saturation[-1]:.1%}\n"
            f"max |input z|={latest['max_abs_input_zscore']:.2f}   "
            f"max |raw tau|={latest['max_abs_raw_torque']:.3f} N m"
        )
        return (image, trace_collection, saturation_line, info)

    interval_ms = max(int(round(1000.0 / config["visualization_fps"])), 1)
    animation = FuncAnimation(
        figure,
        update,
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )
    _ = animation
    if figure.canvas.manager is not None:
        figure.canvas.manager.set_window_title(f"{candidate_label} reservoir dynamics")
    print(
        f"Live reservoir visualization started with {plt.get_backend()} at "
        f"{config['visualization_fps']:g} FPS.",
        flush=True,
    )
    try:
        plt.show()
    except Exception as exc:
        print(f"Live reservoir visualization failed: {exc}", flush=True)


__all__ = ["LiveReservoirVisualizer"]

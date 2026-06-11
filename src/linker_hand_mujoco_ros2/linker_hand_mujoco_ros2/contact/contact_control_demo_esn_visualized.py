#!/usr/bin/env python3
"""Run the ESN contact-control demo with live reservoir visualization."""

import argparse
import multiprocessing as mp
import queue
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

if __package__:
    from .contact_control_demo_esn import add_esn_arguments, build_esn_simulation
else:
    from pathlib import Path

    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo_esn import (
        add_esn_arguments,
        build_esn_simulation,
    )


def _is_noninteractive_agg_backend(backend):
    """Return True only for the non-interactive Agg backend, not QtAgg/TkAgg."""
    normalized = backend.lower()
    return normalized == "agg" or normalized.endswith("backend_agg")


def _reservoir_heatmap_process(state_queue, nr, rows, cols, vis_fps, vis_backend):
    """Run the matplotlib reservoir heatmap in its own GUI process."""
    try:
        import matplotlib

        backend_candidates = []
        if vis_backend:
            backend_candidates.append(vis_backend)
        backend_candidates.extend(["QtAgg", "TkAgg"])

        backend_errors = []
        for backend in backend_candidates:
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception as exc:
                backend_errors.append(f"{backend}: {exc}")
        else:
            print(
                "Reservoir heatmap could not select an interactive backend:\n"
                + "\n".join(backend_errors),
                flush=True,
            )
            return

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if _is_noninteractive_agg_backend(plt.get_backend()):
            print(
                "Reservoir heatmap needs an interactive matplotlib backend; "
                f"current backend is {plt.get_backend()}. "
                "Try --vis-backend QtAgg or --vis-backend TkAgg.",
                flush=True,
            )
            return
        print(
            f"Reservoir heatmap using matplotlib backend: {plt.get_backend()}",
            flush=True,
        )
    except Exception as exc:
        print(f"Reservoir heatmap failed to start: {exc}", flush=True)
        return

    try:
        grid = np.zeros((rows, cols), dtype=float)
        sim_time = 0.0
        step_count = 0

        fig, ax = plt.subplots()
        image = ax.imshow(
            grid,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_xlabel("Reservoir column")
        ax.set_ylabel("Reservoir row")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Activation")
        fig.tight_layout()

        def drain_queue():
            nonlocal grid, sim_time, step_count
            while True:
                try:
                    item = state_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    plt.close(fig)
                    return

                sim_time, step_count, flat_state = item
                padded = np.zeros(rows * cols, dtype=float)
                flat_state = np.asarray(flat_state, dtype=float).reshape(-1)
                padded[: min(nr, flat_state.size)] = flat_state[:nr]
                grid = padded.reshape(rows, cols)

        def animate(_frame):
            drain_queue()
            image.set_data(grid)
            title = (
                f"ESN Reservoir t={sim_time:6.3f}s "
                f"step={step_count} "
                f"min={grid.min(): .3f} max={grid.max(): .3f}"
            )
            ax.set_title(title)
            if fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title(title)
            return (image,)

        interval_ms = max(int(1000.0 / max(float(vis_fps), 1e-6)), 1)
        animation = FuncAnimation(
            fig,
            animate,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        _ = animation
        plt.show()
    except Exception as exc:
        print(f"Reservoir heatmap window failed: {exc}", flush=True)


class ReservoirHeatmap:
    """Send ESN reservoir activations to a live heatmap process."""

    def __init__(self, esn, vis_fps, vis_backend):
        self.esn = esn
        self.vis_period = 1.0 / max(float(vis_fps), 1e-6)
        self.last_update_wall = -self.vis_period
        self.nr = int(esn.nr)
        self.rows = int(np.ceil(np.sqrt(self.nr)))
        self.cols = int(np.ceil(self.nr / self.rows))
        self.context = mp.get_context("spawn")
        self.queue = self.context.Queue(maxsize=3)
        self.process = self.context.Process(
            target=_reservoir_heatmap_process,
            args=(
                self.queue,
                self.nr,
                self.rows,
                self.cols,
                vis_fps,
                vis_backend,
            ),
        )
        self.process.start()
        self.update(0.0, force=True)

    def update(self, sim_time, force=False):
        """Publish the latest reservoir state at the visualization frame rate."""
        if not self.process.is_alive():
            return

        now = time.time()
        if not force and now - self.last_update_wall < self.vis_period:
            return
        self.last_update_wall = now

        payload = (
            float(sim_time),
            int(self.esn.step_count),
            np.asarray(self.esn.x, dtype=float).reshape(-1).copy(),
        )
        self._put_latest(payload)

    def _put_latest(self, payload):
        try:
            self.queue.put_nowait(payload)
            return
        except queue.Full:
            pass

        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            pass

    def close(self):
        """Ask the heatmap process to close and clean up if it is still alive."""
        if not self.process.is_alive():
            return
        self._put_latest(None)
        self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)


def run_demo(args):
    """Run the ESN-control demo with MuJoCo viewer and reservoir heatmap."""
    simulation = build_esn_simulation(args)
    reservoir_view = ReservoirHeatmap(
        simulation.controller.controller,
        args.vis_fps,
        args.vis_backend,
    )

    print("ESN contact-control demo with reservoir visualization started.")
    print(f"model: {args.model_xml}")
    print(f"esn: {args.esn_model}")
    print(f"target: {args.target}")
    print(f"horizon: {args.horizon}")
    print(f"vis_fps: {args.vis_fps:.2f}")
    print(f"vis_backend: {args.vis_backend or 'auto'}")
    print("Press Ctrl+C in the terminal to stop.", flush=True)

    last_log = -args.log_interval

    def step_once():
        nonlocal last_log
        contact_state = simulation.step()
        if simulation.data.time - last_log >= args.log_interval:
            last_log = simulation.data.time
            names = ",".join(contact_state.geoms) if contact_state.geoms else "none"
            tau = simulation.controller.last_tau
            tau_str = ",".join(f"{t:7.4f}" for t in tau)
            print(
                f"t={simulation.data.time:6.3f}s "
                f"contact={int(contact_state.in_contact)} "
                f"normal_force={contact_state.normal_force:8.4f}N "
                f"raw={contact_state.raw_normal_force:8.4f}N "
                f"tau=[{tau_str}] geoms={names}",
                flush=True,
            )

    try:
        if args.headless:
            end_time = simulation.data.time + args.duration
            while simulation.data.time < end_time:
                step_once()
                reservoir_view.update(simulation.data.time)
            return

        with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                with viewer.lock():
                    step_once()
                viewer.sync()
                reservoir_view.update(simulation.data.time)
                sleep_time = simulation.model.opt.timestep - (time.time() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        reservoir_view.close()


def add_visualized_esn_arguments(parser):
    """Add ESN visualization demo arguments to an argument parser."""
    add_esn_arguments(parser)
    parser.add_argument(
        "--vis-fps",
        type=float,
        default=15.0,
        help="Reservoir heatmap refresh rate in Hz.",
    )
    parser.add_argument(
        "--vis-backend",
        default="QtAgg",
        help="Matplotlib backend for the reservoir heatmap window.",
    )
    return parser


def parse_args(argv):
    """Parse ESN visualization demo command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an L20a hand/platform MuJoCo ESN torque-control demo "
            "with live reservoir visualization."
        )
    )
    add_visualized_esn_arguments(parser)
    parser.add_argument("--log-interval", type=float, default=0.25)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=6.0)
    return parser.parse_args(argv)


def main():
    """Run the ESN torque-control demo with live reservoir visualization."""
    args = parse_args(sys.argv[1:])
    run_demo(args)


if __name__ == "__main__":
    main()

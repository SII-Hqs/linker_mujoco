"""Save and restore MuJoCo passive viewer free-camera state."""

import json
import threading
import time
from pathlib import Path

import mujoco.viewer


def camera_state_from_viewer(viewer):
    """Return a JSON-serializable free-camera state."""
    cam = viewer.cam
    return {
        "lookat": [float(value) for value in cam.lookat],
        "distance": float(cam.distance),
        "azimuth": float(cam.azimuth),
        "elevation": float(cam.elevation),
    }


def apply_camera_state(viewer, state):
    """Apply a saved free-camera state to a viewer."""
    cam = viewer.cam
    cam.lookat[:] = state["lookat"]
    cam.distance = float(state["distance"])
    cam.azimuth = float(state["azimuth"])
    cam.elevation = float(state["elevation"])


class ViewerCameraMixin:
    """Mixin for SimulationWorker subclasses with camera save/load support."""

    def _init_camera_controls(self, camera_path=None):
        """Initialize viewer camera state shared with the Qt thread."""
        self.camera_path = Path(camera_path or "mujoco_camera_view.json")
        self.viewer = None
        self.viewer_handle_lock = threading.Lock()
        self.pending_camera_state = None

    def current_camera_state(self):
        """Return current viewer camera state or None when viewer is closed."""
        with self.viewer_handle_lock:
            viewer = self.viewer
        if viewer is None:
            return None
        with viewer.lock():
            return camera_state_from_viewer(viewer)

    def save_camera(self, path=None):
        """Save current viewer camera state to JSON."""
        state = self.current_camera_state()
        if state is None:
            raise RuntimeError("MuJoCo viewer is not running")
        camera_path = Path(path or self.camera_path)
        camera_path.parent.mkdir(parents=True, exist_ok=True)
        with camera_path.open("w", encoding="utf-8") as file:
            json.dump(state, file, indent=2)
            file.write("\n")
        return camera_path

    def load_camera(self, path=None):
        """Load viewer camera state from JSON and apply it if possible."""
        camera_path = Path(path or self.camera_path)
        with camera_path.open("r", encoding="utf-8") as file:
            state = json.load(file)
        with self.viewer_handle_lock:
            viewer = self.viewer
        if viewer is None:
            self.pending_camera_state = state
        else:
            with viewer.lock():
                apply_camera_state(viewer, state)
        return camera_path

    def _draw_viewer_overlays(self, viewer):
        """Draw extra viewer overlays after each simulation step."""

    def _run_with_viewer(self):
        with mujoco.viewer.launch_passive(
            self.simulation.model,
            self.simulation.data,
        ) as viewer:
            with self.viewer_handle_lock:
                self.viewer = viewer
            try:
                if self.pending_camera_state is not None:
                    with viewer.lock():
                        apply_camera_state(viewer, self.pending_camera_state)
                    self.pending_camera_state = None
                while not self.stop_event.is_set() and viewer.is_running():
                    step_start = time.time()
                    with viewer.lock():
                        self._step_once()
                        self._draw_viewer_overlays(viewer)
                    viewer.sync()
                    self._sleep_to_realtime(step_start)
            finally:
                with self.viewer_handle_lock:
                    self.viewer = None
        self.stop_event.set()

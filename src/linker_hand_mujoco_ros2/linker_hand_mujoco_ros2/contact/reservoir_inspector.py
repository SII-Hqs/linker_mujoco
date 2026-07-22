"""On-demand PyQt5 inspector for current-info ESN recurrence snapshots."""

import collections

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


DETAIL_FIELDS = (
    "input_term",
    "recurrent_term",
    "bias_term",
    "pre_activation",
    "tanh_value",
    "x_old",
    "x_dot",
    "x_base_next",
    "raw_correction",
    "k",
    "scaled_correction",
    "x_before_zeroing",
    "configured_extreme_zero_n",
    "effective_extreme_zero_n",
    "force_raw",
    "force_gate_active",
    "x_next",
)

SCALAR_FIELDS = {
    "k",
    "configured_extreme_zero_n",
    "effective_extreme_zero_n",
    "force_raw",
    "force_gate_active",
}


class ReservoirInspectorWindow(QMainWindow):
    """Inspect every captured current-info ESN recurrence on demand."""

    def __init__(self, worker, owner=None, history_limit=300, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.owner = owner
        self.history = collections.deque(maxlen=int(history_limit))
        self.detail_labels = {}
        self.slow_mode = False
        self.previous_speed = None
        self.debug_active = False

        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("ESN 神经元递推检查器")
        self.setMinimumSize(1120, 680)
        self.setCentralWidget(self._build_content())

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_from_controller)
        self.refresh_timer.start(50)
        self._sync_debug_state()

    @property
    def current_esn(self):
        return self.worker.simulation.controller.current_controller

    def _build_content(self):
        content = QWidget()
        root_layout = QVBoxLayout(content)

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Neuron index:"))
        self.neuron_input = QSpinBox()
        self.neuron_input.setRange(0, int(self.current_esn.nr) - 1)
        self.neuron_input.valueChanged.connect(self.update_display)
        control_layout.addWidget(self.neuron_input)

        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.pause_simulation)
        control_layout.addWidget(self.pause_button)
        self.slow_button = QPushButton("慢速连续运行 0.05x")
        self.slow_button.clicked.connect(self.start_slow_run)
        control_layout.addWidget(self.slow_button)
        self.step_button = QPushButton("前进一个 ESN 步骤")
        self.step_button.clicked.connect(self.step_one_esn_update)
        control_layout.addWidget(self.step_button)
        self.clear_button = QPushButton("清除检查历史")
        self.clear_button.clicked.connect(self.clear_history)
        control_layout.addWidget(self.clear_button)
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.close)
        control_layout.addWidget(self.close_button)
        root_layout.addLayout(control_layout)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)

        display_layout = QHBoxLayout()
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumHeight(300)
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("bottom", "ESN step")
        self.plot_widget.setLabel("left", "Selected neuron value")
        self.plot_widget.addLegend(offset=(8, 8))
        self.curves = {
            "x_old": self.plot_widget.plot(
                pen=pg.mkPen(color=(0, 114, 178), width=2), name="x_old"
            ),
            "x_base_next": self.plot_widget.plot(
                pen=pg.mkPen(color=(230, 159, 0), width=2), name="x_base_next"
            ),
            "scaled_correction": self.plot_widget.plot(
                pen=pg.mkPen(color=(204, 121, 167), width=2),
                name="scaled_correction",
            ),
            "x_next": self.plot_widget.plot(
                pen=pg.mkPen(color=(0, 158, 115), width=2), name="x_next"
            ),
        }
        self.plot_widget.addItem(
            pg.InfiniteLine(
                pos=0.0,
                angle=0,
                pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DashLine),
            )
        )
        left_layout.addWidget(self.plot_widget)

        detail_layout = QFormLayout()
        for field in DETAIL_FIELDS:
            label = QLabel("-")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.detail_labels[field] = label
            detail_layout.addRow(f"{field}:", label)
        left_layout.addLayout(detail_layout)
        display_layout.addWidget(left_panel, 3)

        ranking_group = QGroupBox("当前步骤全部神经元（按 x_next 从大到小）")
        ranking_layout = QVBoxLayout(ranking_group)
        ranking_group.setTitle("All neurons ranked by x_next (largest to smallest)")
        self.ranking_step_label = QLabel("ESN step: -")
        ranking_layout.addWidget(self.ranking_step_label)
        self.ranking_table = QTableWidget(int(self.current_esn.nr), 4)
        self.ranking_table.setHorizontalHeaderLabels(
            ["排名", "神经元序号", "x_next 激活值", "交换对象"]
        )
        self.ranking_table.setHorizontalHeaderLabels(
            ["Rank", "Neuron", "x_next", "Zeroed this step"]
        )
        self.ranking_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.ranking_table.verticalHeader().setVisible(False)
        self.ranking_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ranking_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ranking_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ranking_table.cellClicked.connect(self.select_ranked_neuron)
        for row in range(int(self.current_esn.nr)):
            for column in range(4):
                item = QTableWidgetItem("-")
                item.setTextAlignment(Qt.AlignCenter)
                self.ranking_table.setItem(row, column, item)
        ranking_layout.addWidget(self.ranking_table)
        display_layout.addWidget(ranking_group, 2)
        root_layout.addLayout(display_layout)
        return content

    def _supported(self):
        return int(self.worker.simulation.controller.horizon) == 0

    def _sync_debug_state(self):
        supported = self._supported()
        self.neuron_input.setEnabled(supported)
        self.slow_button.setEnabled(supported)
        self.step_button.setEnabled(supported)
        if supported and not self.debug_active:
            self.current_esn.set_debug_enabled(True)
            self.debug_active = True
            self.status_label.setText("正在记录 Horizon=0 的每一次 60 Hz ESN递推。")
        elif not supported:
            if self.debug_active:
                self.current_esn.set_debug_enabled(False)
                self.debug_active = False
            self.status_label.setText("不支持：神经元递推检查器只用于 Horizon=0 current-info ESN。")
        return supported

    def refresh_from_controller(self):
        if not self._sync_debug_state():
            return
        snapshots = self.current_esn.drain_debug_snapshots()
        if snapshots:
            self.history.extend(snapshots)
            self.update_display()
        state = "暂停" if self.worker.is_paused() else "运行"
        self.status_label.setText(
            f"Horizon=0 | {state} | 已显示 {len(self.history)} 个ESN步骤"
        )

    def _neuron_value(self, snapshot, field, index):
        value = snapshot[field]
        if field in SCALAR_FIELDS:
            return float(value)
        return float(np.asarray(value, dtype=float).reshape(-1)[index])

    def update_display(self, *_args):
        if not self.history:
            return
        index = int(self.neuron_input.value())
        steps = np.asarray([item["step_count"] for item in self.history], dtype=float)
        for field, curve in self.curves.items():
            values = np.asarray(
                [self._neuron_value(item, field, index) for item in self.history],
                dtype=float,
            )
            curve.setData(steps, values)
        if steps.size:
            left = max(float(steps[-1]) - 299.0, float(steps[0]))
            self.plot_widget.setXRange(left, max(left + 1.0, float(steps[-1])), padding=0.02)

        latest = self.history[-1]
        for field, label in self.detail_labels.items():
            label.setText(f"{self._neuron_value(latest, field, index): .8f}")
        self.update_zeroing_ranking(latest)

    def update_zeroing_ranking(self, snapshot):
        """Rank x_next and mark neurons zeroed by the current controller."""
        values = np.asarray(snapshot["x_next"], dtype=float).reshape(-1)
        if values.size != int(self.current_esn.nr):
            raise ValueError(
                f"Expected {self.current_esn.nr} reservoir values, got {values.size}"
            )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Cannot rank non-finite reservoir values")

        order = np.argsort(-values, kind="stable")
        zeroed_indices = set(
            int(value)
            for value in np.asarray(
                snapshot["zeroed_indices"], dtype=int
            ).reshape(-1)
        )
        zeroed_text = (
            ", ".join(str(value) for value in sorted(zeroed_indices))
            if zeroed_indices
            else "none"
        )
        self.ranking_step_label.setText(
            f"ESN step: {int(snapshot['step_count'])} | "
            f"configured n={int(snapshot['configured_extreme_zero_n'])} | "
            f"effective n={int(snapshot['effective_extreme_zero_n'])} | "
            f"F={float(snapshot['force_raw']):.4f} N | "
            f"gate={'ON' if snapshot['force_gate_active'] else 'OFF'} | "
            f"zeroed={zeroed_text}"
        )
        for row, neuron_index in enumerate(order):
            neuron_index = int(neuron_index)
            value = float(values[neuron_index])
            was_zeroed = neuron_index in zeroed_indices
            self.ranking_table.item(row, 0).setText(str(row + 1))
            self.ranking_table.item(row, 1).setText(str(neuron_index))
            self.ranking_table.item(row, 2).setText(f"{value:+.8f}")
            self.ranking_table.item(row, 3).setText(
                "yes" if was_zeroed else ""
            )

            strength = min(abs(value), 1.0)
            if was_zeroed:
                color = QColor(255, 247, 204)
            elif value > 0.0:
                color = QColor(
                    255, int(250 - 55 * strength), int(250 - 55 * strength)
                )
            elif value < 0.0:
                color = QColor(
                    int(250 - 55 * strength), int(250 - 25 * strength), 255
                )
            else:
                color = QColor(255, 255, 255)
            for column in range(4):
                self.ranking_table.item(row, column).setBackground(color)

    def update_ranking(self, snapshot):
        """Show all reservoir values while preserving each original index."""
        values = np.asarray(snapshot["x_next"], dtype=float).reshape(-1)
        if values.size != int(self.current_esn.nr):
            raise ValueError(
                f"Expected {self.current_esn.nr} reservoir values, got {values.size}"
            )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Cannot rank non-finite reservoir values")

        order = np.argsort(-values, kind="stable")
        low_indices = np.asarray(snapshot["swapped_low_indices"], dtype=int).reshape(-1)
        high_indices = np.asarray(snapshot["swapped_high_indices"], dtype=int).reshape(-1)
        swap_partners = {}
        swap_pairs = []
        for low_index, high_index in zip(low_indices, high_indices):
            low_index = int(low_index)
            high_index = int(high_index)
            swap_partners[low_index] = high_index
            swap_partners[high_index] = low_index
            swap_pairs.append(f"{high_index}<->{low_index}")
        swap_text = ", ".join(swap_pairs) if swap_pairs else "无"
        self.ranking_step_label.setText(
            f"ESN step: {int(snapshot['step_count'])} | "
            f"n={int(snapshot['extreme_swap_n'])} | 本步交换: {swap_text}"
        )
        for row, neuron_index in enumerate(order):
            value = float(values[neuron_index])
            self.ranking_table.item(row, 0).setText(str(row + 1))
            self.ranking_table.item(row, 1).setText(str(int(neuron_index)))
            self.ranking_table.item(row, 2).setText(f"{value:+.8f}")
            swap_partner = swap_partners.get(int(neuron_index))
            was_swapped = swap_partner is not None
            self.ranking_table.item(row, 3).setText(
                str(swap_partner) if was_swapped else ""
            )

            strength = min(abs(value), 1.0)
            if was_swapped:
                color = QColor(255, 247, 204)
            elif value > 0.0:
                color = QColor(255, int(250 - 55 * strength), int(250 - 55 * strength))
            elif value < 0.0:
                color = QColor(int(250 - 55 * strength), int(250 - 25 * strength), 255)
            else:
                color = QColor(255, 255, 255)
            for column in range(4):
                self.ranking_table.item(row, column).setBackground(color)

    def select_ranked_neuron(self, row, _column):
        """Use a clicked ranking row as the neuron shown in the curves."""
        item = self.ranking_table.item(row, 1)
        if item is not None and item.text().isdigit():
            self.neuron_input.setValue(int(item.text()))

    def pause_simulation(self):
        self.worker.set_paused(True)

    def start_slow_run(self):
        if not self._supported():
            return
        if not self.slow_mode:
            self.previous_speed = self.worker.get_realtime_factor()
            self.slow_mode = True
        self.worker.cancel_single_esn_steps()
        self.worker.set_realtime_factor(0.05)
        self.worker.set_paused(False)

    def step_one_esn_update(self):
        if not self._supported():
            return
        self.worker.set_paused(True)
        self.worker.request_single_esn_step()

    def clear_history(self):
        self.history.clear()
        self.current_esn.drain_debug_snapshots()
        for curve in self.curves.values():
            curve.setData([], [])
        for label in self.detail_labels.values():
            label.setText("-")
        self.ranking_step_label.setText("ESN step: -")
        self.ranking_table.clearSelection()
        for row in range(self.ranking_table.rowCount()):
            for column in range(self.ranking_table.columnCount()):
                item = self.ranking_table.item(row, column)
                item.setText("-")
                item.setBackground(QColor(255, 255, 255))

    def closeEvent(self, event):
        self.refresh_timer.stop()
        self.worker.cancel_single_esn_steps()
        if self.slow_mode:
            self.worker.set_paused(True)
            self.worker.set_realtime_factor(self.previous_speed)
        self.current_esn.set_debug_enabled(False)
        self.debug_active = False
        self.history.clear()
        if self.owner is not None:
            self.owner.inspector_window = None
            self.owner.sync_worker_controls()
        event.accept()

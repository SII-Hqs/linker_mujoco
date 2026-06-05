#!/usr/bin/env python3
"""Plot fsr_force_g, tau_contact_fsr, tau_motion_no_contact, tau_drive_with_fsr from l20a_dynamics_torque.csv."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "retargeting_data"
    / "linker_l20a_052202"
    / "l20a_dynamics_torque.csv"
)

JOINT_SUFFIXES = ["index_joint0", "index_joint1", "index_joint2", "index_joint3"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot dynamics torque CSV produced by the L20a contact pipeline.",
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(DEFAULT_CSV_PATH),
        help=f"Path to the dynamics torque CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PNG path. Defaults to <csv_stem>_plot.png next to the CSV.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive matplotlib window (only save the PNG).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"[plot_dynamics_torque] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = csv_path.parent / f"{csv_path.stem}_plot.png"

    df = pd.read_csv(csv_path)
    t = df["timestamp"] - df["timestamp"].iloc[0]  # 相对时间 (s)

    # 2 rows (fsr_force + tau_contact_fsr) + 4 rows (each joint comparison)
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(csv_path.name, fontsize=10)

    # --- fsr_force_g ---
    axes[0].plot(t, df["fsr_force_g"], color="tab:blue", linewidth=1.0)
    axes[0].set_ylabel("fsr_force_g (g)")
    axes[0].set_title("FSR Force")
    axes[0].grid(True, alpha=0.3)

    # --- tau_contact_fsr ---
    ax1 = axes[1]
    for suffix in JOINT_SUFFIXES:
        col = f"tau_contact_fsr_{suffix}"
        ax1.plot(t, df[col], linewidth=1.0, label=suffix)
    ax1.set_ylabel("Torque (N·m)")
    ax1.set_title("tau_contact_fsr")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- tau_motion_no_contact vs tau_drive_with_fsr (per joint) ---
    for i, suffix in enumerate(JOINT_SUFFIXES):
        ax = axes[2 + i]
        col_motion = f"tau_motion_no_contact_{suffix}"
        col_drive = f"tau_required_with_fsr_{suffix}"
        ax.plot(t, df[col_motion], linewidth=1.0, linestyle="--", label="motion_no_contact")
        ax.plot(t, df[col_drive], linewidth=1.0, label="drive_with_fsr")
        ax.set_ylabel("Torque (N·m)")
        ax.set_title(f"{suffix}: motion_no_contact vs drive_with_fsr")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"[plot_dynamics_torque] Saved figure to: {output_path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
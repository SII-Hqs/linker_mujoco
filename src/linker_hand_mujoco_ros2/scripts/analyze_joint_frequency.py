#!/usr/bin/env python3
"""Analyze selected MuJoCo hand-joint logs in the frequency domain."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal


TARGET_COLUMN = re.compile(
    r"^(?P<signal>tau_inverse|qpos|qvel)_"
    r"(?P<finger>index|middle|ring|little)_joint(?P<joint>[1-3])$"
)
SIGNAL_UNITS = {
    "qpos": "rad",
    "qvel": "rad/s",
    "tau_inverse": "N*m",
}
SUMMARY_FIELDS = (
    "file",
    "signal",
    "joint",
    "sample_count",
    "duration_s",
    "sample_rate_hz",
    "rms_detrended",
    "peak_frequency_hz",
    "median_frequency_hz",
    "frequency_95pct_hz",
    "total_power",
    "power_0p1_to_1_hz",
    "power_1_to_3_hz",
    "power_3_to_10_hz",
    "power_10_hz_to_nyquist",
)


def target_columns(fieldnames, signal_names=("qpos", "qvel")):
    """Return target columns in finger and joint order."""
    selected = []
    for field in fieldnames:
        match = TARGET_COLUMN.fullmatch(field)
        if match is not None and match.group("signal") in signal_names:
            selected.append(field)
    return selected


def read_log(path, signal_names=("qpos", "qvel")):
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "sim_time" not in fieldnames:
        raise ValueError(f"Missing sim_time column: {path}")
    columns = target_columns(fieldnames, signal_names)
    if not columns:
        raise ValueError(f"No target joint columns found: {path}")
    if len(rows) < 4:
        raise ValueError(f"At least four samples are required: {path}")

    try:
        times = np.asarray(
            [float(row["sim_time"]) for row in rows], dtype=float
        )
        values = np.asarray(
            [[float(row[column]) for column in columns] for row in rows],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Target columns must contain numeric values: {path}"
        ) from error

    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError(f"Log contains non-finite values: {path}")
    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        raise ValueError(f"sim_time must be strictly increasing: {path}")
    interval = float(np.median(intervals))
    tolerance = max(1e-9, interval * 1e-5)
    if not np.allclose(intervals, interval, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"sim_time is not uniformly sampled in {path}; resample it first"
        )
    return times, columns, values, 1.0 / interval


def frequency_at_fraction(frequencies, psd, fraction):
    if len(frequencies) < 2:
        return float(frequencies[0])
    bin_power = psd * float(frequencies[1] - frequencies[0])
    cumulative = np.cumsum(bin_power)
    if cumulative[-1] <= 0.0:
        return 0.0
    index = int(
        np.searchsorted(cumulative, fraction * cumulative[-1], side="left")
    )
    return float(frequencies[min(index, len(frequencies) - 1)])


def integrate_band(frequencies, psd, lower, upper):
    """Integrate PSD bins whose centers fall in [lower, upper)."""
    if len(frequencies) < 2:
        return 0.0
    mask = (frequencies >= lower) & (frequencies < upper)
    return float(np.sum(psd[mask]) * (frequencies[1] - frequencies[0]))


def analyze_trace(values, sample_rate, nperseg=256, min_peak_frequency=0.1):
    segment_length = min(int(nperseg), len(values))
    frequencies, psd = scipy_signal.welch(
        values,
        fs=sample_rate,
        window="hann",
        nperseg=segment_length,
        noverlap=segment_length // 2,
        detrend="linear",
        scaling="density",
    )
    detrended = scipy_signal.detrend(values, type="linear")
    peak_candidates = np.flatnonzero(frequencies >= min_peak_frequency)
    peak_index = (
        int(peak_candidates[np.argmax(psd[peak_candidates])])
        if len(peak_candidates)
        else int(np.argmax(psd))
    )
    nyquist = sample_rate / 2.0
    summary = {
        "rms_detrended": float(np.sqrt(np.mean(np.square(detrended)))),
        "peak_frequency_hz": float(frequencies[peak_index]),
        "median_frequency_hz": frequency_at_fraction(frequencies, psd, 0.50),
        "frequency_95pct_hz": frequency_at_fraction(frequencies, psd, 0.95),
        "total_power": integrate_band(frequencies, psd, 0.0, np.inf),
        "power_0p1_to_1_hz": integrate_band(frequencies, psd, 0.1, 1.0),
        "power_1_to_3_hz": integrate_band(frequencies, psd, 1.0, 3.0),
        "power_3_to_10_hz": integrate_band(frequencies, psd, 3.0, 10.0),
        "power_10_hz_to_nyquist": integrate_band(
            frequencies, psd, 10.0, nyquist + np.finfo(float).eps
        ),
    }
    return frequencies, psd, summary


def analyze_logs(paths, signal_names, nperseg, min_peak_frequency):
    summaries = []
    spectra = []
    expected_columns = None
    expected_frequencies = None

    for path in paths:
        times, columns, values, sample_rate = read_log(path, signal_names)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(
                f"Target columns differ from the first log: {path}"
            )

        for column_index, column in enumerate(columns):
            frequencies, psd, metrics = analyze_trace(
                values[:, column_index],
                sample_rate,
                nperseg=nperseg,
                min_peak_frequency=min_peak_frequency,
            )
            if expected_frequencies is None:
                expected_frequencies = frequencies
            elif not np.allclose(frequencies, expected_frequencies):
                raise ValueError(
                    "Logs produce different frequency bins; use a common "
                    "sampling rate "
                    "and ensure each log has at least nperseg samples"
                )

            match = TARGET_COLUMN.fullmatch(column)
            signal_name = match.group("signal")
            joint = f"{match.group('finger')}_joint{match.group('joint')}"
            summaries.append(
                {
                    "file": path.name,
                    "signal": signal_name,
                    "joint": joint,
                    "sample_count": len(times),
                    "duration_s": float(times[-1] - times[0]),
                    "sample_rate_hz": sample_rate,
                    **metrics,
                }
            )
            spectra.append(
                {
                    "file": path.name,
                    "signal": signal_name,
                    "joint": joint,
                    "frequencies": frequencies,
                    "psd": psd,
                }
            )
    return summaries, spectra


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_spectra(spectra, min_peak_frequency):
    grouped = {}
    for spectrum in spectra:
        key = (spectrum["signal"], spectrum["joint"])
        grouped.setdefault(key, []).append(spectrum)

    summary_rows = []
    spectrum_rows = []
    aggregates = {}
    for (signal_name, joint), group in grouped.items():
        frequencies = group[0]["frequencies"]
        psds = np.asarray([item["psd"] for item in group])
        median_psd = np.median(psds, axis=0)
        q25_psd = np.quantile(psds, 0.25, axis=0)
        q75_psd = np.quantile(psds, 0.75, axis=0)
        candidates = np.flatnonzero(frequencies >= min_peak_frequency)
        if not len(candidates):
            candidates = np.arange(len(frequencies))
        peak_index = int(candidates[np.argmax(median_psd[candidates])])
        file_peaks = []
        for item in group:
            index = int(candidates[np.argmax(item["psd"][candidates])])
            file_peaks.append(float(frequencies[index]))
        file_peaks = np.asarray(file_peaks)
        total_power = integrate_band(frequencies, median_psd, 0.0, np.inf)
        band_powers = {
            "power_0p1_to_1_hz": integrate_band(
                frequencies, median_psd, 0.1, 1.0
            ),
            "power_1_to_3_hz": integrate_band(
                frequencies, median_psd, 1.0, 3.0
            ),
            "power_3_to_10_hz": integrate_band(
                frequencies, median_psd, 3.0, 10.0
            ),
            "power_10_hz_to_nyquist": integrate_band(
                frequencies, median_psd, 10.0, np.inf
            ),
        }
        summary_rows.append(
            {
                "signal": signal_name,
                "joint": joint,
                "log_count": len(group),
                "aggregate_peak_frequency_hz": float(frequencies[peak_index]),
                "file_peak_median_hz": float(np.median(file_peaks)),
                "file_peak_q25_hz": float(np.quantile(file_peaks, 0.25)),
                "file_peak_q75_hz": float(np.quantile(file_peaks, 0.75)),
                "median_frequency_hz": frequency_at_fraction(
                    frequencies, median_psd, 0.50
                ),
                "frequency_95pct_hz": frequency_at_fraction(
                    frequencies, median_psd, 0.95
                ),
                "median_total_power": total_power,
                **band_powers,
                **{
                    f"{name}_percent": 100.0 * power / total_power
                    if total_power > 0.0
                    else 0.0
                    for name, power in band_powers.items()
                },
            }
        )
        for frequency, median, q25, q75 in zip(
            frequencies, median_psd, q25_psd, q75_psd
        ):
            spectrum_rows.append(
                {
                    "signal": signal_name,
                    "joint": joint,
                    "frequency_hz": float(frequency),
                    "median_psd": float(median),
                    "q25_psd": float(q25),
                    "q75_psd": float(q75),
                }
            )
        aggregates[(signal_name, joint)] = (
            frequencies,
            median_psd,
            q25_psd,
            q75_psd,
        )
    return summary_rows, spectrum_rows, aggregates


def plot_aggregates(
    output_dir,
    aggregates,
    signal_names,
    max_frequency,
    min_peak_frequency,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fingers = ("index", "middle", "ring", "little")
    for signal_name in signal_names:
        fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharex=True)
        lower_values = []
        upper_values = []
        for finger_index, finger in enumerate(fingers):
            for joint_number in range(1, 4):
                axis = axes[finger_index, joint_number - 1]
                joint = f"{finger}_joint{joint_number}"
                frequencies, median, q25, q75 = aggregates[
                    (signal_name, joint)
                ]
                mask = frequencies <= max_frequency
                positive_lower = q25[mask][q25[mask] > 0.0]
                positive_upper = q75[mask][q75[mask] > 0.0]
                lower_values.extend(positive_lower.tolist())
                upper_values.extend(positive_upper.tolist())
                peak_candidates = np.flatnonzero(
                    frequencies >= min_peak_frequency
                )
                if not len(peak_candidates):
                    peak_candidates = np.arange(len(frequencies))
                peak_index = int(
                    peak_candidates[np.argmax(median[peak_candidates])]
                )
                axis.fill_between(
                    frequencies[mask], q25[mask], q75[mask], alpha=0.25
                )
                axis.plot(frequencies[mask], median[mask], linewidth=1.2)
                axis.plot(
                    frequencies[peak_index],
                    median[peak_index],
                    marker="o",
                    markersize=3.5,
                )
                axis.set_title(
                    f"{joint.replace('_', ' ')} | peak "
                    f"{frequencies[peak_index]:.2f} Hz",
                    fontsize=10,
                )
                axis.grid(True, which="both", alpha=0.25, linewidth=0.5)
                axis.set_yscale("log")
                if joint_number == 1:
                    unit = SIGNAL_UNITS[signal_name]
                    axis.set_ylabel(f"PSD [({unit})^2/Hz]")
                if finger_index == len(fingers) - 1:
                    axis.set_xlabel("Frequency [Hz]")

        if lower_values and upper_values:
            lower = max(
                float(np.quantile(lower_values, 0.01)) * 0.5, 1e-18
            )
            upper = max(float(np.max(upper_values)) * 2.0, lower * 10.0)
            for axis in axes.flat:
                axis.set_ylim(lower, upper)
                axis.set_xlim(0.0, max_frequency)
        fig.suptitle(
            f"{signal_name} Welch PSD: median and interquartile range "
            "across logs"
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        fig.savefig(output_dir / f"{signal_name}_welch_psd.png", dpi=200)
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="CSV file or directory of CSV logs"
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--signal",
        choices=("both", "qpos", "qvel", "tau_inverse"),
        default="both",
    )
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--min-peak-frequency", type=float, default=0.1)
    parser.add_argument("--max-frequency", type=float, default=15.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.nperseg < 4:
        raise ValueError("nperseg must be at least 4")
    if args.min_peak_frequency < 0.0:
        raise ValueError("min-peak-frequency cannot be negative")
    if args.max_frequency <= 0.0:
        raise ValueError("max-frequency must be positive")

    paths = (
        [args.input]
        if args.input.is_file()
        else sorted(args.input.glob("*.csv"))
    )
    if not paths:
        raise FileNotFoundError(f"No CSV files found: {args.input}")
    output_dir = args.output or (
        args.input.parent / f"{args.input.stem}_frequency_analysis"
        if args.input.is_file()
        else args.input / "frequency_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_names = (
        ("qpos", "qvel") if args.signal == "both" else (args.signal,)
    )

    summaries, spectra = analyze_logs(
        paths,
        signal_names,
        args.nperseg,
        args.min_peak_frequency,
    )
    aggregate_summary, aggregate_rows, aggregates = aggregate_spectra(
        spectra, args.min_peak_frequency
    )
    write_csv(output_dir / "per_log_summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(
        output_dir / "aggregate_summary.csv",
        tuple(aggregate_summary[0]),
        aggregate_summary,
    )
    write_csv(
        output_dir / "aggregate_spectra.csv",
        tuple(aggregate_rows[0]),
        aggregate_rows,
    )
    plot_aggregates(
        output_dir,
        aggregates,
        signal_names,
        args.max_frequency,
        args.min_peak_frequency,
    )

    print(
        f"Analyzed {len(paths)} logs and "
        f"{len(aggregates)} signal/joint pairs"
    )
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

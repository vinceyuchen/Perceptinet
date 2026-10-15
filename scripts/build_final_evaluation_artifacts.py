from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BG = (18, 18, 20)
SURFACE = (27, 27, 30)
SURFACE_2 = (36, 36, 40)
GRID = (58, 58, 64)
TEXT = (238, 238, 242)
MUTED = (165, 165, 173)
BLUE = (164, 145, 112)
GREEN = (132, 166, 126)
AMBER = (82, 166, 210)
RED = (92, 92, 216)


def number(value) -> Optional[float]:
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def mean(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def stdev(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None

    return statistics.stdev(values)


def load_latest():
    pointer_path = (
        REPO_ROOT
        / "results"
        / "validation"
        / "latest_suite.json"
    )

    if not pointer_path.exists():
        raise FileNotFoundError(
            "results/validation/latest_suite.json not found. "
            "Run final validation first."
        )

    pointer = json.loads(
        pointer_path.read_text(encoding="utf-8")
    )

    summary_path = Path(pointer["summary"])
    csv_path = Path(pointer["results_csv"])

    if not summary_path.is_absolute():
        summary_path = REPO_ROOT / summary_path

    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    suite_root = Path(pointer["suite_root"])

    if not suite_root.is_absolute():
        suite_root = REPO_ROOT / suite_root

    return pointer, summary, rows, suite_root


def put_text(
    canvas: np.ndarray,
    text: str,
    origin,
    *,
    scale: float = 0.7,
    color=TEXT,
    thickness: int = 1,
):
    cv2.putText(
        canvas,
        str(text),
        tuple(map(int, origin)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def rect(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    fill=SURFACE,
    border=GRID,
):
    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        fill,
        -1,
    )
    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        border,
        1,
        cv2.LINE_AA,
    )


def fmt_mean_sd(values: List[float], unit: str = "") -> str:
    if not values:
        return "N/A"

    m = mean(values)
    s = stdev(values)

    return f"{m:.3f} ± {s:.3f}{unit}"


def collect(rows: List[Dict[str, str]], key: str) -> List[float]:
    values = []

    for row in rows:
        value = number(row.get(key))

        if value is not None:
            values.append(value)

    return values


def draw_metric_card(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    value: str,
    subtitle: str,
    accent,
):
    rect(
        canvas,
        x,
        y,
        x + w,
        y + h,
        fill=SURFACE,
    )

    cv2.rectangle(
        canvas,
        (x, y),
        (x + 5, y + h),
        accent,
        -1,
    )

    put_text(
        canvas,
        title.upper(),
        (x + 24, y + 32),
        scale=0.48,
        color=MUTED,
        thickness=1,
    )

    put_text(
        canvas,
        value,
        (x + 24, y + 78),
        scale=0.92,
        color=TEXT,
        thickness=2,
    )

    put_text(
        canvas,
        subtitle,
        (x + 24, y + 108),
        scale=0.40,
        color=MUTED,
        thickness=1,
    )


def render_overview(
    summary: Dict,
    rows: List[Dict[str, str]],
    output_path: Path,
):
    canvas = np.full(
        (900, 1600, 3),
        BG,
        dtype=np.uint8,
    )

    put_text(
        canvas,
        "PERCEPTINET FINAL EVALUATION",
        (54, 66),
        scale=1.05,
        color=TEXT,
        thickness=2,
    )

    put_text(
        canvas,
        "Occluded-pedestrian cooperative perception · paired baseline vs cooperative trials",
        (56, 100),
        scale=0.46,
        color=MUTED,
    )

    passed = int(summary.get("passed_pairs", 0))
    total = int(summary.get("pair_count", 0))

    put_text(
        canvas,
        f"{passed}/{total} PAIRED TRIALS PASSED",
        (1230, 67),
        scale=0.55,
        color=GREEN if passed == total and total else AMBER,
        thickness=2,
    )

    metrics = [
        (
            "RSU detection lead",
            fmt_mean_sd(
                collect(
                    rows,
                    "cooperative_rsu_detection_lead_s",
                ),
                " s",
            ),
            "Earlier stable detection than Onboard",
            BLUE,
        ),
        (
            "V2I latency",
            fmt_mean_sd(
                collect(rows, "v2i_mean_ms"),
                " ms",
            ),
            "Mean per cooperative trial",
            BLUE,
        ),
        (
            "End-to-end latency",
            fmt_mean_sd(
                collect(rows, "end_to_end_mean_ms"),
                " ms",
            ),
            "Detection-to-control pipeline",
            BLUE,
        ),
        (
            "Clearance gain",
            fmt_mean_sd(
                collect(rows, "clearance_gain_m"),
                " m",
            ),
            "Cooperative final minus baseline minimum",
            GREEN,
        ),
        (
            "Travel distance saved",
            fmt_mean_sd(
                collect(rows, "travel_distance_saved_m"),
                " m",
            ),
            "Earlier stop under cooperative perception",
            GREEN,
        ),
        (
            "Brake trigger TTC",
            fmt_mean_sd(
                collect(rows, "brake_trigger_ttc_s"),
                " s",
            ),
            "Cooperative braking trigger",
            AMBER,
        ),
    ]

    card_w = 470
    card_h = 130
    xs = [54, 565, 1076]
    ys = [145, 305]

    idx = 0

    for y in ys:
        for x in xs:
            title, value, subtitle, accent = metrics[idx]
            draw_metric_card(
                canvas,
                x,
                y,
                card_w,
                card_h,
                title,
                value,
                subtitle,
                accent,
            )
            idx += 1

    # Occlusion detection comparison.
    panel_x1 = 54
    panel_y1 = 480
    panel_x2 = 765
    panel_y2 = 820

    rect(
        canvas,
        panel_x1,
        panel_y1,
        panel_x2,
        panel_y2,
    )

    put_text(
        canvas,
        "DETECTION WHILE OCCLUDED",
        (78, 520),
        scale=0.62,
        color=TEXT,
        thickness=2,
    )

    onboard = (
        number(
            summary.get(
                "mean_onboard_occluded_detection_rate"
            )
        )
        or 0.0
    )

    rsu = (
        number(
            summary.get(
                "mean_rsu_occluded_detection_rate"
            )
        )
        or 0.0
    )

    bar_left = 100
    bar_right = 720
    bar_width = bar_right - bar_left

    for label, rate, y, accent in [
        ("Onboard Camera", onboard, 610, MUTED),
        ("RSU Camera", rsu, 715, BLUE),
    ]:
        put_text(
            canvas,
            label,
            (100, y - 25),
            scale=0.48,
            color=MUTED,
        )

        cv2.rectangle(
            canvas,
            (bar_left, y),
            (bar_right, y + 28),
            SURFACE_2,
            -1,
        )

        end_x = bar_left + int(
            max(0.0, min(1.0, rate)) * bar_width
        )

        cv2.rectangle(
            canvas,
            (bar_left, y),
            (end_x, y + 28),
            accent,
            -1,
        )

        put_text(
            canvas,
            f"{rate * 100.0:.1f}%",
            (bar_right - 82, y + 22),
            scale=0.48,
            color=TEXT,
            thickness=2,
        )

    # Evidence chain panel.
    rect(
        canvas,
        800,
        480,
        1546,
        820,
    )

    put_text(
        canvas,
        "SYSTEM-LEVEL EVIDENCE",
        (824, 520),
        scale=0.62,
        color=TEXT,
        thickness=2,
    )

    evidence = [
        (
            "1",
            "RSU sees the occluded pedestrian earlier",
            BLUE,
        ),
        (
            "2",
            "Remote detections cross the V2I channel",
            BLUE,
        ),
        (
            "3",
            "Fusion and Risk trigger earlier response",
            AMBER,
        ),
        (
            "4",
            "Vehicle stops sooner with more clearance",
            GREEN,
        ),
    ]

    y = 575

    for index, line, accent in evidence:
        cv2.circle(
            canvas,
            (850, y - 5),
            18,
            accent,
            -1,
            cv2.LINE_AA,
        )

        put_text(
            canvas,
            index,
            (844, y + 2),
            scale=0.42,
            color=BG,
            thickness=2,
        )

        put_text(
            canvas,
            line,
            (885, y + 3),
            scale=0.48,
            color=TEXT,
            thickness=1,
        )

        y += 62

    put_text(
        canvas,
        "Prototype simulation results only · not production automotive safety certification",
        (54, 866),
        scale=0.40,
        color=MUTED,
    )

    cv2.imwrite(
        str(output_path),
        canvas,
    )


def render_trials(
    rows: List[Dict[str, str]],
    output_path: Path,
):
    canvas = np.full(
        (900, 1600, 3),
        BG,
        dtype=np.uint8,
    )

    put_text(
        canvas,
        "PER-TRIAL VALIDATION CONSISTENCY",
        (54, 66),
        scale=1.0,
        color=TEXT,
        thickness=2,
    )

    put_text(
        canvas,
        "Positive values indicate cooperative-perception advantage",
        (56, 100),
        scale=0.46,
        color=MUTED,
    )

    panels = [
        (
            "Clearance gain",
            "clearance_gain_m",
            "m",
            GREEN,
        ),
        (
            "Travel distance saved",
            "travel_distance_saved_m",
            "m",
            GREEN,
        ),
        (
            "RSU detection lead",
            "cooperative_rsu_detection_lead_s",
            "s",
            BLUE,
        ),
        (
            "V2I mean latency",
            "v2i_mean_ms",
            "ms",
            BLUE,
        ),
        (
            "End-to-end mean latency",
            "end_to_end_mean_ms",
            "ms",
            AMBER,
        ),
        (
            "Brake trigger TTC",
            "brake_trigger_ttc_s",
            "s",
            AMBER,
        ),
    ]

    positions = [
        (54, 145),
        (565, 145),
        (1076, 145),
        (54, 490),
        (565, 490),
        (1076, 490),
    ]

    for (
        title,
        key,
        unit,
        accent,
    ), (x, y) in zip(panels, positions):
        w = 470
        h = 285

        rect(
            canvas,
            x,
            y,
            x + w,
            y + h,
        )

        put_text(
            canvas,
            title.upper(),
            (x + 22, y + 36),
            scale=0.55,
            color=TEXT,
            thickness=2,
        )

        values = collect(rows, key)

        if not values:
            put_text(
                canvas,
                "N/A",
                (x + 22, y + 100),
                scale=0.8,
                color=MUTED,
            )
            continue

        max_value = max(values)

        if max_value <= 0.0:
            max_value = 1.0

        chart_left = x + 42
        chart_right = x + w - 24
        chart_top = y + 75
        chart_bottom = y + h - 48

        cv2.line(
            canvas,
            (chart_left, chart_bottom),
            (chart_right, chart_bottom),
            GRID,
            1,
            cv2.LINE_AA,
        )

        count = len(values)
        slot = (
            (chart_right - chart_left)
            / max(1, count)
        )

        for index, value in enumerate(values):
            bar_x1 = int(
                chart_left
                + index * slot
                + slot * 0.22
            )
            bar_x2 = int(
                chart_left
                + (index + 1) * slot
                - slot * 0.22
            )

            usable_height = chart_bottom - chart_top
            height = int(
                (value / max_value)
                * usable_height
                * 0.86
            )

            bar_y1 = chart_bottom - height

            cv2.rectangle(
                canvas,
                (bar_x1, bar_y1),
                (bar_x2, chart_bottom),
                accent,
                -1,
            )

            put_text(
                canvas,
                f"{value:.2f}",
                (bar_x1 - 2, bar_y1 - 8),
                scale=0.36,
                color=TEXT,
                thickness=1,
            )

            put_text(
                canvas,
                f"T{index + 1}",
                (bar_x1 + 2, chart_bottom + 24),
                scale=0.38,
                color=MUTED,
            )

        put_text(
            canvas,
            f"Mean ± SD: {fmt_mean_sd(values, ' ' + unit)}",
            (x + 22, y + h - 14),
            scale=0.40,
            color=MUTED,
        )

    cv2.imwrite(
        str(output_path),
        canvas,
    )


def write_markdown(
    summary: Dict,
    rows: List[Dict[str, str]],
    output_path: Path,
):
    rsu_lead = collect(
        rows,
        "cooperative_rsu_detection_lead_s",
    )
    onboard = collect(
        rows,
        "cooperative_onboard_occluded_detection_rate",
    )
    rsu = collect(
        rows,
        "cooperative_rsu_occluded_detection_rate",
    )
    v2i = collect(rows, "v2i_mean_ms")
    e2e = collect(rows, "end_to_end_mean_ms")
    clearance = collect(rows, "clearance_gain_m")
    travel = collect(
        rows,
        "travel_distance_saved_m",
    )

    lines = [
        "# Perceptinet Final Evaluation Results",
        "",
        "## Validation design",
        "",
        (
            "Paired CARLA trials compare a baseline Onboard-only control "
            "against the cooperative Onboard + RSU + V2I + Fusion pipeline."
        ),
        "",
        "## Aggregate results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        (
            f"| Paired trials passed | "
            f"{summary.get('passed_pairs', 0)}/"
            f"{summary.get('pair_count', 0)} |"
        ),
        (
            "| RSU stable-detection lead | "
            f"{fmt_mean_sd(rsu_lead, ' s')} |"
        ),
        (
            "| Onboard detection rate while occluded | "
            f"{fmt_mean_sd(onboard)} |"
        ),
        (
            "| RSU detection rate while occluded | "
            f"{fmt_mean_sd(rsu)} |"
        ),
        (
            "| Mean V2I latency | "
            f"{fmt_mean_sd(v2i, ' ms')} |"
        ),
        (
            "| Mean end-to-end latency | "
            f"{fmt_mean_sd(e2e, ' ms')} |"
        ),
        (
            "| Cooperative clearance gain | "
            f"{fmt_mean_sd(clearance, ' m')} |"
        ),
        (
            "| Travel distance saved | "
            f"{fmt_mean_sd(travel, ' m')} |"
        ),
        "",
        "## Report-ready interpretation",
        "",
        (
            "Across the paired validation trials, roadside cooperative "
            "perception detected the occluded pedestrian substantially more "
            "reliably than the Onboard Camera during the occlusion window and "
            "provided earlier stable detection. The additional information "
            "was delivered through the simulated V2I channel and fused into "
            "the Ego Vehicle world model. This earlier awareness translated "
            "into earlier braking, reduced travel before stopping, and greater "
            "final clearance than the baseline Onboard-only condition."
        ),
        "",
        (
            "These results demonstrate the intended prototype benefit in the "
            "CARLA occluded-pedestrian scenario. They should not be presented "
            "as production automotive safety certification."
        ),
    ]

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    pointer, summary, rows, suite_root = load_latest()

    if not rows:
        raise RuntimeError(
            "No paired validation rows were found."
        )

    artifact_dir = (
        suite_root / "evaluation_artifacts"
    )
    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overview_path = (
        artifact_dir
        / "final_evaluation_overview.png"
    )
    trials_path = (
        artifact_dir
        / "final_evaluation_trials.png"
    )
    markdown_path = (
        artifact_dir
        / "final_evaluation_report.md"
    )

    render_overview(
        summary,
        rows,
        overview_path,
    )
    render_trials(
        rows,
        trials_path,
    )
    write_markdown(
        summary,
        rows,
        markdown_path,
    )

    print(
        "=== Perceptinet Final Evaluation Artifacts ==="
    )
    print("Suite:", pointer.get("suite_id"))
    print(
        "Pairs passed:",
        f"{summary.get('passed_pairs')}/"
        f"{summary.get('pair_count')}",
    )
    print(
        "Overview:",
        overview_path,
    )
    print(
        "Per-trial chart:",
        trials_path,
    )
    print(
        "Report-ready markdown:",
        markdown_path,
    )


if __name__ == "__main__":
    main()

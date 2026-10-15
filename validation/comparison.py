from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _nested(
    data: Dict[str, Any],
    *keys: str,
) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def _latency_stat(
    run: Dict[str, Any],
    metric: str,
    stat: str,
) -> Optional[float]:
    return _number(
        _nested(
            run,
            "pipeline_metrics",
            "latency",
            metric,
            stat,
        )
    )


def compare_pair(
    baseline: Dict[str, Any],
    cooperative: Dict[str, Any],
    *,
    pair_id: str,
) -> Dict[str, Any]:
    baseline_min = _number(
        _nested(
            baseline,
            "safety",
            "minimum_clearance_m",
        )
    )
    cooperative_final = _number(
        _nested(
            cooperative,
            "safety",
            "final_clearance_m",
        )
    )

    clearance_gain = None

    if (
        baseline_min is not None
        and cooperative_final is not None
    ):
        clearance_gain = cooperative_final - baseline_min

    baseline_travel = _number(
        _nested(
            baseline,
            "safety",
            "ego_travelled_m",
        )
    )
    cooperative_travel = _number(
        _nested(
            cooperative,
            "safety",
            "ego_travelled_m",
        )
    )

    travel_saved = None

    if (
        baseline_travel is not None
        and cooperative_travel is not None
    ):
        travel_saved = baseline_travel - cooperative_travel

    rsu_lead = _number(
        _nested(
            cooperative,
            "pipeline_metrics",
            "detection",
            "rsu_detection_lead_s",
        )
    )

    onboard_occ = _number(
        _nested(
            cooperative,
            "detection",
            "onboard_occluded_detection_rate",
        )
    )
    rsu_occ = _number(
        _nested(
            cooperative,
            "detection",
            "rsu_occluded_detection_rate",
        )
    )

    cooperative_safe_stop = bool(
        _nested(
            cooperative,
            "safety",
            "cooperative_safe_stop",
        )
    )

    baseline_failure = bool(
        _nested(
            baseline,
            "safety",
            "baseline_failure_demonstrated",
        )
    )

    perception_advantage = (
        onboard_occ is not None
        and rsu_occ is not None
        and rsu_occ > onboard_occ
    )

    early_detection_advantage = (
        rsu_lead is not None
        and rsu_lead > 0.0
    )

    clearance_advantage = (
        clearance_gain is not None
        and clearance_gain > 0.0
    )

    travel_advantage = (
        travel_saved is not None
        and travel_saved > 0.0
    )

    return {
        "pair_id": pair_id,
        "baseline_run_id": baseline.get("run_id"),
        "cooperative_run_id": cooperative.get("run_id"),
        "baseline_outcome": baseline.get("outcome"),
        "cooperative_outcome": cooperative.get("outcome"),
        "baseline_minimum_clearance_m": baseline_min,
        "cooperative_final_clearance_m": cooperative_final,
        "clearance_gain_m": clearance_gain,
        "baseline_travelled_m": baseline_travel,
        "cooperative_travelled_m": cooperative_travel,
        "travel_distance_saved_m": travel_saved,
        "cooperative_rsu_detection_lead_s": rsu_lead,
        "cooperative_onboard_occluded_detection_rate": onboard_occ,
        "cooperative_rsu_occluded_detection_rate": rsu_occ,
        "v2i_mean_ms": _latency_stat(
            cooperative,
            "v2i_latency_ms",
            "mean",
        ),
        "v2i_p95_ms": _latency_stat(
            cooperative,
            "v2i_latency_ms",
            "p95",
        ),
        "end_to_end_mean_ms": _latency_stat(
            cooperative,
            "end_to_end_latency_ms",
            "mean",
        ),
        "end_to_end_p95_ms": _latency_stat(
            cooperative,
            "end_to_end_latency_ms",
            "p95",
        ),
        "brake_trigger_distance_m": _number(
            _nested(
                cooperative,
                "safety",
                "brake_trigger_distance_m",
            )
        ),
        "brake_trigger_ttc_s": _number(
            _nested(
                cooperative,
                "safety",
                "brake_trigger_ttc_s",
            )
        ),
        "actual_braking_distance_m": _number(
            _nested(
                cooperative,
                "safety",
                "actual_braking_distance_m",
            )
        ),
        "cooperative_safe_stop": cooperative_safe_stop,
        "baseline_failure_demonstrated": baseline_failure,
        "perception_advantage_demonstrated": perception_advantage,
        "early_detection_advantage_demonstrated":
            early_detection_advantage,
        "clearance_advantage_demonstrated":
            clearance_advantage,
        "travel_advantage_demonstrated":
            travel_advantage,
        "pair_pass": (
            cooperative_safe_stop
            and perception_advantage
            and early_detection_advantage
            and clearance_advantage
            and travel_advantage
        ),
    }


def _mean(
    rows: Iterable[Dict[str, Any]],
    key: str,
) -> Optional[float]:
    values = [
        value
        for row in rows
        if (value := _number(row.get(key))) is not None
    ]

    if not values:
        return None

    return float(statistics.fmean(values))


def build_suite_summary(
    pair_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    passed = sum(
        1 for row in pair_rows if row.get("pair_pass")
    )

    total = len(pair_rows)

    return {
        "schema_version": "final-validation-suite-v1",
        "pair_count": total,
        "passed_pairs": passed,
        "pass_rate": (
            None
            if total == 0
            else float(passed) / float(total)
        ),
        "all_pairs_passed": (
            total > 0 and passed == total
        ),
        "mean_clearance_gain_m": _mean(
            pair_rows,
            "clearance_gain_m",
        ),
        "mean_travel_distance_saved_m": _mean(
            pair_rows,
            "travel_distance_saved_m",
        ),
        "mean_rsu_detection_lead_s": _mean(
            pair_rows,
            "cooperative_rsu_detection_lead_s",
        ),
        "mean_onboard_occluded_detection_rate": _mean(
            pair_rows,
            "cooperative_onboard_occluded_detection_rate",
        ),
        "mean_rsu_occluded_detection_rate": _mean(
            pair_rows,
            "cooperative_rsu_occluded_detection_rate",
        ),
        "mean_v2i_latency_ms": _mean(
            pair_rows,
            "v2i_mean_ms",
        ),
        "mean_end_to_end_latency_ms": _mean(
            pair_rows,
            "end_to_end_mean_ms",
        ),
        "pairs": pair_rows,
    }


def load_json(path: Any) -> Dict[str, Any]:
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def write_pair_csv(
    rows: List[Dict[str, Any]],
    path: Any,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path

    fieldnames = list(rows[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def write_suite_json(
    summary: Dict[str, Any],
    path: Any,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path


def write_report(
    summary: Dict[str, Any],
    path: Any,
) -> Path:
    def fmt(value: Any, unit: str = "") -> str:
        number = _number(value)

        if number is None:
            return "N/A"

        return f"{number:.3f}{unit}"

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "PERCEPTINET FINAL SYSTEM VALIDATION",
        "=" * 36,
        "",
        f"Paired trials: {summary.get('pair_count', 0)}",
        f"Passed pairs: {summary.get('passed_pairs', 0)}",
        f"All pairs passed: {summary.get('all_pairs_passed')}",
        "",
        "Aggregate evidence",
        "------------------",
        (
            "Mean RSU detection lead: "
            + fmt(
                summary.get("mean_rsu_detection_lead_s"),
                " s",
            )
        ),
        (
            "Mean onboard detection rate while occluded: "
            + fmt(
                summary.get(
                    "mean_onboard_occluded_detection_rate"
                )
            )
        ),
        (
            "Mean RSU detection rate while occluded: "
            + fmt(
                summary.get(
                    "mean_rsu_occluded_detection_rate"
                )
            )
        ),
        (
            "Mean V2I latency: "
            + fmt(
                summary.get("mean_v2i_latency_ms"),
                " ms",
            )
        ),
        (
            "Mean end-to-end latency: "
            + fmt(
                summary.get("mean_end_to_end_latency_ms"),
                " ms",
            )
        ),
        (
            "Mean cooperative clearance gain: "
            + fmt(
                summary.get("mean_clearance_gain_m"),
                " m",
            )
        ),
        (
            "Mean travel distance saved: "
            + fmt(
                summary.get("mean_travel_distance_saved_m"),
                " m",
            )
        ),
        "",
        "Interpretation",
        "--------------",
        (
            "A paired trial passes when the cooperative run achieves a "
            "safe stop, RSU perception outperforms onboard perception during "
            "the scenario-defined occlusion window, RSU stable detection is "
            "earlier than onboard stable detection, cooperative final "
            "clearance is greater than baseline minimum clearance, and the "
            "cooperative run travels less distance before stopping. The "
            "baseline is allowed to achieve a late local stop; it does not "
            "need to be forced into an artificial collision or near-miss."
        ),
        "",
        (
            "All thresholds are prototype/demo evaluation parameters and "
            "must not be presented as production automotive safety limits."
        ),
    ]

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return output_path

"""Generate a truthful comparison report from local CARLA cycle results.

Run:
  py -3.12 scripts/generate_teacher_demo_report.py
Outputs (where possible):
  results/teacher_demo/report/Perceptinet_Comparative_Data_Report.pdf
  results/teacher_demo/report/Perceptinet_Comparative_Data_Report.csv
  results/teacher_demo/report/Perceptinet_Comparative_Data_Report.json

Only completed scene_result.json records are included; no measurements
are invented when files or optional values are absent.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "clear": "Scenario 1 | Clear / Normal V2I / Moving pedestrian",
    "rain_v2i": "Scenario 2 | Heavy rain / Impaired V2I / Moving pedestrian",
}
FIELDS = (
    "run", "scenario", "end_reason", "safe_stop_verified",
    "first_rsu_detection_s", "first_fused_detection_s",
    "first_brake_s", "crossing_started_sim_s", "brake_before_crossing_s",
    "pedestrian_motion_verified", "pedestrian_walked_m",
    "minimum_center_distance_m", "kinematic_contact_observed",
    "physical_collision_observed", "collision_sensor_available",
    "v2i_total_sent", "v2i_total_dropped", "v2i_outage_dropped",
    "v2i_random_dropped", "v2i_expired", "v2i_delivered",
    "v2i_mean_delivered_latency_ms",
)


def _finite(value):
    try:
        result = float(value)
        if result != result or result in (float("inf"), -float("inf")):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _round(value, places=3):
    n = _finite(value)
    return None if n is None else round(n, places)


def _get_latency(metrics):
    obj = (metrics.get("latency") or {}).get("v2i_latency_ms") or {}
    return _round(obj.get("mean"))


def read_runs(root: Path):
    rows = []
    for p in sorted(root.glob("cycle_*/scenario_result.json")):
        try:
            result = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"SKIP malformed result {p}: {exc}")
            continue
        if result.get("completed") is not True:
            continue
        key = result.get("scenario")
        if key not in SCENARIOS:
            continue
        metrics_path = p.with_name("metrics_summary.json")
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}
        v2i = result.get("v2i") or {}
        start = _finite(result.get("first_rsu_detection_s"))
        fused = _finite(result.get("first_fused_detection_s"))
        brake = _finite(result.get("first_brake_s"))
        crossing = _finite(result.get("crossing_started_sim_s"))
        row = {
            "run": p.parent.name,
            "scenario": key,
            "end_reason": result.get("reason"),
            "safe_stop_verified": result.get("safe_stop_verified"),
            "first_rsu_detection_s": _round(start),
            "first_fused_detection_s": _round(fused),
            "first_brake_s": _round(brake),
            "crossing_started_sim_s": _round(crossing),
            "brake_before_crossing_s": (
                _round(crossing - brake)
                if crossing is not None and brake is not None else None
            ),
            "pedestrian_motion_verified": result.get("pedestrian_motion_verified"),
            "pedestrian_walked_m": _round(result.get("pedestrian_walked_m")),
            "minimum_center_distance_m": _round(
                result.get("minimum_center_distance_m")
            ),
            "kinematic_contact_observed": result.get(
                "kinematic_contact_observed"
            ),
            "physical_collision_observed": result.get(
                "physical_collision_observed"
            ),
            "collision_sensor_available": result.get(
                "collision_sensor_available"
            ),
            "v2i_total_sent": v2i.get("total_sent"),
            "v2i_total_dropped": v2i.get("total_dropped"),
            "v2i_outage_dropped": v2i.get("outage_dropped"),
            "v2i_random_dropped": v2i.get("random_dropped"),
            "v2i_expired": v2i.get("expired"),
            "v2i_delivered": v2i.get("delivered"),
            "v2i_mean_delivered_latency_ms": _get_latency(metrics),
        }
        # Preserve values for each run, even when optional fields are absent.
        rows.append(row)
    return rows


def summarize(rows):
    results = {}
    for key in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == key]
        verified = sum(r["safe_stop_verified"] is True for r in subset)
        observed_contacts = sum(
            r["kinematic_contact_observed"] is True for r in subset
        )
        moved = sum(
            r["pedestrian_motion_verified"] is True for r in subset
        )
        distances = [
            _finite(r["minimum_center_distance_m"]) for r in subset
        ]
        distances = [x for x in distances if x is not None]
        reports = sum(r["safe_stop_verified"] is not None for r in subset)
        results[key] = {
            "cycles": len(subset),
            "verified_safe_stops": verified,
            "verified_safe_stop_rate": (
                round(100 * verified / reports, 1) if reports else None
            ),
            "evaluated_safe_stop_records": reports,
            "walker_motion_verified": moved,
            "kinematic_contact_cycles": observed_contacts,
            "minimum_center_distance_m": (
                round(min(distances), 3) if distances else None
            ),
            "mean_minimum_center_distance_m": (
                round(statistics.mean(distances), 3) if distances else None
            ),
        }
    return results


def make_pdf(output: Path, results, rows, created):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak,
        )
    except ImportError:
        print("PDF not generated: install with py -3.12 -m pip install reportlab")
        return False

    pdf = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=38, rightMargin=38,
        topMargin=40, bottomMargin=38,
        title="Perceptinet Comparative Data Report - FIT3163",
    )
    style = getSampleStyleSheet()
    style.add(ParagraphStyle(
        name="SmallCell", parent=style["Normal"], fontSize=8.2, leading=11,
    ))
    style.add(ParagraphStyle(
        name="SmallNote", parent=style["Normal"], fontSize=8, leading=12,
        textColor=colors.HexColor("#48556A"),
    ))
    style.add(ParagraphStyle(
        name="ReportTitle", parent=style["Title"], fontSize=18,
        leading=23, alignment=TA_CENTER,
    ))
    cell = lambda x: Paragraph(str(x), style["SmallCell"])
    absent = lambda x: "Pending" if x is None else str(x)
    story = [
        Paragraph("PERCEPTINET", style["ReportTitle"]),
        Paragraph("Comparative Experiment Data Report | FIT3163",
                  style["Heading2"]),
        Paragraph("Generated: " + created, style["SmallNote"]),
        Spacer(1, 14),
        Paragraph("Experimental design", style["Heading2"]),
        Paragraph(
            "Scenario 1: clear weather, normal simulated V2I and "
            "a moving CARLA pedestrian. Scenario 2: heavy rain, "
            "injected V2I delay/loss and the same crossing parameters.",
            style["Normal"],
        ),
        Spacer(1, 10),
        Paragraph("Observed results (never fabricated)", style["Heading2"]),
    ]
    grid = [[cell(v) for v in (
        "Scenario", "Runs", "Safe stops", "Walk verified",
        "Contact envelopes", "Min. centre distance")]]
    for k in SCENARIOS:
        t = results[k]
        grid.append([cell(x) for x in (
            k, t["cycles"],
            f'{t["verified_safe_stops"]}/{t["evaluated_safe_stop_records"]}',
            t["walker_motion_verified"],
            t["kinematic_contact_cycles"],
            absent(t["minimum_center_distance_m"]),
        )])
    table = Table(grid, colWidths=[86, 40, 70, 72, 91, 99],
                  hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#152B45")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D3DAE2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F4F7FB")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(table)
    story.extend([
        Spacer(1, 13),
        Paragraph(
            "The measured minimum is the Ego-to-pedestrian centre-point "
            "distance; it is not bumper-to-person clearance. "
            "A kinematic footprint overlap is NOT a physics-simulated crash. "
            "Collision sensor events (if any) are separately recorded.",
            style["SmallNote"],
        ),
        Spacer(1, 12),
        Paragraph("Configured experimental parameters", style["Heading2"]),
    ])
    params = [
        ("Weather", "Clear", "Heavy rain (90 precipitation)"),
        ("V2I delay", "25 +/- 5 ms, metadata", "1200 +/- 200 ms, queued"),
        ("Random packet loss", "0%", "30%, seeded"),
        ("Injected blackout", "None", "3.0 s after first RSU detection"),
        ("Pedestrian motion", "Shared CARLA WalkerControl", "Shared CARLA WalkerControl"),
        ("Ego motion", "Kinematic, physics disabled", "Kinematic, physics disabled"),
    ]
    pt = Table([[cell(x) for x in row] for row in
                [("Parameter", "Scenario 1", "Scenario 2")] + params],
               colWidths=[125, 165, 168], hAlign="LEFT")
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EDF4")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D3DAE2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(pt)
    if not rows:
        story.extend([
            Spacer(1, 14),
            Paragraph("PENDING LIVE VALIDATION", style["Heading2"]),
            Paragraph(
                "No completed scenario_result.json records were available. "
                "Run both CARLA scenarios before drawing conclusions.",
                style["Normal"],
            )
        ])
    else:
        story.extend([PageBreak(), Paragraph("Per-cycle evidence", style["Heading2"])])
        for r in rows:
            story.append(Paragraph(
                f"{r['run']} / {r['scenario']}: {absent(r['end_reason'])}",
                style["Heading3"],
            ))
            summary = [
                ("Safe stop verified", absent(r["safe_stop_verified"])),
                ("Pedestrian moved (m)", absent(r["pedestrian_walked_m"])),
                ("Minimum centre distance (m)", absent(r["minimum_center_distance_m"])),
                ("Brake trigger (sim s)", absent(r["first_brake_s"])),
                ("Crossing start (sim s)", absent(r["crossing_started_sim_s"])),
                ("Footprint contact", absent(r["kinematic_contact_observed"])),
                ("Collision sensor report", absent(r["physical_collision_observed"])),
            ]
            story.append(Paragraph(
                " | ".join(f"{name}: {value}" for name, value in summary),
                style["SmallNote"],
            ))
            story.append(Spacer(1, 10))
    pdf.build(story)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=(
        ROOT / "results" / "teacher_demo" / "runs"
    ))
    ap.add_argument("--output", type=Path, default=(
        ROOT / "results" / "teacher_demo" / "report"
    ))
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_runs(args.runs_root)
    results = summarize(rows)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    basename = "Perceptinet_Comparative_Data_Report"
    csv_path = args.output / (basename + ".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output / (basename + ".json")
    json_path.write_text(json.dumps({
        "generated_utc": created,
        "run_count": len(rows),
        "by_scenario": results,
        "runs": rows,
        "caveat": "No synthetic measurements; centre distances and "
                  "kinematic envelope contacts are not physical impact data.",
    }, indent=2), encoding="utf-8")
    pdf_path = args.output / (basename + ".pdf")
    made_pdf = make_pdf(pdf_path, results, rows, created)
    print("Completed run records:", len(rows))
    print("CSV:", csv_path)
    print("JSON:", json_path)
    if made_pdf:
        print("PDF:", pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

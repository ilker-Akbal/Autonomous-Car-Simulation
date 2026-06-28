#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = REPO_ROOT / "autonomous_driving" / "outputs" / "teknofest_sim_logs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "report_assets" / "figures"


def warn(message: str) -> None:
    print(f"WARN: {message}", file=sys.stderr)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        warn(f"Log dosyasi bulunamadi: {path}")
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                warn(f"{path.name}:{line_no} JSON okunamadi: {exc}")
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                warn(f"{path.name}:{line_no} kayit dict degil, atlandi")
    return rows


def nested_get(record: dict[str, Any], dotted_path: str) -> Any:
    value: Any = record
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def first_value(record: dict[str, Any], paths: Iterable[str]) -> Any:
    for path in paths:
        value = nested_get(record, path)
        if value is not None:
            return value
    return None


def as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def relative_times(
    rows: list[dict[str, Any]],
    time_paths: Iterable[str] = ("timestamp", "debug.stamp", "plan.stamp", "payload.stamp"),
) -> list[Optional[float]]:
    stamps = [as_float(first_value(row, time_paths)) for row in rows]
    valid = [stamp for stamp in stamps if stamp is not None]
    if not valid:
        return [None for _ in rows]
    start = min(valid)
    return [round(stamp - start, 6) if stamp is not None else None for stamp in stamps]


def collect_series(
    rows: list[dict[str, Any]],
    value_paths: Iterable[str],
    *,
    time_paths: Iterable[str] = ("timestamp", "debug.stamp", "plan.stamp", "payload.stamp"),
) -> tuple[list[float], list[float]]:
    values: list[float] = []
    times: list[float] = []
    fallback_times = relative_times(rows, time_paths)

    for index, row in enumerate(rows):
        value = as_float(first_value(row, value_paths))
        if value is None:
            continue
        time_value = fallback_times[index] if fallback_times[index] is not None else float(index)
        times.append(time_value)
        values.append(value)

    return times, values


def collect_points(
    rows: list[dict[str, Any]],
    x_paths: Iterable[str],
    y_paths: Iterable[str],
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x_value = as_float(first_value(row, x_paths))
        y_value = as_float(first_value(row, y_paths))
        if x_value is None or y_value is None:
            continue
        xs.append(x_value)
        ys.append(y_value)
    return xs, ys


def setup_axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax


def save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"OK: {path}")


def save_placeholder(path: Path, title: str, message: str) -> None:
    warn(f"{title}: {message}")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=13,
    )
    ax.set_axis_off()
    save_plot(fig, path)


def plot_speed(control_rows: list[dict[str, Any]], path: Path) -> None:
    title = "Hiz: Hedef ve Gercek"
    target_t, target_v = collect_series(
        control_rows,
        (
            "target_speed_before_control_mps",
            "debug.target_speed_mps",
            "debug.target_speed",
            "target_speed_before_guard_mps",
        ),
    )
    actual_t, actual_v = collect_series(
        control_rows,
        (
            "debug.current_speed_mps",
            "debug.current_speed",
            "debug.speed_mps",
            "current_speed_mps",
            "speed_mps",
        ),
    )
    if not target_v and not actual_v:
        save_placeholder(path, title, "Hedef veya gercek hiz verisi bulunamadi.")
        return

    fig, ax = setup_axes(title, "Zaman (s)", "Hiz (m/s)")
    if target_v:
        ax.plot(target_t, target_v, label="Hedef hiz", linewidth=2.0)
    else:
        warn(f"{title}: hedef hiz serisi yok")
    if actual_v:
        ax.plot(actual_t, actual_v, label="Gercek hiz", linewidth=2.0)
    else:
        warn(f"{title}: gercek hiz serisi yok")
    ax.legend()
    save_plot(fig, path)


def plot_throttle_brake(control_rows: list[dict[str, Any]], path: Path) -> None:
    title = "Gaz ve Fren Komutlari"
    throttle_t, throttle_v = collect_series(
        control_rows,
        ("command.throttle", "final_vehicle_command.throttle", "debug.final_vehicle_command.throttle"),
    )
    brake_t, brake_v = collect_series(
        control_rows,
        ("command.brake", "final_vehicle_command.brake", "debug.final_vehicle_command.brake"),
    )
    if not throttle_v and not brake_v:
        save_placeholder(path, title, "Gaz veya fren komutu verisi bulunamadi.")
        return

    fig, ax = setup_axes(title, "Zaman (s)", "Komut (0-1)")
    if throttle_v:
        ax.plot(throttle_t, throttle_v, label="Gaz", linewidth=1.8)
    else:
        warn(f"{title}: gaz serisi yok")
    if brake_v:
        ax.plot(brake_t, brake_v, label="Fren", linewidth=1.8)
    else:
        warn(f"{title}: fren serisi yok")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    save_plot(fig, path)


def plot_steering(control_rows: list[dict[str, Any]], path: Path) -> None:
    title = "Direksiyon Komutu"
    steer_t, steer_v = collect_series(
        control_rows,
        ("command.steer", "final_vehicle_command.steer", "debug.final_vehicle_command.steer"),
    )
    if not steer_v:
        save_placeholder(path, title, "Direksiyon komutu verisi bulunamadi.")
        return

    fig, ax = setup_axes(title, "Zaman (s)", "Direksiyon (-1..1)")
    ax.plot(steer_t, steer_v, label="Direksiyon", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    save_plot(fig, path)


def plot_trajectory(lane_rows: list[dict[str, Any]], path: Path) -> None:
    title = "Arac Yorungesi"
    xs, ys = collect_points(
        lane_rows,
        (
            "ego_x",
            "debug.ego_x",
            "debug.ego_center_x",
            "payload.ego_x",
            "plan.ego_x",
            "debug.biased_target_x",
            "plan.biased_target_x",
            "plan.target.x",
            "debug.raw_target_x",
        ),
        (
            "ego_y",
            "debug.ego_y",
            "debug.ego_center_y",
            "payload.ego_y",
            "plan.ego_y",
            "debug.biased_target_y",
            "plan.biased_target_y",
            "plan.target.y",
            "debug.raw_target_y",
        ),
    )
    raw_xs, raw_ys = collect_points(lane_rows, ("debug.raw_target_x",), ("debug.raw_target_y",))
    task_xs, task_ys = collect_points(
        lane_rows,
        ("debug.task_stop_x", "plan.task_stop_x"),
        ("debug.task_stop_y", "plan.task_stop_y"),
    )
    if not xs:
        save_placeholder(path, title, "Yorunge veya hedef nokta koordinati bulunamadi.")
        return

    fig, ax = setup_axes(title, "X (m)", "Y (m)")
    ax.plot(xs, ys, label="Izlenen nokta / ego", linewidth=1.8)
    ax.scatter(xs[0], ys[0], label="Baslangic", s=45, marker="o")
    ax.scatter(xs[-1], ys[-1], label="Bitis", s=45, marker="x")
    if raw_xs:
        ax.plot(raw_xs, raw_ys, label="Ham hedef", linewidth=1.0, alpha=0.5)
    if task_xs:
        ax.scatter(task_xs, task_ys, label="Gorev durak noktasi", s=24, alpha=0.7)
    ax.axis("equal")
    ax.legend()
    save_plot(fig, path)


def plot_mission_distance(lane_rows: list[dict[str, Any]], path: Path) -> None:
    title = "Gorev Yaklasma Mesafesi"
    distance_t, distance_v = collect_series(
        lane_rows,
        (
            "plan.distance_to_goal_m",
            "debug.task_stop_distance_m",
            "debug.distance_to_goal_m",
            "distance_to_goal_m",
        ),
    )
    task_t, task_v = collect_series(
        lane_rows,
        (
            "debug.task_stop_distance_m",
            "plan.task_stop_distance_m",
            "task_stop_distance_m",
        ),
    )
    if not distance_v and not task_v:
        save_placeholder(path, title, "Gorev mesafesi verisi bulunamadi.")
        return

    fig, ax = setup_axes(title, "Zaman (s)", "Mesafe (m)")
    if distance_v:
        ax.plot(distance_t, distance_v, label="Gorev hedefine mesafe", linewidth=1.8)
    if task_v:
        ax.plot(task_t, task_v, label="Gorev durak mesafesi", linewidth=1.8)
    ax.legend()
    save_plot(fig, path)


def plot_traffic_light_distance(route_rows: list[dict[str, Any]], path: Path) -> None:
    title = "Trafik Isigi Mesafe"
    stop_t, stop_v = collect_series(
        route_rows,
        (
            "payload.distance_to_stop_m",
            "debug.selected_distance_to_stop_m",
            "debug.distance_to_stop_m",
            "distance_to_stop_m",
        ),
    )
    light_t, light_v = collect_series(
        route_rows,
        (
            "payload.distance_to_light_m",
            "debug.selected_distance_to_light_m",
            "debug.distance_to_light_m",
            "distance_to_light_m",
        ),
    )
    if not stop_v and not light_v:
        save_placeholder(path, title, "Trafik isigi mesafesi verisi bulunamadi.")
        return

    fig, ax = setup_axes(title, "Zaman (s)", "Mesafe (m)")
    if stop_v:
        ax.plot(stop_t, stop_v, label="Stop cizgisine mesafe", linewidth=1.8)
    if light_v:
        ax.plot(light_t, light_v, label="Trafik isigina mesafe", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.legend()
    save_plot(fig, path)


def plot_route_events(route_rows: list[dict[str, Any]], path: Path) -> None:
    title = "Rota Olaylari"
    events: list[tuple[float, str]] = []
    fallback_times = relative_times(route_rows)
    for index, row in enumerate(route_rows):
        event = as_text(
            first_value(
                row,
                (
                    "payload.event",
                    "debug.selected_event",
                    "debug.event",
                    "event",
                    "plan.route_event",
                ),
            )
        )
        if event is None:
            continue
        time_value = fallback_times[index] if fallback_times[index] is not None else float(index)
        events.append((time_value, event))

    if not events:
        save_placeholder(path, title, "Rota olayi verisi bulunamadi.")
        return

    counts = Counter(event for _, event in events)
    ordered_events = [event for event, _ in counts.most_common()]
    event_to_y = {event: idx for idx, event in enumerate(ordered_events)}

    fig, (ax_timeline, ax_count) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        dpi=150,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_timeline.set_title(title)
    ax_timeline.scatter(
        [time_value for time_value, _ in events],
        [event_to_y[event] for _, event in events],
        s=14,
        alpha=0.75,
    )
    ax_timeline.set_xlabel("Zaman (s)")
    ax_timeline.set_ylabel("Olay")
    ax_timeline.set_yticks(range(len(ordered_events)))
    ax_timeline.set_yticklabels(ordered_events)
    ax_timeline.grid(True, alpha=0.3)

    ax_count.barh(ordered_events, [counts[event] for event in ordered_events])
    ax_count.set_xlabel("Kayit sayisi")
    ax_count.set_ylabel("Olay")
    ax_count.grid(True, axis="x", alpha=0.3)
    save_plot(fig, path)


def resolve_session(log_root: Path, session: Optional[str]) -> Path:
    if session:
        session_path = Path(session).expanduser()
        if not session_path.is_absolute():
            session_path = log_root / session_path
        return session_path

    if not log_root.exists():
        raise FileNotFoundError(f"Log root bulunamadi: {log_root}")

    candidates = [path for path in log_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Log root altinda oturum klasoru yok: {log_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def safe_run(name: str, plotter: Callable[[], None]) -> None:
    try:
        plotter()
    except Exception as exc:
        warn(f"{name} grafigi uretilemedi: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 JSONL loglarindan rapor PNG grafiklerini uretir."
    )
    parser.add_argument(
        "session",
        nargs="?",
        help=(
            "Oturum adi veya oturum klasoru. Verilmezse "
            "autonomous_driving/outputs/teknofest_sim_logs altindaki en yeni oturum secilir."
        ),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f"Log kok dizini (varsayilan: {DEFAULT_LOG_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"PNG cikti dizini (varsayilan: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        session_dir = resolve_session(args.log_root.expanduser(), args.session)
    except FileNotFoundError as exc:
        warn(str(exc))
        return 1

    output_dir = args.output_dir.expanduser()
    print(f"INFO: session={session_dir}")
    print(f"INFO: output_dir={output_dir}")

    control_rows = read_jsonl(session_dir / "control_node.jsonl")
    lane_rows = read_jsonl(session_dir / "lane_follower.jsonl")
    route_rows = read_jsonl(session_dir / "route_event_analyzer.jsonl")

    safe_run(
        "01_hiz_hedef_gercek",
        lambda: plot_speed(control_rows, output_dir / "01_hiz_hedef_gercek.png"),
    )
    safe_run(
        "02_gaz_fren_komutlari",
        lambda: plot_throttle_brake(control_rows, output_dir / "02_gaz_fren_komutlari.png"),
    )
    safe_run(
        "03_direksiyon_komutu",
        lambda: plot_steering(control_rows, output_dir / "03_direksiyon_komutu.png"),
    )
    safe_run(
        "04_arac_yorungesi",
        lambda: plot_trajectory(lane_rows, output_dir / "04_arac_yorungesi.png"),
    )
    safe_run(
        "05_gorev_yaklasma_mesafesi",
        lambda: plot_mission_distance(lane_rows, output_dir / "05_gorev_yaklasma_mesafesi.png"),
    )
    safe_run(
        "06_trafik_isigi_mesafe",
        lambda: plot_traffic_light_distance(route_rows, output_dir / "06_trafik_isigi_mesafe.png"),
    )
    safe_run(
        "07_rota_olaylari",
        lambda: plot_route_events(route_rows, output_dir / "07_rota_olaylari.png"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

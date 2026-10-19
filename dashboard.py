"""Streamlit industrial dashboard for continuous motor health monitoring."""

from __future__ import annotations

import csv
import html
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from model import (
    DEFAULT_HORIZON,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAIN_DATA,
    FEATURE_COLUMNS,
    PROJECT_DIR,
    SensorGRU,
    train_model,
)

DEFAULT_LIVE_DATA = PROJECT_DIR / "data" / "dashboard_live_stream.csv"
DEFAULT_LOG_PATH = PROJECT_DIR / "data" / "dashboard_streamer.log"
HISTORY_LENGTH = 60

st.set_page_config(
    page_title="Motor Health | Industrial Operations",
    page_icon="⚙️",
    layout="wide",
)


@st.cache_resource
def load_monitor_model(
    checkpoint_path: str,
    modified_time_ns: int,
) -> tuple[SensorGRU, np.ndarray, np.ndarray, int, int]:
    """Load a model checkpoint and its training-time normalization values."""
    del modified_time_ns
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("The saved model's sensor features do not match the dashboard.")

    model = SensorGRU(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = checkpoint["scaler_mean"].cpu().numpy()
    scale = checkpoint["scaler_scale"].cpu().numpy()
    if mean.shape != (len(FEATURE_COLUMNS),) or scale.shape != mean.shape:
        raise ValueError("The saved model contains invalid normalization values.")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("The saved model contains invalid normalization values.")
    return model, mean, scale, int(checkpoint["sequence_length"]), int(
        checkpoint["horizon"]
    )


def risk_summary(probability: float) -> tuple[str, str, str]:
    """Return a plain-language risk level, action, and dashboard color."""
    if probability >= 0.75:
        return (
            "Critical",
            "Arrange a controlled shutdown and inspect this motor now.",
            "#f16c62",
        )
    if probability >= 0.50:
        return (
            "High risk",
            "Schedule a maintenance inspection before the next shift.",
            "#f3aa55",
        )
    if probability >= 0.25:
        return (
            "Watch",
            "Check vibration and operating load at the next round.",
            "#f2d36c",
        )
    return (
        "Normal",
        "Continue operation and routine monitoring.",
        "#72c69a",
    )


def stop_simulator() -> None:
    """Stop the bundled sensor simulator, if this dashboard started one."""
    process = st.session_state.get("simulator_process")
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    st.session_state["simulator_process"] = None


def start_monitoring(
    live_path: Path,
    checkpoint_path: Path,
    threshold: float,
    start_simulator: bool,
    simulator_interval: float,
) -> None:
    """Validate the model/source and initialize the live reader."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {checkpoint_path}. Train the model first."
        )

    checkpoint_stat = checkpoint_path.stat()
    load_monitor_model(str(checkpoint_path), checkpoint_stat.st_mtime_ns)

    process = None
    if start_simulator:
        if simulator_interval <= 0:
            raise ValueError("The simulator interval must be greater than zero.")
        live_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(PROJECT_DIR / "standalone_sensor_streamer.py"),
            "--output",
            str(live_path),
            "--interval",
            str(simulator_interval),
            "--count",
            "0",
            "--quiet",
        ]
        with DEFAULT_LOG_PATH.open("ab") as log_file:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                stdout=subprocess.DEVNULL,
                stderr=log_file,
            )
        deadline = time.monotonic() + 5
        while not live_path.is_file() and process.poll() is None:
            if time.monotonic() >= deadline:
                process.terminate()
                process.wait(timeout=3)
                raise TimeoutError("The sensor simulator did not create its CSV file.")
            time.sleep(0.05)

    if not live_path.is_file():
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
        raise FileNotFoundError(
            f"Live sensor CSV not found at {live_path}. Start a data source or choose "
            "an existing CSV."
        )

    st.session_state["simulator_process"] = process
    st.session_state["monitoring"] = True
    st.session_state["monitor"] = {
        "path": live_path,
        "model_path": checkpoint_path,
        "threshold": threshold,
        "offset": 0,
        "headers": None,
        "pending": b"",
        "buffers": defaultdict(lambda: deque(maxlen=30)),
        "latest": {},
        "history": defaultdict(lambda: deque(maxlen=HISTORY_LENGTH)),
        "last_seen": None,
        "rows_seen": 0,
    }


def poll_sensor_data(monitor: dict) -> int:
    """Read new complete CSV rows and refresh per-motor GRU predictions."""
    path: Path = monitor["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Live sensor CSV not found: {path}")
    if path.stat().st_size < monitor["offset"]:
        monitor["offset"] = 0
        monitor["headers"] = None
        monitor["pending"] = b""
        monitor["buffers"].clear()
        monitor["latest"].clear()
        monitor["history"].clear()
        monitor["last_seen"] = None
        monitor["rows_seen"] = 0

    with path.open("rb") as stream:
        if monitor["headers"] is None:
            header_line = stream.readline()
            if not header_line:
                return 0
            headers = next(
                csv.reader([header_line.decode("utf-8-sig").strip()])
            )
            required = {"timestamp", "machine_id", *FEATURE_COLUMNS}
            missing = sorted(required - set(headers))
            if missing:
                raise ValueError(
                    f"Live CSV is missing required columns: {', '.join(missing)}"
                )
            monitor["headers"] = headers
            monitor["offset"] = stream.tell()
        else:
            stream.seek(monitor["offset"])
        chunk = stream.read()
        monitor["offset"] += len(chunk)

    lines = (monitor["pending"] + chunk).split(b"\n")
    monitor["pending"] = lines.pop()
    checkpoint_stat = monitor["model_path"].stat()
    model, scaler_mean, scaler_scale, sequence_length, horizon = load_monitor_model(
        str(monitor["model_path"]), checkpoint_stat.st_mtime_ns
    )
    rows_processed = 0

    for raw_line in lines:
        if not raw_line.strip():
            continue
        values = next(csv.reader([raw_line.decode("utf-8")]))
        headers = monitor["headers"]
        if len(values) != len(headers):
            raise ValueError(
                f"Live CSV row has {len(values)} values; expected {len(headers)}."
            )
        row = dict(zip(headers, values))
        machine_id = row["machine_id"]
        timestamp = row["timestamp"]
        try:
            parsed_timestamp = pd.Timestamp(timestamp)
            if pd.isna(parsed_timestamp):
                raise ValueError("Timestamp is empty.")
            sensor_values = np.asarray(
                [float(row[column]) for column in FEATURE_COLUMNS], dtype=np.float32
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Invalid sensor reading for {machine_id} at {timestamp}."
            ) from error
        if not np.isfinite(sensor_values).all():
            raise ValueError(
                f"Non-finite sensor reading for {machine_id} at {timestamp}."
            )

        normalized_values = (sensor_values - scaler_mean) / scaler_scale
        buffer = monitor["buffers"][machine_id]
        buffer.append(normalized_values)
        monitor["rows_seen"] += 1
        rows_processed += 1
        monitor["last_seen"] = timestamp
        if len(buffer) < sequence_length:
            continue

        inputs = torch.from_numpy(np.asarray(buffer, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            probability = float(torch.sigmoid(model(inputs)).item())
        level, action, color = risk_summary(probability)
        reading = {
            "machine_id": machine_id,
            "timestamp": timestamp,
            "risk": probability,
            "level": level,
            "action": action,
            "color": color,
            "temperature_c": float(sensor_values[0]),
            "vibration_mm_s": float(sensor_values[1]),
            "current_a": float(sensor_values[2]),
            "rpm": float(sensor_values[3]),
            "horizon": horizon,
        }
        monitor["latest"][machine_id] = reading
        monitor["history"][machine_id].append(
            {"time": timestamp, "risk": probability * 100}
        )
    return rows_processed


def render_dashboard() -> None:
    """Render the continuously refreshing plant-health panel."""
    monitor = st.session_state.get("monitor")
    if not st.session_state.get("monitoring") or monitor is None:
        st.info(
            "Monitoring is stopped. Press **Start monitoring** to connect to a live "
            "CSV or launch the built-in sensor simulator."
        )
        return

    try:
        rows_processed = poll_sensor_data(monitor)
    except (
        OSError,
        ValueError,
        UnicodeDecodeError,
        csv.Error,
        RuntimeError,
    ) as error:
        stop_simulator()
        st.session_state["monitoring"] = False
        st.error(f"Monitoring stopped: {error}")
        return

    latest = sorted(
        monitor["latest"].values(),
        key=lambda reading: reading["risk"],
        reverse=True,
    )
    alerts = sum(reading["risk"] >= monitor["threshold"] for reading in latest)
    average_risk = (
        sum(reading["risk"] for reading in latest) / len(latest) if latest else 0.0
    )

    status_column, motor_column, alert_column, risk_column = st.columns(4)
    status_column.metric("System", "LIVE", help="Sensor file is being polled.")
    motor_column.metric("Motors reporting", len(latest))
    alert_column.metric("Motors needing attention", alerts)
    risk_column.metric("Average failure risk", f"{average_risk:.0%}")
    st.caption(
        f"Updated {monitor['last_seen'] or 'waiting for first reading'} · "
        f"{monitor['rows_seen']:,} sensor readings received · "
        f"Alert threshold {monitor['threshold']:.0%}"
    )

    if rows_processed == 0 and not latest:
        st.info(
            "Waiting for sensor readings. The model needs 30 consecutive readings "
            "per motor before it can make a prediction."
        )
        return
    if not latest:
        st.info(
            "Collecting sensor history. Predictions will appear after each motor "
            "has supplied 30 readings."
        )
        return

    st.subheader("Motor condition")
    for start in range(0, len(latest), 2):
        cards = st.columns(2)
        for column, reading in zip(cards, latest[start : start + 2]):
            with column:
                st.markdown(
                    f"""
                    <div style="border:1px solid #263746;border-left:5px solid {reading['color']};
                    border-radius:10px;padding:16px 18px;margin:6px 0 14px;background:#111c27">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:1.2rem;font-weight:700;color:#edf3f8">
                          {html.escape(reading['machine_id'])}
                        </span>
                        <span style="font-weight:700;color:{reading['color']}">
                          {reading['level'].upper()}
                        </span>
                      </div>
                      <div style="font-size:2rem;font-weight:700;color:{reading['color']};margin:8px 0">
                        {reading['risk']:.0%}
                        <span style="font-size:.92rem;font-weight:400;color:#9aaaba">
                          chance of failure within {reading['horizon']} min
                        </span>
                      </div>
                      <div style="color:#d7e0e8">{reading['action']}</div>
                      <div style="color:#9aaaba;font-size:.82rem;margin-top:9px">
                        Temperature {reading['temperature_c']:.1f} °C
                        &nbsp;·&nbsp; Vibration {reading['vibration_mm_s']:.2f} mm/s
                        &nbsp;·&nbsp; {reading['rpm']:.0f} rpm
                        &nbsp;·&nbsp; {reading['current_a']:.1f} A
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.subheader("Recent risk trend")
    trend_rows = [
        {"Time": item["time"], "Motor": machine_id, "Risk (%)": item["risk"]}
        for machine_id, history in monitor["history"].items()
        for item in history
    ]
    if trend_rows:
        trend = pd.DataFrame(trend_rows).pivot(
            index="Time", columns="Motor", values="Risk (%)"
        )
        st.line_chart(trend, height=260)

    st.subheader("Latest readings")
    table_rows = [
        {
            "Motor": reading["machine_id"],
            "Condition": reading["level"],
            "Failure risk": reading["risk"],
            "Latest reading": reading["timestamp"],
            "Temperature (°C)": reading["temperature_c"],
            "Vibration (mm/s)": reading["vibration_mm_s"],
            "RPM": reading["rpm"],
            "Current (A)": reading["current_a"],
            "Recommended action": reading["action"],
        }
        for reading in latest
    ]
    st.dataframe(
        pd.DataFrame(table_rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Failure risk": st.column_config.ProgressColumn(
                "Failure risk", min_value=0.0, max_value=1.0, format="%.0%%"
            )
        },
    )
    st.caption(
        "Predictions are model-based maintenance guidance, not a replacement for "
        "site safety procedures or an engineer's inspection."
    )


st.markdown(
    """
    <style>
      .stApp { background: #0b1118; color: #edf3f8; }
      [data-testid="stSidebar"] { background: #111c27; }
      [data-testid="stMetric"] {
        background: #111c27; border: 1px solid #263746; padding: 14px;
        border-radius: 10px;
      }
      h1, h2, h3 { letter-spacing: .01em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚙️ Motor health control room")
st.caption(
    "Predictive maintenance · GRU sensor monitoring · "
    f"{DEFAULT_HORIZON}-minute failure-risk outlook"
)

if "monitoring" not in st.session_state:
    st.session_state["monitoring"] = False
if "simulator_process" not in st.session_state:
    st.session_state["simulator_process"] = None

with st.sidebar:
    st.header("Operations")
    live_path_text = st.text_input("Live sensor CSV", str(DEFAULT_LIVE_DATA))
    model_path_text = st.text_input("Trained model", str(DEFAULT_MODEL_PATH))
    threshold = st.slider(
        "Attention alert threshold",
        min_value=10,
        max_value=95,
        value=50,
        step=5,
        format="%d%%",
        help="Motors at or above this predicted risk are counted as needing attention.",
    )
    start_simulator = st.checkbox(
        "Launch built-in sensor simulator",
        value=True,
        help=(
            "Generates demo readings in the separate dashboard_live_stream.csv file. "
            "Uncheck to monitor an existing live sensor CSV."
        ),
    )
    simulator_interval = st.slider(
        "Simulator interval (seconds)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        disabled=not start_simulator,
        help="Wall-clock interval between simulated sensor readings.",
    )

    start_column, stop_column = st.columns(2)
    with start_column:
        start_clicked = st.button(
            "▶ Start monitoring",
            type="primary",
            use_container_width=True,
            disabled=st.session_state["monitoring"],
        )
    with stop_column:
        stop_clicked = st.button(
            "■ Stop",
            use_container_width=True,
            disabled=not st.session_state["monitoring"],
        )

    st.divider()
    st.subheader("Model training")
    st.caption(f"Training data: {DEFAULT_TRAIN_DATA.name}")
    epochs = st.number_input("Training epochs", min_value=1, max_value=100, value=15)
    train_clicked = st.button(
        "Train model",
        use_container_width=True,
        disabled=st.session_state["monitoring"],
    )
    st.caption("Training uses normalized sensor readings and chronological splits.")

if stop_clicked:
    stop_simulator()
    st.session_state["monitoring"] = False
    st.rerun()

if start_clicked:
    try:
        start_monitoring(
            Path(live_path_text),
            Path(model_path_text),
            threshold / 100,
            start_simulator,
            simulator_interval,
        )
        st.rerun()
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        stop_simulator()
        st.error(f"Could not start monitoring: {error}")

if train_clicked:
    try:
        with st.spinner("Training the GRU model on motor sensor history..."):
            result = train_model(
                csv_path=DEFAULT_TRAIN_DATA,
                model_path=Path(model_path_text),
                epochs=int(epochs),
            )
        load_monitor_model.clear()
        metrics = result["test_metrics"]
        st.success(
            "Training complete. "
            f"Test accuracy {metrics['accuracy']:.1%}, "
            f"recall {metrics['recall']:.1%}, "
            f"precision {metrics['precision']:.1%}."
        )
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Training failed: {error}")

@st.fragment(run_every="1s")
def live_panel() -> None:
    render_dashboard()


live_panel()

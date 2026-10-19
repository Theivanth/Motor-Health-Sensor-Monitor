"""Train and run a GRU monitor for motor sensor readings."""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_DATA = PROJECT_DIR / "synthetic_motors.csv"
DEFAULT_MONITOR_DATA = PROJECT_DIR / "data" / "live_stream.csv"
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "motor_gru.pt"

FEATURE_COLUMNS = ["temperature_c", "vibration_mm_s", "current_a", "rpm"]
REQUIRED_COLUMNS = [
    "timestamp",
    "machine_id",
    *FEATURE_COLUMNS,
    "failure_event",
]
DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_HORIZON = 60


class SensorGRU(nn.Module):
    """Classify whether a motor may fail within the forecast horizon."""

    def __init__(self, input_size: int = len(FEATURE_COLUMNS), hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.gru(inputs)
        return self.classifier(sequence_output[:, -1, :]).squeeze(-1)


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate sensor readings from a CSV file."""
    df = pd.read_csv(csv_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
    if df.empty:
        raise ValueError(f"CSV contains no sensor readings: {csv_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("CSV contains missing or invalid timestamps.")
    if df["machine_id"].isna().any():
        raise ValueError("CSV contains missing machine IDs.")

    for column in [*FEATURE_COLUMNS, "failure_event"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(f"CSV column '{column}' contains missing or non-numeric values.")

    if not df["failure_event"].isin([0, 1]).all():
        raise ValueError("'failure_event' must contain only 0 or 1.")

    df["machine_id"] = df["machine_id"].astype(str)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    if df.duplicated(["machine_id", "timestamp"]).any():
        raise ValueError("CSV contains duplicate timestamps for a machine.")
    return df


def _make_splits(
    normalized_df: pd.DataFrame,
    sequence_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create chronological, per-machine windows and their future failure labels."""
    windows: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    labels: dict[str, list[float]] = {"train": [], "validation": [], "test": []}

    for _, machine_df in normalized_df.groupby("machine_id", sort=False):
        sensor_values = machine_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        failure_events = machine_df["failure_event"].to_numpy(dtype=np.int64)
        row_count = len(machine_df)
        train_end = int(row_count * 0.70)
        validation_end = int(row_count * 0.85)

        if train_end <= sequence_length + horizon or validation_end >= row_count:
            raise ValueError(
                f"Machine {machine_df['machine_id'].iloc[0]} does not have enough "
                "readings for the selected sequence length and horizon."
            )

        last_start = row_count - sequence_length - horizon
        for start in range(last_start + 1):
            first_future_row = start + sequence_length
            target_end = first_future_row + horizon
            if target_end <= train_end:
                split = "train"
            elif target_end <= validation_end:
                split = "validation"
            else:
                split = "test"

            windows[split].append(sensor_values[start:first_future_row])
            labels[split].append(
                float(failure_events[first_future_row:target_end].any())
            )

    if not windows["train"] or not windows["validation"] or not windows["test"]:
        raise ValueError("The dataset is too small to create train, validation, and test windows.")

    return (
        np.asarray(windows["train"], dtype=np.float32),
        np.asarray(labels["train"], dtype=np.float32),
        np.asarray(windows["validation"], dtype=np.float32),
        np.asarray(labels["validation"], dtype=np.float32),
        np.asarray(windows["test"], dtype=np.float32),
        np.asarray(labels["test"], dtype=np.float32),
    )


def _evaluate(
    model: SensorGRU,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    loss_function: nn.Module,
) -> tuple[float, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(features).to(device)
        targets = torch.from_numpy(labels).to(device)
        logits = model(inputs)
        probabilities = torch.sigmoid(logits)
        loss = loss_function(logits, targets).item()
        predictions = probabilities >= 0.5
        actuals = targets >= 0.5

        true_positives = int((predictions & actuals).sum().item())
        false_positives = int((predictions & ~actuals).sum().item())
        false_negatives = int((~predictions & actuals).sum().item())
        correct = int((predictions == actuals).sum().item())
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        return loss, {
            "accuracy": correct / len(labels),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def train_model(
    csv_path: str | Path = DEFAULT_TRAIN_DATA,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    horizon: int = DEFAULT_HORIZON,
    epochs: int = 15,
    batch_size: int = 128,
    learning_rate: float = 0.001,
) -> dict[str, object]:
    """Normalize training data, fit the GRU, and save a monitor-ready checkpoint."""
    if sequence_length < 1 or horizon < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("Sequence length, horizon, epochs, and batch size must be positive.")

    df = load_data(csv_path)
    machine_groups = list(df.groupby("machine_id", sort=False))
    training_readings = []
    for _, machine_df in machine_groups:
        train_end = int(len(machine_df) * 0.70)
        if train_end <= sequence_length + horizon:
            raise ValueError("The dataset does not have enough training readings per machine.")
        training_readings.append(machine_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)[:train_end])

    training_values = np.concatenate(training_readings, axis=0)
    scaler_mean = training_values.mean(axis=0)
    scaler_scale = training_values.std(axis=0)
    scaler_scale[scaler_scale == 0] = 1.0

    # Keep the normalized readings in df so training and inspection use the same data.
    df.loc[:, FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS] - scaler_mean
    ) / scaler_scale
    x_train, y_train, x_validation, y_validation, x_test, y_test = _make_splits(
        df, sequence_length, horizon
    )
    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    if positive_count == 0:
        raise ValueError(
            "No training windows contain a failure in the selected horizon; "
            "the RNN cannot learn a failure class."
        )

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SensorGRU().to(device)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        validation_loss, validation_metrics = _evaluate(
            model, x_validation, y_validation, device, loss_function
        )
        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={total_loss / len(y_train):.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_recall={validation_metrics['recall']:.3f}"
        )

    test_loss, test_metrics = _evaluate(model, x_test, y_test, device, loss_function)
    checkpoint_path = Path(model_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "sequence_length": sequence_length,
            "horizon": horizon,
            "scaler_mean": torch.tensor(scaler_mean, dtype=torch.float32),
            "scaler_scale": torch.tensor(scaler_scale, dtype=torch.float32),
        },
        checkpoint_path,
    )
    print(
        f"Test loss={test_loss:.4f} "
        f"accuracy={test_metrics['accuracy']:.3f} "
        f"precision={test_metrics['precision']:.3f} "
        f"recall={test_metrics['recall']:.3f} "
        f"f1={test_metrics['f1']:.3f}"
    )
    print(f"Saved trained model to {checkpoint_path.resolve()}")

    return {"df": df, "model": model, "test_metrics": test_metrics}


def _read_csv_header(csv_path: Path) -> tuple[list[str], int]:
    """Read and validate a live CSV header, returning its byte offset."""
    with csv_path.open("rb") as stream:
        header_line = stream.readline()
        if not header_line:
            raise ValueError(f"Live CSV has no header: {csv_path}")
        headers = next(csv.reader([header_line.decode("utf-8-sig").strip()]))
        missing_columns = sorted(set(REQUIRED_COLUMNS[:-1]) - set(headers))
        if missing_columns:
            raise ValueError(
                f"Live CSV is missing required columns: {', '.join(missing_columns)}"
            )
        return headers, stream.tell()


def monitor_csv(
    csv_path: str | Path = DEFAULT_MONITOR_DATA,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = 0.5,
    poll_interval: float = 1.0,
) -> None:
    """Continuously score new CSV rows, maintaining one sensor window per motor."""
    if not 0 <= threshold <= 1:
        raise ValueError("The alert threshold must be between 0 and 1.")
    if poll_interval <= 0:
        raise ValueError("The polling interval must be greater than zero.")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    feature_columns = checkpoint["feature_columns"]
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError("The model checkpoint sensor features do not match this script.")
    sequence_length = int(checkpoint["sequence_length"])
    horizon = int(checkpoint["horizon"])
    scaler_mean = checkpoint["scaler_mean"].numpy()
    scaler_scale = checkpoint["scaler_scale"].numpy()

    model = SensorGRU(input_size=len(feature_columns))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    live_path = Path(csv_path)
    headers, byte_offset = _read_csv_header(live_path)
    buffers: dict[str, deque[np.ndarray]] = defaultdict(
        lambda: deque(maxlen=sequence_length)
    )
    pending = b""
    print(f"Monitoring {live_path.resolve()} (Ctrl+C to stop)")

    while True:
        if live_path.stat().st_size < byte_offset:
            headers, byte_offset = _read_csv_header(live_path)
            pending = b""
            buffers.clear()
            print("Live CSV was truncated; restarted sensor windows.")

        with live_path.open("rb") as stream:
            stream.seek(byte_offset)
            chunk = stream.read()
        byte_offset += len(chunk)

        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        for raw_line in lines:
            if not raw_line.strip():
                continue
            row_values = next(csv.reader([raw_line.decode("utf-8")]))
            if len(row_values) != len(headers):
                raise ValueError(
                    f"Live CSV row has {len(row_values)} values; expected {len(headers)}."
                )
            row = dict(zip(headers, row_values))
            machine_id = row["machine_id"]
            timestamp = row["timestamp"]
            try:
                sensor_values = np.asarray(
                    [float(row[column]) for column in feature_columns], dtype=np.float32
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid sensor reading for machine {machine_id} at {timestamp}."
                ) from error
            if not np.isfinite(sensor_values).all():
                raise ValueError(
                    f"Non-finite sensor reading for machine {machine_id} at {timestamp}."
                )

            normalized_values = (sensor_values - scaler_mean) / scaler_scale
            machine_buffer = buffers[machine_id]
            machine_buffer.append(normalized_values)
            if len(machine_buffer) < sequence_length:
                continue

            inputs = torch.from_numpy(np.asarray(machine_buffer, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                probability = torch.sigmoid(model(inputs)).item()
            status = "ALERT" if probability >= threshold else "OK"
            print(
                f"{status} {timestamp} {machine_id}: "
                f"failure risk within {horizon} min = {probability:.1%}"
            )

        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "monitor"), nargs="?", default="train")
    parser.add_argument("--data", type=Path, help="Training CSV or live monitoring CSV.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            csv_path=args.data or DEFAULT_TRAIN_DATA,
            model_path=args.model_path,
            sequence_length=args.sequence_length,
            horizon=args.horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    else:
        monitor_csv(
            csv_path=args.data or DEFAULT_MONITOR_DATA,
            model_path=args.model_path,
            threshold=args.threshold,
            poll_interval=args.poll_interval,
        )


if __name__ == "__main__":
    main()

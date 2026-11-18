import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create a single sensor traffic speed time series."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required_columns = [
        "timestamp",
        "sensor_id",
        "traffic_speed"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df["traffic_speed"] = pd.to_numeric(
        df["traffic_speed"], errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "sensor_id", "traffic_speed"]
    )

    sensor_counts = df["sensor_id"].value_counts()

    if sensor_counts.empty:
        raise ValueError("No valid sensor readings found.")

    selected_sensor = sensor_counts.idxmax()

    sensor_df = df[
        df["sensor_id"] == selected_sensor
    ].copy()

    sensor_df = sensor_df.sort_values("timestamp")

    result = sensor_df[
        ["timestamp", "traffic_speed"]
    ].reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    result.to_csv(output_path, index=False)

    print("Selected Sensor:", selected_sensor)
    print("Total Readings:", len(result))
    print("Saved:", output_path)
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
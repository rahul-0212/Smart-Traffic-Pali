import argparse
from pathlib import Path
import pandas as pd

def load_wide_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf", ".hdf5"}:
        return pd.read_hdf(path)
    if suffix == ".csv":
        # Most benchmark CSVs have timestamp in the first column.
        df = pd.read_csv(path)
        first = df.columns[0]
        parsed = pd.to_datetime(df[first], errors="coerce")
        if parsed.notna().mean() > 0.8:
            df = df.drop(columns=[first])
            df.index = parsed
        else:
            # Also support CSVs where timestamps are saved as the index.
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, errors="coerce")
        return df
    raise ValueError("Input must be .h5, .hdf5, or .csv")

def main():
    parser = argparse.ArgumentParser(
        description="Convert wide traffic benchmark data into long-format CSV."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="public-traffic")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    wide = load_wide_table(input_path)
    if not isinstance(wide, pd.DataFrame):
        raise TypeError("Input did not load as a pandas DataFrame.")

    wide.index = pd.to_datetime(wide.index, errors="coerce")
    wide = wide.loc[wide.index.notna()].sort_index()
    wide = wide.apply(pd.to_numeric, errors="coerce")

    print("Input shape (timestamps x sensors):", wide.shape)
    long_df = (
        wide.rename_axis("timestamp")
        .reset_index()
        .melt(id_vars="timestamp", var_name="sensor_id",
              value_name="traffic_speed")
    )
    long_df = long_df.dropna(subset=["timestamp", "traffic_speed"])
    long_df["dataset_name"] = args.dataset_name

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output_path, index=False)
    print("Saved:", output_path)
    print("Rows:", len(long_df))
    print(long_df.head())
    print("Reminder: benchmark target is traffic speed, not vehicle volume.")

if __name__ == "__main__":
    main()

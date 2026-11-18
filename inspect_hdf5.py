import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    print("HDF5 store keys:")
    with pd.HDFStore(args.input, mode="r") as store:
        print(store.keys())
        for key in store.keys():
            print(f"\n--- {key} ---")
            try:
                obj = store.get(key)
                print("Type:", type(obj))
                if hasattr(obj, "shape"):
                    print("Shape:", obj.shape)
                if isinstance(obj, pd.DataFrame):
                    print(obj.head())
            except Exception as exc:
                print("Could not read key:", exc)

if __name__ == "__main__":
    main()

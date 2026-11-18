# Phase 2 – Public Traffic Dataset Preparation (Windows)

This phase prepares public traffic data and converts it into a clean CSV format.
It does not yet claim to model Pali traffic. METR-LA / PEMS-BAY are external
benchmark datasets. Local Pali observations will be collected in a later phase.

## A. Windows setup

Open PowerShell in this folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

## B. Download data

Use the official DCRNN repository's dataset instructions and linked data source.
The common METR-LA / PEMS-BAY HDF5 files are often named `metr-la.h5` and
`pems-bay.h5`. Put the file you choose into `data/raw/`.

Repository:
https://github.com/liyaguang/DCRNN

Public dataset record:
https://zenodo.org/records/5146275

Check the dataset's licence, citation, file contents, timestamps, and sensor
metadata before using it in a submitted report.

## C. Run the preparation script

For METR-LA:
```powershell
python prepare_public_dataset.py --input data/raw/METR-LA.csv --output data/processed/metr_la_long.csv --dataset-name METR-LA
```

For PEMS-BAY:
```powershell
python prepare_public_dataset.py --input data/raw/PEMS-BAY.csv --output data/processed/pems_bay_long.csv --dataset-name PEMS-BAY
```

The script supports a wide-format HDF5 table or wide-format CSV. If your file
uses a different structure, inspect it with:
```powershell
python inspect_hdf5.py --input data/raw/metr-la.h5
```

## D. Understand the output

Output columns:
- timestamp: observation time
- sensor_id: source sensor identifier
- traffic_speed: recorded speed value
- dataset_name: public dataset name

This is a long-format table. Each row is one sensor at one timestamp.

## E. Build a single sensor time series for the starter model

Choose one sensor with enough complete records:
```powershell
python make_single_sensor_series.py --input data/processed/metr_la_long.csv --output data/processed/traffic_training.csv
```

The starter model expects a target named `traffic_volume`. METR-LA and PEMS-BAY
are speed datasets, not vehicle-count datasets. The script therefore creates
`traffic_speed` as the target; do not rename speed to volume. For this dataset,
use the speed-specific training script `train_speed_model.py`.

For true traffic-volume prediction, obtain a dataset with vehicle counts/flow
and keep its units and interval documented.

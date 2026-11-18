# 1. Libraries import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# 2. Machine Learning model
from sklearn.ensemble import RandomForestRegressor

# 3. Evaluation metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 4. Dataset read
df = pd.read_csv(
    "data/processed/traffic_training.csv"
)

# 5. Timestamp convert
df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# 6. Speed numeric convert
df["target"] = pd.to_numeric(
    df["target"], errors="coerce"
)

# 7. Missing values remove
df = df.dropna(
    subset=["timestamp", "target"]
)

# 8. Time order
df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

# 9. Current time features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.dayofweek

# 10. Previous speed features
for lag in [1, 2, 3, 6, 12]:
    df[f"lag_{lag}"] = df["target"].shift(lag)

# 11. Input features
features = [
    "hour", "day",
    "lag_1", "lag_2",
    "lag_3", "lag_6", "lag_12"
]

# 12. Create model folder
os.makedirs("models", exist_ok=True)

# 13. Forecasting different time periods
horizons = {
    "5 Minutes": 1,
    "15 Minutes": 3,
    "30 Minutes": 6
}

results = {}

for name, steps in horizons.items():

    print("\nTraining:", name)

    # Future speed target
    data = df.copy()

    data["future_speed"] = data["target"].shift(-steps)

    # Remove missing values
    data = data.dropna().reset_index(drop=True)

    X = data[features]
    y = data["future_speed"]

    # Chronological split
    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Accuracy metrics
    mae = mean_absolute_error(
        y_test, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))

    # Save model
    joblib.dump(
        {
            "model": model,
            "features": features,
            "horizon": name
        },
        f"models/model_{steps}_steps.joblib"
    )

    # Actual vs Predicted graph
    plt.figure(figsize=(12, 5))

    plt.plot(
        y_test.iloc[:200].to_numpy(),
        label="Actual Speed"
    )

    plt.plot(
        predictions[:200],
        label="Predicted Speed"
    )

    plt.title(f"Traffic Speed Prediction - {name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Traffic Speed")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"models/graph_{steps}_steps.png"
    )

    plt.show()

# Final results
print("\nFinal Forecasting Results")

for name, score in results.items():
    print(name, score)

print("\nPhase 4 Completed Successfully!")
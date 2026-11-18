# 1. Libraries import karna
import pandas as pd
import numpy as np
import joblib

# 2. Machine Learning models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# 3. Data evaluation ke tools
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 4. Dataset read karna
df = pd.read_csv(
    "data/processed/traffic_training.csv"
)

# 5. Timestamp ko date-time format me badalna
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 6. Target ko numeric banana
df["target"] = pd.to_numeric(
    df["target"], errors="coerce"
)

# 7. Missing values remove karna
df = df.dropna(
    subset=["timestamp", "target"]
)

# 8. Time ke according sort karna
df = df.sort_values("timestamp").reset_index(drop=True)

# 9. Time features banana
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.dayofweek

# 10. Previous traffic speed values
for lag in [1, 2, 3, 6, 12]:
    df[f"lag_{lag}"] = df["target"].shift(lag)

# 11. Missing lag rows remove karna
df = df.dropna().reset_index(drop=True)

# 12. Input features select karna
features = [
    "hour",
    "day",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12"
]

X = df[features]

# 13. Output target select karna
y = df["target"]

# 14. Training aur Testing split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 15. Models banana
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=100,
        random_state=42
    )
}

# 16. Har model ko train aur test karna
results = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test, predictions
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))

# 17. Best model choose karna
best_name = min(
    results,
    key=lambda name: results[name]["MAE"]
)

best_model = models[best_name]

# 18. Model save karna
joblib.dump(
    {
        "model": best_model,
        "model_name": best_name,
        "features": features,
        "metrics": results,
        "target": "traffic_speed"
    },
    "models/best_speed_model.joblib"
)

print("\nBest Model:", best_name)
print("Model saved successfully!")
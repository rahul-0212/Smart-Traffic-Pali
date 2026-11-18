# 1. Libraries import karna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# 2. TensorFlow import karna
import tensorflow as tf

# 3. LSTM model ke components
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 4. Data file read karna
df = pd.read_csv(
    "data/processed/traffic_training.csv"
)

# 5. Timestamp ko date-time banana
df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# 6. Traffic speed ko number banana
df["target"] = pd.to_numeric(
    df["target"], errors="coerce"
)

# 7. Invalid values remove karna
df = df.dropna(
    subset=["timestamp", "target"]
)

# 8. Time ke according arrange karna
df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

# 9. Speed values lena
values = df["target"].to_numpy(
    dtype=np.float32
).reshape(-1, 1)

# 10. Check karna ki data enough hai
if len(values) < 100:
    raise ValueError(
        "Dataset me bahut kam readings hain."
    )

# 11. Training aur testing boundary
split_index = int(len(values) * 0.8)

# 12. Training data ka minimum aur maximum
train_min = values[:split_index].min()
train_max = values[:split_index].max()

# 13. Data ko 0 se 1 ke beech scale karna
if train_max == train_min:
    raise ValueError(
        "Training data me speed change nahi ho rahi."
    )

scaled = (
    (values - train_min) /
    (train_max - train_min)
)

# 14. Pichhli 12 readings ka use
window_size = 12

X = []
y = []
target_positions = []

# 15. Sequence banana
for i in range(window_size, len(scaled)):

    # Pichhli 12 readings
    X.append(
        scaled[i-window_size:i]
    )

    # Agli reading ka target
    y.append(
        scaled[i, 0]
    )

    target_positions.append(i)

# 16. NumPy arrays banana
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
target_positions = np.array(target_positions)

# 17. Chronological train-test split
train_mask = target_positions < split_index
test_mask = target_positions >= split_index

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("Training sequences:", len(X_train))
print("Testing sequences:", len(X_test))

# 18. LSTM model banana
model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(window_size, 1)
    ),

    Dropout(0.2),

    LSTM(32),

    Dense(1)
])

# 19. Model compile karna
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# 20. Training jaldi rokne ka option
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# 21. Model training
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

# 22. Test data par prediction
pred_scaled = model.predict(
    X_test
).reshape(-1)

# 23. Original speed scale me wapas lana
predictions = (
    pred_scaled * (train_max - train_min)
    + train_min
)

actual = (
    y_test * (train_max - train_min)
    + train_min
)

# 24. Evaluation metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(
    actual, predictions
)

rmse = np.sqrt(
    mean_squared_error(actual, predictions)
)

r2 = r2_score(
    actual, predictions
)

print("\nLSTM RESULTS")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 3))

# 25. Models folder banana
os.makedirs("models", exist_ok=True)

# 26. LSTM model save karna
model.save(
    "models/traffic_lstm.keras"
)

# 27. Scaling information save karna
joblib.dump(
    {
        "train_min": float(train_min),
        "train_max": float(train_max),
        "window_size": window_size
    },
    "models/lstm_scaler.joblib"
)

# 28. Actual vs Predicted graph
plt.figure(figsize=(12, 5))

plt.plot(
    actual[:200],
    label="Actual Speed"
)

plt.plot(
    predictions[:200],
    label="Predicted Speed"
)

plt.title("LSTM Traffic Speed Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Traffic Speed")
plt.legend()
plt.tight_layout()

plt.savefig(
    "models/lstm_prediction.png"
)

plt.show()

# 29. Training loss graph
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training Performance")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    "models/lstm_training_loss.png"
)

plt.show()

print("\nLSTM Model Successfully Saved!")
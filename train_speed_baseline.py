import pandas as pd
import matplotlib.pyplot as plt

# Dataset read karna
df = pd.read_csv(
    "data/processed/traffic_training.csv"
)

# Dono graphs ke liye jagah banana
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Graph 1: Traffic Speed
axes[0].plot(df["target"])
axes[0].set_title("Traffic Speed Analysis")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Traffic Speed")

# Graph 2: Speed Distribution
axes[1].hist(df["target"], bins=30)
axes[1].set_title("Traffic Speed Distribution")
axes[1].set_xlabel("Traffic Speed")
axes[1].set_ylabel("Frequency")

# Graphs ke beech space
plt.tight_layout()

# Dono graphs show karna
plt.show()
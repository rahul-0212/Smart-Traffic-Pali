import pandas as pd
import os

# Pali data ke liye folder banana
os.makedirs("data/pali", exist_ok=True)

# CSV file ka path
file_path = "data/pali/pali_traffic_data.csv"

# Dataset ke columns
columns = [
    "timestamp",
    "location_id",
    "speed_kmph",
    "vehicle_count",
    "weather",
    "road_condition",
    "source"
]

# Agar file pehle se nahi hai, tabhi create karo
if not os.path.exists(file_path):
    df = pd.DataFrame(columns=columns)
    df.to_csv(file_path, index=False)
    print("Pali dataset successfully created!")
else:
    print("Pali dataset already exists!")

print("File location:", file_path)
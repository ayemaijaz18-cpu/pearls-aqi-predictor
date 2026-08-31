import hopsworks
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# 1. Connect
# -----------------------------
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

print("Connected to Hopsworks!")

# -----------------------------
# 2. Load data
# -----------------------------
fg = fs.get_feature_group(
    name="aqi_features_v3",
    version=1
)

df = fg.select_all().read()

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# -----------------------------
# 3. Create lag features
# -----------------------------
df["aqi_lag_1"] = df["us_aqi"].shift(1)
df["aqi_lag_3"] = df["us_aqi"].shift(3)
df["aqi_lag_6"] = df["us_aqi"].shift(6)
df["aqi_lag_12"] = df["us_aqi"].shift(12)
df["aqi_lag_24"] = df["us_aqi"].shift(24)

df = df.dropna().reset_index(drop=True)

latest = df.iloc[-1]

# -----------------------------
# 4. Features
# -----------------------------
features = [
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "hour",
    "day",
    "month",
    "day_of_week"
]

X = pd.DataFrame(
    [latest[features].values],
    columns=features
)

# -----------------------------
# 5. Function to load model
# -----------------------------
def load_model(model_name):
    model = mr.get_model(
        name=model_name,
        version=1
    )

    model_dir = model.download()

    model_files = list(Path(model_dir).glob("*.pkl"))

    return joblib.load(model_files[0])


# -----------------------------
# 6. Load all 3 models
# -----------------------------
model_24 = load_model("aqi_predictor_24h")
model_48 = load_model("aqi_predictor_48h")
model_72 = load_model("aqi_predictor_72h")

print("All 3 registered models loaded!")

# -----------------------------
# 7. Predict
# -----------------------------
prediction_24 = model_24.predict(X)[0]
prediction_48 = model_48.predict(X)[0]
prediction_72 = model_72.predict(X)[0]

# -----------------------------
# 8. Display forecast
# -----------------------------
print()
print("=============================")
print("LAHORE AQI 3-DAY FORECAST")
print("=============================")
print("Latest AQI :", latest["us_aqi"])
print("24 hours   :", round(float(prediction_24), 2))
print("48 hours   :", round(float(prediction_48), 2))
print("72 hours   :", round(float(prediction_72), 2))
print("=============================")
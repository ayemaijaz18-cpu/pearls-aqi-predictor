import hopsworks
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1. Connect to Hopsworks
# -----------------------------
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

print("Connected to Hopsworks!")


# -----------------------------
# 2. Load Feature Group
# -----------------------------
fg = fs.get_feature_group(
    name="aqi_features_v3",
    version=1
)

df = fg.select_all().read()

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

print("Data loaded!")
print("Rows:", len(df))


# -----------------------------
# 3. Create lag features
# -----------------------------
df["aqi_lag_1"] = df["us_aqi"].shift(1)
df["aqi_lag_3"] = df["us_aqi"].shift(3)
df["aqi_lag_6"] = df["us_aqi"].shift(6)
df["aqi_lag_12"] = df["us_aqi"].shift(12)
df["aqi_lag_24"] = df["us_aqi"].shift(24)


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


# -----------------------------
# 5. Train each horizon
# -----------------------------
for hours_ahead in [24, 48, 72]:

    print()
    print("=============================")
    print(f"TRAINING {hours_ahead}-HOUR MODEL")
    print("=============================")

    # Future AQI target
    df["target_aqi"] = df["us_aqi"].shift(-hours_ahead)

    data = df[features + ["target_aqi"]].dropna()

    X = data[features]
    y = data["target_aqi"]

    print("Usable rows:", len(data))

    # Time-based split
    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # -----------------------------
    # Train
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.3f}")

    # -----------------------------
    # Save locally
    # -----------------------------
    model_path = f"/hopsfs/Users/ayemaija/aqi_model_{hours_ahead}h.pkl"

    joblib.dump(model, model_path)

    print("Saved:", model_path)

    # -----------------------------
    # Register model
    # -----------------------------
    model_name = f"aqi_predictor_{hours_ahead}h"

    registered_model = mr.python.create_model(
        name=model_name,
        metrics={
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        },
        description=f"Lahore AQI forecasting model for {hours_ahead} hours ahead"
    )

    registered_model.save(model_path)

    print("Registered:", model_name)


print()
print("=============================")
print("ALL FORECAST MODELS COMPLETE")
print("=============================")
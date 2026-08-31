import hopsworks
import pandas as pd
import requests

# -----------------------------
# 1. Connect to Hopsworks
# -----------------------------
project = hopsworks.login()
fs = project.get_feature_store()

print("Connected to Hopsworks!")

# -----------------------------
# 2. Lahore coordinates
# -----------------------------
LATITUDE = 31.5204
LONGITUDE = 74.3587

START_DATE = "2026-05-01"
END_DATE = "2026-08-26"

# -----------------------------
# 3. AQI data
# -----------------------------
aqi_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    f"&start_date={START_DATE}"
    f"&end_date={END_DATE}"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
    "ozone,sulphur_dioxide,us_aqi"
    "&timezone=Asia%2FKarachi"
)

aqi_response = requests.get(aqi_url, timeout=60)
aqi_response.raise_for_status()

aqi_data = aqi_response.json()

aqi_df = pd.DataFrame(aqi_data["hourly"])

# -----------------------------
# 4. Weather data
# -----------------------------
weather_url = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    f"&start_date={START_DATE}"
    f"&end_date={END_DATE}"
    "&hourly=temperature_2m,relative_humidity_2m,"
    "wind_speed_10m,precipitation"
    "&timezone=Asia%2FKarachi"
)
weather_response = requests.get(weather_url, timeout=60)
weather_response.raise_for_status()

weather_data = weather_response.json()

weather_df = pd.DataFrame(weather_data["hourly"])

# -----------------------------
# 5. Combine AQI + weather
# -----------------------------
df = pd.merge(
    aqi_df,
    weather_df,
    on="time",
    how="inner"
)

# -----------------------------
# 6. Add city
# -----------------------------
df["city"] = "Lahore"

# -----------------------------
# 7. Time features
# -----------------------------
df["time"] = pd.to_datetime(df["time"])

df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["day_of_week"] = df["time"].dt.dayofweek

# -----------------------------
# 8. AQI change rate
# -----------------------------
df["aqi_change_rate"] = df["us_aqi"].diff()

# Remove first row because it has no previous AQI
df = df.dropna().reset_index(drop=True)

print("Combined AQI + weather data!")
print("Rows:", len(df))
print(df.head())

# -----------------------------
# 9. Insert into new Feature Group
# -----------------------------
aqi_fg = fs.get_or_create_feature_group(
    name="aqi_features_v3",
    version=1,
    primary_key=["time"],
    description="AQI and weather features for Lahore"
)

aqi_fg.insert(df)

print("AQI + weather features successfully inserted!")
import hopsworks
import pandas as pd
import requests

# -----------------------------
# 1. Connect to Hopsworks
# -----------------------------
project = hopsworks.login()
fs = project.get_feature_store()

print("Connected to Hopsworks successfully!")

# -----------------------------
# 2. Lahore coordinates
# -----------------------------
LATITUDE = 31.5204
LONGITUDE = 74.3587

# -----------------------------
# 3. Fetch AQI + weather data
# -----------------------------
url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
    "ozone,sulphur_dioxide,us_aqi"
    "&timezone=Asia%2FKarachi"
)

response = requests.get(url, timeout=30)
response.raise_for_status()

data = response.json()

df = pd.DataFrame(data["hourly"])

print("AQI data downloaded successfully!")
print(df.head())

# -----------------------------
# 4. Add city
# -----------------------------
df["city"] = "Lahore"

# -----------------------------
# 5. Convert time
# -----------------------------
df["time"] = pd.to_datetime(df["time"])

# -----------------------------
# 6. Create feature group
# -----------------------------
aqi_fg = fs.get_or_create_feature_group(
    name="aqi_features_v2",
    version=1,
    primary_key=["time"],
    description="Hourly AQI features for Lahore"
)

# -----------------------------
# 7. Insert data
# -----------------------------
aqi_fg.insert(df)

print("AQI data successfully inserted into Hopsworks!")

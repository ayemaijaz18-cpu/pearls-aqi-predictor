import streamlit as st
import hopsworks
import pandas as pd
import joblib
from pathlib import Path


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Lahore AQI Predictor",
    page_icon="🌫️",
    layout="centered"
)

st.title("🌫️ Lahore AQI Predictor")
st.write("Machine Learning based AQI forecasting for Lahore")


# -----------------------------
# Load Hopsworks + data
# -----------------------------
@st.cache_resource
def load_models():

    project = hopsworks.login()

    fs = project.get_feature_store()
    mr = project.get_model_registry()

    fg = fs.get_feature_group(
        name="aqi_features_v3",
        version=1
    )

    df = fg.select_all().read()

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Create lag features
    df["aqi_lag_1"] = df["us_aqi"].shift(1)
    df["aqi_lag_3"] = df["us_aqi"].shift(3)
    df["aqi_lag_6"] = df["us_aqi"].shift(6)
    df["aqi_lag_12"] = df["us_aqi"].shift(12)
    df["aqi_lag_24"] = df["us_aqi"].shift(24)

    df = df.dropna().reset_index(drop=True)

    latest = df.iloc[-1]

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

    # Load registered models
    models = {}

    for hours in [24, 48, 72]:

        registered_model = mr.get_model(
            name=f"aqi_predictor_{hours}h",
            version=1
        )

        model_dir = registered_model.download()

        model_files = list(
            Path(model_dir).glob("*.pkl")
        )

        models[hours] = joblib.load(model_files[0])

    return latest, X, models


# -----------------------------
# Load
# -----------------------------
with st.spinner("Loading AQI prediction system..."):

    latest, X, models = load_models()


# -----------------------------
# Current AQI
# -----------------------------
st.subheader("Current AQI")

st.metric(
    label="Lahore",
    value=int(latest["us_aqi"])
)


# -----------------------------
# Predictions
# -----------------------------
prediction_24 = models[24].predict(X)[0]
prediction_48 = models[48].predict(X)[0]
prediction_72 = models[72].predict(X)[0]


st.subheader("3-Day AQI Forecast")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "24 Hours",
        f"{prediction_24:.0f}"
    )

with col2:
    st.metric(
        "48 Hours",
        f"{prediction_48:.0f}"
    )

with col3:
    st.metric(
        "72 Hours",
        f"{prediction_72:.0f}"
    )


# -----------------------------
# Explanation
# -----------------------------
st.divider()

st.write(
    "The predictions are generated using historical AQI, "
    "weather conditions, time-based features, and machine learning."
)

st.caption(
    "Model: Random Forest | Location: Lahore, Pakistan"
)
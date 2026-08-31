 Lahore AQI Predictor 

This project is about predicting the Air Quality Index (AQI) of Lahore for the next 3 days using machine learning.

 What I did
 
I collected air quality and weather data and used it to train machine learning models. The data was processed and stored using Hopsworks, and the trained models were also saved there.
I trained separate Random Forest models to predict AQI for:
24 hours
48 hours
72 hours

The final predictions are shown through a Streamlit web app.

Tools Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest
* Joblib
* Hopsworks
* Streamlit
* Open-Meteo API

How the project works

The data is first collected from the API and then processed to create useful features. These features are stored in Hopsworks.
I also created historical data using the backfill pipeline so that there was enough data to train the models.
The training pipeline uses the historical data to train the Random Forest models. The trained models are then stored in the Hopsworks Model Registry.
The Streamlit app loads the latest data and the trained models and displays the current AQI along with the predictions.

Features

The models use previous AQI values along with weather and time information.

Some of the features used are:

* Previous AQI values
* Temperature
* Humidity
* Wind speed
* Precipitation
* Hour
* Day
* Month
* Day of week

Model Results

| Forecast |   MAE |  RMSE |     R² |
| -------- | ----: | ----: | -----: |
| 24 hours | 17.61 | 22.49 |  0.055 |
| 48 hours | 21.87 | 27.70 | -0.430 |
| 72 hours | 25.80 | 32.17 | -1.030 |
The model performs better for the shorter forecast. The 48 and 72 hour predictions are more difficult because AQI can change quite a lot over a longer period.

Streamlit App
The app shows the current AQI in Lahore and gives predictions for the next 24, 48 and 72 hours.

Files
'app.py' - Streamlit app
'feature_pipeline.py' - gets and processes the data
'backfill.py' - creates historical data
'training_pipeline.py' - trains the models
'prediction.py' - prediction related code

Future Improvements
The project can be improved further by adding automatic updates, model retraining, AQI alerts and model explainability.

Pearls AQI Predictor

Lahore, Pakistan

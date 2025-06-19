import pandas as pd
import numpy as np
import requests
import fastf1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import shap

#Load the 2024 Monaco session data
session_2024 = fastf1.get_session(2024, 8, "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)



#Convert lap and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

#Aggregate sector times 
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()


sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

#Clean air race pace
clean_air_race_pace = {
    "VER": 93.191067, "HAM": 93.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
    "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
    "OCO": 95.682128
}

#Qualifying data for 2025 Monaco GP
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
               "HAM", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime (s)": [
        70.669, 69.954, 70.129, None, 71.362, 71.213, 70.063, 70.942,
        70.382, 72.563, 71.994, 70.924, 71.596
    ]
})
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)
qualifying_2025["WetPerformanceFactor"] = 1.1

# Weather API with enhanced error handling
API_KEY = "Your API Key"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=43.7384&lon=7.4246&appid={API_KEY}&units=metric"
try:
    response = requests.get(weather_url)
    response.raise_for_status()
    weather_data = response.json()

    if "list" in weather_data:
        forecast_time = "2025-05-26 12:00:00"
        forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)
        if forecast_data:
            rain_probability = forecast_data.get("pop", 0)
            temperature = forecast_data.get("main", {}).get("temp", 20)
        else:
            print(f"Warning: No forecast data found for {forecast_time}. Using default values.")
            rain_probability = 0
            temperature = 20
    else:
        print(f"Error: 'list' key not found in API response. Response: {weather_data}")
        rain_probability = 0
        temperature = 20
except requests.exceptions.RequestException as e:
    print(f"Error fetching weather data: {e}")
    rain_probability = 0
    temperature = 20

#Adjust qualifying time based on weather
if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

#Add constructor's data
team_points = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Williams": 51, "Ferrari": 114,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 10, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

#Average position change at Monaco
average_position_change_monaco = {
    "VER": -1.0, "NOR": 1.0, "PIA": 0.2, "RUS": 0.5, "SAI": -0.3, "ALB": 0.8,
    "LEC": -1.5, "OCO": -0.2, "HAM": 0.3, "STR": 1.1, "GAS": -0.4, "ALO": -0.6, "HUL": 0.0
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_monaco)


#Merge qualifying and sector times data
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
merged_data["QualifyingTime"] = merged_data["QualifyingTime"]

valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers]

#Define features (X) and target (y)
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore",
    "CleanAirRacePace (s)", "AveragePositionChange"
]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])


imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)


X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)

#Train gradient boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=3, random_state=37)
model.fit(X_train, y_train)
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

#SHAP Analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_imputed)

# Plot for Sumaary
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_imputed_df, feature_names=X.columns, show=True)
plt.title("SHAP Summary Plot for Race Time Prediction")
plt.tight_layout()
plt.show()

#Plot for QualifyingTime
plt.figure(figsize=(10, 6))
shap.dependence_plot("QualifyingTime", shap_values, X_imputed_df, feature_names=X.columns, show=True)
plt.title("SHAP Dependence Plot for QualifyingTime")
plt.tight_layout()
plt.show()

#Force Plots for Top 3 Drivers
final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
top_3_drivers = final_results["Driver"].iloc[:3].values
for i, driver in enumerate(top_3_drivers):
    driver_idx = final_results[final_results["Driver"] == driver].index[0]
    shap.force_plot(explainer.expected_value, shap_values[driver_idx], X_imputed_df.iloc[driver_idx],
                    feature_names=X.columns, matplotlib=True, show=True)
    plt.title(f"SHAP Force Plot for {driver}")
    plt.tight_layout()
    plt.show()

#Sort the results to find the predicted winner
print("\n🏁 Predicted 2025 Monaco GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

#Plot effect of clean air race pace
plt.figure(figsize=(12, 8))
plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedRaceTime (s)"])
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["CleanAirRacePace (s)"].iloc[i], final_results["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), textcoords='offset points')
plt.xlabel("Clean Air Race Pace (s)")
plt.ylabel("Predicted Race Time (s)")
plt.title("Effect of Clean Air Race Pace on Predicted Race Results")
plt.tight_layout()
plt.show()


feature_importance = model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()

#Sort results and get top 3
podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)"]]
print("\n🏆 Predicted in the Top 3 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")
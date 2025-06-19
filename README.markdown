# Monaco Grand Prix 2025 Race Time Prediction

This project predicts the race times for drivers in the 2025 Monaco Grand Prix using machine learning techniques, specifically a Gradient Boosting Regressor. It leverages historical 2024 Monaco GP data, 2025 qualifying times, weather forecasts, and team performance metrics to make informed predictions. The project also incorporates SHAP (SHapley Additive exPlanations) analysis to interpret the model’s predictions and visualize feature importance. Various plots are generated to provide insights into the predictions and the factors influencing them.

## Project Overview

The goal of this project is to predict the average lap times for drivers in the 2025 Monaco Grand Prix and determine the likely race winner and top three finishers. The model accounts for several factors, including:

- **Qualifying Times**: 2025 Monaco GP qualifying times for each driver.
- **Historical Sector Times**: Mean sector times from the 2024 Monaco GP race.
- **Clean Air Race Pace**: Estimated race pace in optimal conditions for select drivers.
- **Weather Conditions**: Forecasted rain probability and temperature for the race day (May 26, 2025).
- **Team Performance**: Normalized constructor championship points as a proxy for team competitiveness.
- **Average Position Change**: Historical data on typical position changes at Monaco for each driver.

The project uses a Gradient Boosting Regressor to model the relationship between these features and the target variable (average lap time). SHAP analysis is employed to explain the model’s predictions, highlighting which features most influence each driver’s predicted race time.

## Features of the Project

1. **Data Sources**:
   - **FastF1 Library**: Provides 2024 Monaco GP race data, including lap and sector times.
   - **OpenWeatherMap API**: Fetches weather forecasts for Monaco on the race date.
   - **Manually Provided Data**:
     - 2025 qualifying times.
     - Clean air race pace for select drivers.
     - Team constructor points.
     - Average position changes at Monaco.

2. **Data Preprocessing**:
   - Converts lap and sector times to seconds for consistency.
   - Handles missing values using median imputation.
   - Merges 2024 sector times with 2025 qualifying data.
   - Adjusts qualifying times based on weather conditions (10% slower in wet conditions if rain probability ≥ 75%).

3. **Machine Learning**:
   - **Model**: Gradient Boosting Regressor with 100 estimators, a learning rate of 0.7, and a maximum depth of 3.
   - **Training**: Uses a 70/30 train-test split with a random state of 37 for reproducibility.
   - **Evaluation**: Reports Mean Absolute Error (MAE) on the test set.

4. **SHAP Analysis**:
   - Provides interpretability by calculating SHAP values for each feature.
   - Generates visualizations to explain feature contributions to predictions.

5. **Output**:
   - Predicted race times for each driver.
   - Predicted winner and top three finishers (podium).
   - Various visualizations to analyze the results and feature importance.

## Outputs and Visualizations

The project generates several plots to provide insights into the predictions and the model’s behavior. Each plot serves a specific purpose:

1. **SHAP Summary Plot**:
   - **Description**: A bar plot showing the average impact of each feature on the model’s predictions across all drivers.
   - **Purpose**: Identifies which features (e.g., QualifyingTime, CleanAirRacePace) have the greatest influence on predicted race times.
   - **Key Insight**: Features with longer bars are more important in determining race time predictions.

2. **SHAP Dependence Plot for QualifyingTime**:
   - **Description**: A scatter plot showing how QualifyingTime affects the model’s predictions, with colors indicating the interaction with another feature (automatically selected by SHAP).
   - **Purpose**: Visualizes the relationship between QualifyingTime and predicted race time, highlighting non-linear effects or interactions.
   - **Key Insight**: Helps understand how faster or slower qualifying times impact race time predictions.

3. **SHAP Force Plots for Top 3 Drivers**:
   - **Description**: Individual force plots for the top three predicted drivers, showing how each feature contributes to their predicted race time.
   - **Purpose**: Provides a detailed explanation of why each driver is predicted to finish in the top three.
   - **Key Insight**: Positive (red) and negative (blue) contributions indicate whether a feature increases or decreases the predicted race time.

4. **Clean Air Race Pace vs. Predicted Race Time Scatter Plot**:
   - **Description**: A scatter plot with CleanAirRacePace on the x-axis and PredictedRaceTime on the y-axis, annotated with driver names.
   - **Purpose**: Examines the relationship between a driver’s clean air race pace and their predicted race time.
   - **Key Insight**: Drivers with faster clean air race paces tend to have faster predicted race times, but other factors (e.g., weather, team performance) also play a role.

5. **Feature Importance Bar Plot**:
   - **Description**: A horizontal bar plot showing the relative importance of each feature in the Gradient Boosting model.
   - **Purpose**: Quantifies the contribution of each feature to the model’s predictions, based on the model’s internal feature importance scores.
   - **Key Insight**: Complements the SHAP summary plot by providing a model-specific perspective on feature importance.

6. **Console Output**:
   - **Predicted Race Times**: A table listing each driver and their predicted average lap time.
   - **Model Error**: The Mean Absolute Error (MAE) of the model on the test set, indicating prediction accuracy.
   - **Predicted Podium**: The top three drivers (P1, P2, P3) based on the lowest predicted race times.

## Special Peculiarities

1. **Weather Integration**:
   - The project dynamically adjusts qualifying times based on weather forecasts. If the rain probability is 75% or higher, qualifying times are scaled by a WetPerformanceFactor (1.1, i.e., 10% slower).
   - Robust error handling ensures default values (rain probability = 0, temperature = 20°C) are used if the API request fails.

2. **Monaco-Specific Considerations**:
   - The model accounts for Monaco’s unique characteristics, such as limited overtaking opportunities, by incorporating historical average position changes for each driver.
   - Sector times from 2024 are used to capture driver performance on Monaco’s tight, technical layout.

3. **SHAP for Interpretability**:
   - Unlike traditional black-box models, this project uses SHAP to provide transparent explanations of predictions, making it easier to understand why certain drivers are favored.
   - Force plots for the top three drivers offer granular insights into individual predictions.

4. **Team Performance Score**:
   - Constructor points are normalized to create a TeamPerformanceScore, reflecting the relative strength of each team. This feature captures the impact of car performance on race outcomes.

5. **Handling Missing Data**:
   - Median imputation is used to handle missing values in features like QualifyingTime or CleanAirRacePace, ensuring the model can make predictions for all drivers.
   - The dataset is filtered to include only drivers present in both 2024 and 2025 data, avoiding extrapolation for new drivers.

## Dependencies

The project requires the following Python libraries:

- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical computations.
- `requests`: Fetching weather data from the OpenWeatherMap API.
- `fastf1`: Accessing Formula 1 race data.
- `scikit-learn`: Machine learning (GradientBoostingRegressor, train-test split, mean_absolute_error, SimpleImputer).
- `matplotlib`: Plotting visualizations.
- `shap`: SHAP analysis for model interpretability.

Install dependencies using:
```bash
pip install pandas numpy requests fastf1 scikit-learn matplotlib shap
```

Additionally, an OpenWeatherMap API key is required for weather data. Replace `API_KEY` in the code with your own key.

## How to Run

1. **Set Up Environment**:
   - Ensure Python 3.7+ is installed.
   - Install dependencies (see above).
   - Obtain an OpenWeatherMap API key and update the `API_KEY` variable.

2. **Prepare Data**:
   - The code includes hardcoded 2025 qualifying times, clean air race pace, team points, and average position changes. Modify these as needed for updated data.
   - The FastF1 library automatically fetches 2024 Monaco GP data.

3. **Execute the Code**:
   - Run the script in a Python environment (e.g., Jupyter Notebook or a `.py` file).
   - Ensure internet access for the weather API and FastF1 data.

4. **View Outputs**:
   - Check the console for predicted race times, model error, and the podium.
   - Review the generated plots for insights into feature importance and driver predictions.

## Limitations and Future Improvements

1. **Limitations**:
   - The model relies on 2024 data, which may not fully reflect 2025 car performance or track changes.
   - Clean air race pace data is available for only a subset of drivers, limiting its impact.
   - The weather forecast is based on a single point in time (May 26, 2025, 12:00), which may not capture race-day variability.
   - The WetPerformanceFactor (1.1) is a simplistic assumption and may not accurately reflect wet-weather performance.

2. **Future Improvements**:
   - Incorporate additional historical data (e.g., 2023, 2022) to improve model robustness.
   - Use more granular weather data (e.g., hourly forecasts) to better model race conditions.
   - Add driver-specific wet-weather performance metrics based on historical wet races.
   - Experiment with other models (e.g., Random Forest, XGBoost) or hyperparameter tuning to improve accuracy.
   - Include tire strategy and pit stop data to model race dynamics more comprehensively.

## Example Output

Below is an example of the console output:

```
🏁 Predicted 2025 Monaco GP Winner 🏁

   Driver  PredictedRaceTime (s)
0     NOR                70.234
1     PIA                70.456
2     VER                70.678
...

Model Error (MAE): 0.45 seconds

🏆 Predicted in the Top 3 🏆
🥇 P1: NOR
🥈 P2: PIA
🥉 P3: VER
```

## Conclusion

This project demonstrates a data-driven approach to predicting Formula 1 race outcomes, combining historical data, real-time weather forecasts, and machine learning. The use of SHAP analysis enhances interpretability, making it a valuable tool for understanding the factors driving race predictions. While tailored to the 2025 Monaco GP, the methodology can be adapted to other races or seasons with appropriate data.

For questions or contributions, please contact the project maintainer.
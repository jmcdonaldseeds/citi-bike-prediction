# Citi Bike Trip Duration Prediction (Improved Version)
# Author: Justin McDonald

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from geopy.distance import geodesic

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Load and Sample Data ---
df = pd.read_csv(
    "/Users/justin/Downloads/202408-citibike-tripdata/202408-citibike-tripdata_3.csv",
    low_memory=False
)
df_sample = df.sample(n=50000, random_state=42).copy()

# --- Feature Engineering ---
df_sample['started_at'] = pd.to_datetime(df_sample['started_at'])
df_sample['ended_at'] = pd.to_datetime(df_sample['ended_at'])
df_sample['trip_duration'] = (df_sample['ended_at'] - df_sample['started_at']).dt.total_seconds() / 60
df_sample['hour'] = df_sample['started_at'].dt.hour
df_sample['day_of_week'] = df_sample['started_at'].dt.dayofweek
df_sample['is_weekend'] = df_sample['day_of_week'] >= 5
df_sample['is_rush_hour'] = df_sample['hour'].isin([7, 8, 9, 16, 17, 18])

# Filter duration range
df_model = df_sample[(df_sample['trip_duration'] >= 1) & (df_sample['trip_duration'] <= 60)].copy()
df_model['log_trip_duration'] = np.log1p(df_model['trip_duration'])

# --- Modeling ---
# Features and target
X = df_model[['hour', 'is_weekend', 'rideable_type', 'day_of_week', 'is_rush_hour']]  # + 'dist_to_midtown' if using
y = df_model['log_trip_duration']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['hour']  # + 'dist_to_midtown' if enabled
categorical_features = ['is_weekend', 'rideable_type', 'day_of_week', 'is_rush_hour']

# Pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

pipeline.fit(X_train, y_train)

# Predictions
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# Evaluation
model_rmse = mean_squared_error(y_true, y_pred) ** 0.5
baseline_rmse = mean_squared_error(y_true, [y_train.mean()] * len(y_true)) ** 0.5
improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100

# --- Residual Analysis ---
residuals = y_true - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Trip Duration (min)")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Gradient Boosting)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=50, color='darkblue', edgecolor='black')
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.tight_layout()
plt.show()

# --- Trip Start Density Heatmap (Hexbin) ---
df_coords = df_sample.dropna(subset=['start_lat', 'start_lng'])

plt.figure(figsize=(8, 8))
plt.hexbin(
    df_coords['start_lng'],
    df_coords['start_lat'],
    gridsize=50,
    cmap='viridis',
    mincnt=1
)
plt.colorbar(label='Trip Count')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trip Start Density (Hexbin)')
plt.tight_layout()
plt.show()

# --- Summary Output ---
print(f"Baseline RMSE: {baseline_rmse:.2f}")
print(f"Model RMSE (Gradient Boosting): {model_rmse:.2f}")
print(f"Improvement over baseline: {improvement:.2f}%")

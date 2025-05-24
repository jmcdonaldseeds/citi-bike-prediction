# Citi Bike Trip Duration Prediction

This project analyzes and predicts trip durations from NYC's Citi Bike system using both R and Python. It combines exploratory data analysis, feature engineering, and gradient boosting to model ride behavior across time and location. A supervised machine learning model was trained on 50,000+ rides using engineered features, reducing prediction error by 28% compared to a naive baseline. Additional analysis includes feature scaling, residual diagnostics, and trip density heatmaps to identify demand hotspots and system inefficiencies.

---

## Project Objectives

- Analyze how trip duration varies by time of day, day of week, user type, and bike type
- Build and evaluate regression models to predict trip duration
- Visualize usage patterns to identify rider trends
- Compare model performance across R and Python workflows

---

## Project Structure

```bash
├── CitiBikeAnalysis.Rmd           # R Markdown: EDA + regression models
├── CitiBikeAnalysis.html          # HTML report (knitted from R Markdown)
├── citibike-prediction.py         # Python script for ML prediction
├── data/                          # (optional) Raw or cleaned CSV data
└── README.md                      # You're here
```

---

## R Analysis

**File**: [`CitiBikeAnalysis.Rmd`](CitiBikeAnalysis.Rmd)  
**HTML Report**: [`CitiBikeAnalysis.html`](CitiBikeAnalysis.html)

### Features:
- Loads and samples 50,000 rows from a Citi Bike trip dataset
- Engineers time-based variables: trip duration, hour, weekday/weekend
- Visualizes distribution of trip durations and user differences
- Builds:
  - Simple regression: `trip_duration ~ hour`
  - Multiple regression: `trip_duration ~ hour + is_weekend + rideable_type`
- Interprets coefficient outputs and statistical significance

---

## Python Modeling

**File**: [`citibike-prediction.py`](citibike-prediction.py)

### Features:
- Cleans and transforms data using `pandas`
- Splits data into training and testing sets
- Builds and evaluates models using:
  - Linear regression
  - Decision tree regressor
- Compares results to a naive baseline (mean-only predictor)

---

## Key Insights

- Casual riders take longer trips than members
- Trips on weekends are ~1.6 minutes longer on average
- Hour of day has a small, positive effect on trip duration
- R² values are low, suggesting more predictors (e.g., location, weather) are needed for stronger models

---

## Future Improvements

- Add geospatial features (station location, trip distance)
- Include weather data (rain, temperature, wind)
- Create heatmaps of demand by time and station
- Try advanced models (e.g., Random Forests, XGBoost)

---

## Technologies Used

| R Language             | Python Stack             |
|------------------------|--------------------------|
| `ggplot2`, `dplyr`     | `pandas`, `scikit-learn` |
| `lubridate`, `infer`   | `matplotlib`, `seaborn`  |
| `data.table`           | `numpy`                  |

---

## Data Source

Citi Bike System Data  
[https://s3.amazonaws.com/tripdata/index.html](https://s3.amazonaws.com/tripdata/index.html)

---

## Author

**Justin McDonald**  
Rutgers University – Information Technology Major  
justmcdonald03@gmail.com

---

## License

This project is for educational and personal portfolio use. Feel free to fork, cite, or build on it with credit.

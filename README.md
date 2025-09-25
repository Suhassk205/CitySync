# CitySync 🚕

This project aims to forecast the hourly demand for taxis in New York City. By leveraging raw, public trip data and external weather information, we build a machine learning pipeline to predict demand patterns, enabling better fleet management and resource allocation for taxi operators.

---

## 📜 Problem Statement

How can we accurately predict the hourly taxi demand in various zones across NYC using historical trip data and relevant external factors like weather?

---

## 💾 Dataset

The primary dataset used is the **NYC TLC Trip Record Data** for Yellow Taxis. This raw data is sourced directly from the official NYC government data portals.

* **Data Source Homepage:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
* **Direct Data Download Portal:** [NYC Open Data](https://data.cityofnewyork.us/browse?Dataset-Information_Agency=Taxi+and+Limousine+Commission+(TLC))

Additionally, historical hourly **weather data** for NYC was integrated from a public source to enrich the feature set with variables like temperature and precipitation.

---

## ⚙️ Project Workflow

This project follows a structured, end-to-end data science pipeline to ensure reproducibility and clarity. Each step builds upon the last, from raw data to a final, evaluated model.

#### 1. Data Acquisition
The initial step involves downloading the raw **NYC Yellow Taxi trip data**. The data is loaded from its efficient **Parquet** format into a pandas DataFrame to begin the analysis.

```python
import pandas as pd
# Load the dataset from a local PARQUET file
df = pd.read_parquet('data/yellow_tripdata_2023-05.parquet')
```

### 2. Data Cleaning & Preprocessing
Raw data is often messy and requires significant cleaning to be useful. Our process focuses on ensuring data quality and structuring the data correctly for a time-series forecasting problem. The most crucial step is aggregating individual trip records into an hourly count per location to create our demand target variable.

```python
# Convert timestamp column to datetime objects
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
# Filter out illogical trips (e.g., trip duration < 1 min or > 2 hours)
# Note: duration_minutes must be calculated first
df = df[(df['duration_minutes'] >= 1) & (df['duration_minutes'] <= 120)]
# Aggregate trips into hourly demand per location
demand_df = df.groupby('PULocationID').resample('H', on='tpep_pickup_datetime').size().reset_index(name='demand')
```

### 3. Exploratory Data Analysis (EDA)
With a clean, aggregated dataset, we perform EDA to uncover underlying patterns. This involves creating visualizations to understand daily and weekly cycles, such as morning/evening rush hours and weekend demand patterns.

```python
import matplotlib.pyplot as plt
# Calculate and plot the average demand for each hour of the day
avg_hourly_demand = demand_df.groupby(demand_df['tpep_pickup_datetime'].dt.hour)['demand'].mean()
avg_hourly_demand.plot(kind='bar', figsize=(12, 6), title='Average Taxi Demand by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Demand')
plt.show()
```

### 4. Feature Engineering
This is the creative process of creating meaningful "clues" (features) for our models. We extract time-based features from the timestamp and enrich the dataset by merging it with external weather data.

```python
# Create time-based features from the timestamp
features_df['hour'] = features_df['tpep_pickup_datetime'].dt.hour
features_df['day_of_week'] = features_df['tpep_pickup_datetime'].dt.dayofweek
features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
```

### 5. Modeling
We employ a two-model strategy and, crucially, use a time-based split for our data. The model is trained on the past and evaluated on the future to simulate a real-world forecasting scenario.

```python
from xgboost import XGBRegressor
# Time-based split (e.g., train on first 3 weeks, test on the last week)
split_date = '2023-05-24'
X_train = features_df[features_df.index < split_date][features]
y_train = features_df[features_df.index < split_date]['demand']
X_test = features_df[features_df.index >= split_date][features]
y_test = features_df[features_df.index >= split_date]['demand']
# Train the XGBoost model
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              early_stopping_rounds=50, 
              verbose=False)
```

6. Evaluation
Finally, we quantitatively assess the performance of our models on the unseen test data. We use RMSE because it gives a higher weight to larger, more significant prediction errors.

```python
from sklearn.metrics import mean_squared_error
import numpy as np
# Make predictions on the test set
predictions = xgb_model.predict(X_test)
# Calculate the Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Final XGBoost Model RMSE: {rmse}")
```

---

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy** (for data manipulation and numerical operations)
* **Scikit-learn** (for Linear Regression and evaluation metrics)
* **XGBoost** (for the main prediction model)
* **Matplotlib & Seaborn** (for data visualization)
* **Jupyter Notebook** (for development and analysis)

---

## 🚀 How to Run

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Launch Jupyter Notebook to explore the code:**
    ```bash
    jupyter notebook
    ```

---

## 📊 Results & Findings

*(This section should be updated with your final results)*

Our XGBoost model achieved a significant performance improvement over the baseline Linear Regression model.

* **Baseline (Linear Regression) RMSE:** `[Enter Value Here]`
* **Main Model (XGBoost) RMSE:** `[Enter Value Here]`

Key insights from the feature importance analysis show that the **hour of the day**, **day of the week**, and **precipitation levels** are the most significant predictors of taxi demand.

---

## 👥 Team Members

This project was developed by **Team CitySync**:

* Suhas Kembavi (Lead)
* Ridam Goyal
* Shreesha b s
* Archit Kulkarni
* Amrutha Tavva

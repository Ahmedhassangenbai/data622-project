import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================
# Load Data
# =========================

df = pd.read_csv(
    "final_merged_energy_weather_population.csv"
)

df["Month"] = pd.to_datetime(
    df["Month"]
)


# =========================
# Keep Valid Boroughs
# =========================

valid_boroughs = [
    "BRONX",
    "BROOKLYN",
    "MANHATTAN",
    "QUEENS",
    "STATEN ISLAND"
]

df = df[
    df["Borough"].isin(valid_boroughs)
].copy()


# =========================
# Sort Data
# =========================

df = df.sort_values(
    ["Borough", "Month"]
).reset_index(drop=True)


# =========================
# Feature Engineering
# =========================

df["Month_Num"] = (
    df["Month"]
    .dt.month
)

df["Year"] = (
    df["Month"]
    .dt.year
)


# =========================
# Cyclical Seasonality
# =========================

df["Sin_Month"] = np.sin(
    2 * np.pi * df["Month_Num"] / 12
)

df["Cos_Month"] = np.cos(
    2 * np.pi * df["Month_Num"] / 12
)


# =========================
# Lag Features
# =========================

df["KWH_Lag1"] = (
    df.groupby("Borough")["Consumption (KWH)"]
    .shift(1)
)

df["Cost_Lag1"] = (
    df.groupby("Borough")["Current Charges"]
    .shift(1)
)

df["KWH_Lag12"] = (
    df.groupby("Borough")["Consumption (KWH)"]
    .shift(12)
)

df["Cost_Lag12"] = (
    df.groupby("Borough")["Current Charges"]
    .shift(12)
)


# =========================
# Encode Borough
# =========================

le = LabelEncoder()

df["Borough_Encoded"] = (
    le.fit_transform(df["Borough"])
)


# =========================
# Replace Infinite Values
# =========================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# =========================
# Fill Missing Population Features
# =========================

df["Electricity Cost Per Person"] = (
    df["Electricity Cost Per Person"]
    .fillna(
        df["Electricity Cost Per Person"].median()
    )
)

df["KWH Per Person"] = (
    df["KWH Per Person"]
    .fillna(
        df["KWH Per Person"].median()
    )
)


# =========================
# Drop Missing Lag Rows
# =========================

df = df.dropna(subset=[
    "Avg_Temp",
    "KWH_Lag1",
    "Cost_Lag1",
    "KWH_Lag12",
    "Cost_Lag12"
])


# =========================
# Define Features & Target
# =========================

features = [
    "Avg_Temp",
    "Month_Num",
    "Year",
    "Sin_Month",
    "Cos_Month",
    "KWH_Lag1",
    "Cost_Lag1",
    "KWH_Lag12",
    "Cost_Lag12",
    "Electricity Cost Per Person",
    "KWH Per Person",
    "Total Population",
    "Borough_Encoded"
]

target = "Current Charges"


# =========================
# Log Transform Target
# =========================

X = df[features]

y = np.log1p(
    df[target]
)


# =========================
# Time-Based Train/Test Split
# =========================

train_mask = (
    df["Year"] <= 2022
)

test_mask = (
    df["Year"] >= 2023
)

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print(
    f"Train size: {X_train.shape[0]} rows"
)

print(
    f"Test size: {X_test.shape[0]} rows"
)

print(
    "\nTest borough counts:"
)

print(
    df[test_mask]["Borough"].value_counts()
)


# =========================
# Ridge Regression
# =========================

ridge = Ridge(alpha=1.0)

ridge.fit(
    X_train,
    y_train
)

y_pred_log = ridge.predict(
    X_test
)


# =========================
# Convert Back from Log Scale
# =========================

y_pred = np.expm1(
    y_pred_log
)

y_test_actual = np.expm1(
    y_test
)


# =========================
# Evaluation Metrics
# =========================

rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        y_pred
    )
)

mae = mean_absolute_error(
    y_test_actual,
    y_pred
)

print("\n--- Ridge Regression ---")

print(
    f"RMSE: {rmse:,.2f}"
)

print(
    f"MAE:  {mae:,.2f}"
)


# =========================
# Feature Importance
# =========================

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": ridge.coef_
})

coef_df = coef_df.sort_values(
    "Coefficient",
    ascending=False
)

print("\n--- Coefficients ---")

print(
    coef_df.to_string(index=False)
)


# =========================
# Save Predictions
# =========================

results = df[test_mask][[
    "Borough",
    "Month",
    target
]].copy()

results["Predicted_Current_Charges"] = (
    y_pred
)

results["Residual"] = (
    results[target] -
    results["Predicted_Current_Charges"]
)

results.to_csv(
    "model_predictions.csv",
    index=False
)


# =========================
# Summary Statistics
# =========================

print("\n--- Target Summary ---")

print(
    df["Current Charges"].describe()
)

print(
    "\nPredictions saved to model_predictions.csv"
)
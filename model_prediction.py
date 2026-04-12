import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# Load Data
# =========================

df = pd.read_csv("final_merged_energy_weather.csv")

df["Month"] = pd.to_datetime(df["Month"])

# Group back to borough level for modeling
valid_boroughs = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
df = df[df["Borough"].isin(valid_boroughs)].copy()

numeric_cols = [
    "Consumption (KWH)",
    "Consumption (KW)",
    "KWH Charges",
    "KW Charges",
    "Current Charges"
]

df = df.groupby(["Borough", "Month"])[numeric_cols + ["Avg_Temp"]].mean().reset_index()
df = df.sort_values(["Borough", "Month"]).reset_index(drop=True)


# =========================
# Feature Engineering
# =========================

df["Month_Num"] = df["Month"].dt.month
df["Year"]      = df["Month"].dt.year

# Cyclical seasonality encoding
df["Sin_Month"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
df["Cos_Month"] = np.cos(2 * np.pi * df["Month_Num"] / 12)

# Lag features (by borough) — use these instead of same-month consumption
df["KWH_Lag1"]  = df.groupby("Borough")["Consumption (KWH)"].shift(1)
df["KW_Lag1"]   = df.groupby("Borough")["Consumption (KW)"].shift(1)
df["Cost_Lag1"] = df.groupby("Borough")["Current Charges"].shift(1)

# Encode borough
le = LabelEncoder()
df["Borough_Encoded"] = le.fit_transform(df["Borough"])

# Drop rows where lag features are NaN (first month per borough)
df = df.dropna(subset=["KWH_Lag1", "KW_Lag1", "Cost_Lag1"])


# =========================
# Define Features & Target
# =========================
# No same-month consumption to avoid data leakage

features = [
    "Avg_Temp",
    "Month_Num",
    "Year",
    "Sin_Month",
    "Cos_Month",
    "KWH_Lag1",
    "KW_Lag1",
    "Cost_Lag1",
    "Borough_Encoded"
]

target = "Current Charges"

X = df[features]
y = df[target]


# =========================
# Time-Based Train/Test Split
# =========================

train_mask = df["Year"] <= 2022
test_mask  = df["Year"] >= 2023

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train size: {X_train.shape[0]} rows")
print(f"Test size:  {X_test.shape[0]} rows")


# =========================
# Linear Regression (Baseline)
# =========================

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr  = mean_absolute_error(y_test, y_pred_lr)

print("\n--- Linear Regression ---")
print(f"RMSE: {rmse_lr:,.2f}")
print(f"MAE:  {mae_lr:,.2f}")


# =========================
# Ridge Regression
# =========================

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge  = mean_absolute_error(y_test, y_pred_ridge)

print("\n--- Ridge Regression ---")
print(f"RMSE: {rmse_ridge:,.2f}")
print(f"MAE:  {mae_ridge:,.2f}")


# =========================
# Coefficients
# =========================

coef_df = pd.DataFrame({
    "Feature": features,
    "LR_Coefficient": lr.coef_,
    "Ridge_Coefficient": ridge.coef_
}).sort_values("LR_Coefficient", ascending=False)

print("\n--- Coefficients ---")
print(coef_df.to_string(index=False))


# =========================
# Save Predictions
# =========================

results = df[test_mask][["Borough", "Month", target]].copy()
results["LR_Predicted"]    = y_pred_lr
results["Ridge_Predicted"] = y_pred_ridge
results["LR_Residual"]     = results[target] - results["LR_Predicted"]
results["Ridge_Residual"]  = results[target] - results["Ridge_Predicted"]

results.to_csv("model_predictions.csv", index=False)
print("\nPredictions saved to model_predictions.csv")
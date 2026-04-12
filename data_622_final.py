import pandas as pd

# =========================
# Load Data
# =========================

energy_df = pd.read_csv("Electric_Consumption_And_Cost_(2010_-_Sep_2025)_20260411.csv")
weather_df = pd.read_csv("weather_data_nyc.csv")
zip_df = pd.read_csv("NYC_ZIP_ZCTA_MODZCTA_Crosswalk.csv")


# =========================
# Energy Data
# =========================

energy_df = energy_df[[
    "Borough",
    "Revenue Month",
    "Consumption (KWH)",
    "Consumption (KW)",
    "KWH Charges",
    "KW Charges",
    "Current Charges"
]]

energy_df["Revenue Month"] = pd.to_datetime(energy_df["Revenue Month"])
energy_df["Borough"] = energy_df["Borough"].str.strip()

numeric_cols = [
    "Consumption (KWH)",
    "Consumption (KW)",
    "KWH Charges",
    "KW Charges",
    "Current Charges"
]

# Remove commas and convert numeric
for col in numeric_cols:
    energy_df[col] = (
        energy_df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    energy_df[col] = pd.to_numeric(energy_df[col], errors="coerce")


# =========================
# Weather Data
# =========================

weather_df.columns = ["Month", "Avg_Temp"]

weather_df["Month"] = pd.to_datetime(
    weather_df["Month"].astype(str),
    format="%Y%m"
)


# =========================
# Aggregate Energy Monthly
# =========================

energy_df["Month"] = energy_df["Revenue Month"].dt.to_period("M")

energy_monthly = energy_df.groupby(
    ["Borough", "Month"]
)[numeric_cols].sum().reset_index()

energy_monthly["Month"] = energy_monthly["Month"].dt.to_timestamp()


# =========================
# Merge
# =========================

final_df = energy_monthly.merge(
    weather_df,
    on="Month",
    how="left"
)


# =========================
# Save Final
# =========================

final_df.to_csv("final_merged_energy_weather.csv", index=False)

print(final_df.head())
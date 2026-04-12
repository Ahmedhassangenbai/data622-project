import pandas as pd

# =========================
# Load Data
# =========================

energy_df = pd.read_csv("Electric_Consumption_And_Cost_(2010_-_Sep_2025)_20260411.csv")
weather_df = pd.read_csv("weather_data_nyc.csv")
zip_df     = pd.read_csv("NYC_ZIP_ZCTA_MODZCTA_Crosswalk.csv")


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
energy_df["Borough"] = energy_df["Borough"].str.strip().str.upper()

numeric_cols = [
    "Consumption (KWH)",
    "Consumption (KW)",
    "KWH Charges",
    "KW Charges",
    "Current Charges"
]

# Remove commas, dollar signs, and whitespace then convert to numeric
for col in numeric_cols:
    energy_df[col] = (
        energy_df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
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
# ZIP Crosswalk
# =========================

# Drop dummy/invalid rows
zip_df = zip_df.dropna(subset=["BOROUGH"])

# Uppercase borough to match energy data
zip_df["BOROUGH"] = zip_df["BOROUGH"].str.strip().str.upper()

# Keep only the 5 real boroughs
valid_boroughs = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
zip_df = zip_df[zip_df["BOROUGH"].isin(valid_boroughs)].copy()

zip_df = zip_df[["ZCTA", "MODZCTA", "UHFCODE", "UHFNAME", "BOROUGH"]]


# =========================
# Aggregate Energy Monthly (Borough Level)
# =========================

energy_df["Month"] = energy_df["Revenue Month"].dt.to_period("M")

energy_monthly = energy_df.groupby(
    ["Borough", "Month"]
)[numeric_cols].sum().reset_index()

energy_monthly["Month"] = energy_monthly["Month"].dt.to_timestamp()

# Keep only valid boroughs
energy_monthly = energy_monthly[energy_monthly["Borough"].isin(valid_boroughs)]


# =========================
# Merge Energy + ZIP Crosswalk
# =========================
# Each borough row expands to all ZIP codes within that borough

final_df = energy_monthly.merge(
    zip_df,
    left_on="Borough",
    right_on="BOROUGH",
    how="left"
).drop(columns=["BOROUGH"])  # drop duplicate borough col from zip_df


# =========================
# Merge with Weather
# =========================

final_df = final_df.merge(
    weather_df,
    on="Month",
    how="left"
)


# =========================
# Reorder Columns
# =========================

final_df = final_df[[
    "ZCTA",
    "UHFCODE",
    "UHFNAME",
    "Borough",
    "Month",
    "Consumption (KWH)",
    "Consumption (KW)",
    "KWH Charges",
    "KW Charges",
    "Current Charges",
    "Avg_Temp"
]]


# =========================
# Save Final
# =========================

final_df.to_csv("final_merged_energy_weather.csv", index=False)

print(f"Final shape: {final_df.shape}")
print(final_df.head(10).to_string())



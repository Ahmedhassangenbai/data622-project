import pandas as pd

# =========================
# Load Data
# =========================

energy_df  = pd.read_csv("Electric_Consumption_And_Cost_(2010_-_Sep_2025)_20260411.csv")
weather_df = pd.read_csv("weather_data_nyc.csv")
zip_map_df = pd.read_csv("NYCHA_Development_Zip_Mapping_CLEAN.csv")


# =========================
# Energy Data
# =========================

energy_df = energy_df[[
    "Borough",
    "Account Name",
    "Revenue Month",
    "Consumption (KWH)",
    "Consumption (KW)",
    "KWH Charges",
    "KW Charges",
    "Current Charges"
]]

energy_df["Revenue Month"] = pd.to_datetime(energy_df["Revenue Month"])
energy_df["Borough"] = energy_df["Borough"].str.strip().str.upper()
energy_df["Account Name"] = energy_df["Account Name"].str.strip().str.upper()

numeric_cols = [
    "Consumption (KWH)",
    "Consumption (KW)",
    "KWH Charges",
    "KW Charges",
    "Current Charges"
]

# Remove commas, dollar signs, whitespace then convert to numeric
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
# ZIP Mapping
# =========================

zip_map_df.columns = ["Account Name", "Zip Code"]
zip_map_df["Account Name"] = zip_map_df["Account Name"].str.strip().str.upper()
zip_map_df["Zip Code"] = (
    zip_map_df["Zip Code"]
    .astype(str)
    .str.replace(r'\.0$', '', regex=True)
    .str.strip()
)

# Drop rows with missing ZIP
zip_map_df = zip_map_df.dropna(subset=["Zip Code"])
zip_map_df = zip_map_df[zip_map_df["Zip Code"] != '']


# =========================
# Merge Energy + ZIP
# =========================

energy_df = energy_df.merge(
    zip_map_df,
    on="Account Name",
    how="left"
)

# Keep only valid boroughs
valid_boroughs = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
energy_df = energy_df[energy_df["Borough"].isin(valid_boroughs)]

# Create Month column
energy_df["Month"] = energy_df["Revenue Month"].dt.to_period("M").dt.to_timestamp()

# Drop rows where ZIP could not be mapped
energy_df = energy_df.dropna(subset=["Zip Code"])


# =========================
# Aggregate Monthly by ZIP
# If multiple developments share the same ZIP + Month,
# we take the average across those developments
# =========================

final_df = energy_df.groupby(
    ["Borough", "Zip Code", "Month"]
)[numeric_cols].mean().reset_index()

final_df = final_df.sort_values(["Borough", "Zip Code", "Month"]).reset_index(drop=True)


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
    "Zip Code",
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
print(f"Unique ZIP codes: {final_df['Zip Code'].nunique()}")
print(f"Date range: {final_df['Month'].min()} to {final_df['Month'].max()}")
print(final_df.head(10).to_string())
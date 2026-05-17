import pandas as pd

# =========================
# Load Data
# =========================

energy_df  = pd.read_csv("Electric_Consumption_And_Cost_(2010_-_Sep_2025)_20260411.csv")
panel_df   = pd.read_csv("nycha_panel_2016_2024.csv", dtype=str)
weather_df = pd.read_csv("weather_data_nyc.csv")


# =========================
# Energy Data - Base Cleaning
# =========================

energy_df = energy_df[[
    "Borough",
    "Account Name",
    "AMP #",
    "Revenue Month",
    "Consumption (KWH)",
    "Consumption (KW)",
    "KWH Charges",
    "KW Charges",
    "Current Charges"
]]

energy_df["Revenue Month"] = pd.to_datetime(
    energy_df["Revenue Month"]
)

energy_df["Borough"] = (
    energy_df["Borough"]
    .astype(str)
    .str.strip()
    .str.upper()
)

energy_df["Account Name"] = (
    energy_df["Account Name"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Clean AMP
energy_df["AMP_CLEAN"] = (
    energy_df["AMP #"]
    .astype(str)
    .str.replace("P$", "", regex=True)
    .str.strip()
)

numeric_cols = [
    "Consumption (KWH)",
    "Consumption (KW)",
    "KWH Charges",
    "KW Charges",
    "Current Charges"
]

for col in numeric_cols:

    energy_df[col] = (
        energy_df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )

    energy_df[col] = pd.to_numeric(
        energy_df[col],
        errors="coerce"
    )


# =========================
# Filter Years
# =========================

energy_df = energy_df[
    (energy_df["Revenue Month"].dt.year >= 2016) &
    (energy_df["Revenue Month"].dt.year <= 2024)
].copy()

energy_df["YEAR"] = (
    energy_df["Revenue Month"]
    .dt.year
    .astype(str)
)


# =========================
# Prepare NYCHA Panel
# =========================

panel_df["AMP_CLEAN"] = (
    panel_df["HUD AMP #"]
    .astype(str)
    .str.strip()
)

panel_df["YEAR"] = (
    panel_df["YEAR"]
    .astype(str)
)

panel_df["TOTAL POPULATION"] = pd.to_numeric(
    panel_df["TOTAL POPULATION"],
    errors="coerce"
)

panel_lookup = (
    panel_df
    .groupby(
        ["AMP_CLEAN", "YEAR"],
        as_index=False
    )["TOTAL POPULATION"]
    .sum()
)


# =========================
# Merge Energy + Population
# =========================

energy_df = energy_df.merge(
    panel_lookup,
    on=["AMP_CLEAN", "YEAR"],
    how="inner"
)


# =========================
# Weather Data
# =========================

weather_df["DATE"] = pd.to_datetime(
    weather_df["DATE"]
)

weather_df["TMAX"] = pd.to_numeric(
    weather_df["TMAX"],
    errors="coerce"
)

weather_df["TMIN"] = pd.to_numeric(
    weather_df["TMIN"],
    errors="coerce"
)

# Average temperature
weather_df["Avg_Temp"] = (
    weather_df["TMAX"] +
    weather_df["TMIN"]
) / 2

# Monthly timestamp
weather_df["Month"] = (
    weather_df["DATE"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

# Monthly average weather
weather_df = (
    weather_df
    .groupby("Month")["Avg_Temp"]
    .mean()
    .reset_index()
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

energy_df = energy_df[
    energy_df["Borough"].isin(valid_boroughs)
]


# =========================
# Create Month Column
# =========================

energy_df["Month"] = (
    energy_df["Revenue Month"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# =========================
# Aggregate Monthly by Borough
# =========================

final_df = (
    energy_df
    .groupby([
        "Borough",
        "Month"
    ])
    .agg({
        "Consumption (KWH)": "sum",
        "Consumption (KW)": "sum",
        "KWH Charges": "sum",
        "KW Charges": "sum",
        "Current Charges": "sum",
        "TOTAL POPULATION": "sum"
    })
    .reset_index()
)

final_df = final_df.sort_values([
    "Borough",
    "Month"
]).reset_index(drop=True)


# =========================
# Rename Population Column
# =========================

final_df = final_df.rename(
    columns={
        "TOTAL POPULATION": "Total Population"
    }
)


# =========================
# Merge Weather
# =========================

final_df = final_df.merge(
    weather_df,
    on="Month",
    how="left"
)


# =========================
# Create Per Person Metrics
# =========================

final_df["Electricity Cost Per Person"] = (
    final_df["Current Charges"] /
    final_df["Total Population"]
)

final_df["KWH Per Person"] = (
    final_df["Consumption (KWH)"] /
    final_df["Total Population"]
)


# =========================
# Final Columns
# =========================

final_df = final_df[[
    "Borough",
    "Month",
    "Consumption (KWH)",
    "Consumption (KW)",
    "KWH Charges",
    "KW Charges",
    "Current Charges",
    "Total Population",
    "Electricity Cost Per Person",
    "KWH Per Person",
    "Avg_Temp"
]]


# =========================
# Save Final CSV
# =========================

final_df.to_csv(
    "final_merged_energy_weather_population.csv",
    index=False
)


# =========================
# Final Debug
# =========================

print("\n=========================")
print("FINAL OUTPUT")
print("=========================")

print(
    "Final shape:",
    final_df.shape
)

print(
    "Unique boroughs:",
    final_df["Borough"].nunique()
)

print(
    "Date range:",
    final_df["Month"].min(),
    "to",
    final_df["Month"].max()
)

print(
    "\nMissing values:"
)

print(
    final_df.isna().sum()
)

print(
    "\nFirst 10 rows:"
)

print(
    final_df.head(10).to_string()
)
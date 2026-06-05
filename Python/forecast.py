import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load files
train = pd.read_csv(r"D:\Data Analytics\Retail_Demand_Forecasting\Data\train.csv")
test = pd.read_csv(r"D:\Data Analytics\Retail_Demand_Forecasting\Data\test.csv")

# Convert dates
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

# Create date features
for df in [train, test]:

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["quarter"] = df["date"].dt.quarter

# Features
features = [
    "store",
    "item",
    "year",
    "month",
    "day",
    "weekday",
    "quarter"
]

X_train = train[features]
y_train = train["sales"]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
X_test = test[features]
predictions = model.predict(X_test)

# Create forecast dataframe
forecast = test[["date", "store", "item"]].copy()

forecast["forecast_sales"] = predictions

# Save file
forecast.to_csv(
    r"D:\Data Analytics\Retail_Demand_Forecasting\Forecast\forecast_sales.csv",
    index=False
)

print("Forecast file created successfully.")
print(forecast.head())
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
DATA_PATH = os.path.join("..", "data", "Housing.csv")
OUTPUT_DIR = os.path.join("..", "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Import and preprocess the dataset
df = pd.read_csv(DATA_PATH)

print("First 5 rows of data:\n", df.head())
print("\nColumns:", df.columns)

# Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Define X & y (target = 'price')
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# 4. Predictions & evaluation
y_pred = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Model coefficients
print("\nIntercept:", lr.intercept_)
print("Coefficients:")
for col, coef in zip(X.columns, lr.coef_):
    print(f"{col}: {coef}")

# 5. Plot regression line (if 'area' exists)
if "area" in X.columns:
    plt.scatter(X_test["area"], y_test, color="blue", label="Actual")
    plt.plot(X_test["area"], lr.predict(X_test), color="red", linewidth=2, label="Predicted")
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Simple Linear Regression (Area vs Price)")
    plt.legend()

    save_path = os.path.join(OUTPUT_DIR, "regression_line.png")
    plt.savefig(save_path)
    plt.show()
    print(f"\n✅ Plot saved at: {save_path}")

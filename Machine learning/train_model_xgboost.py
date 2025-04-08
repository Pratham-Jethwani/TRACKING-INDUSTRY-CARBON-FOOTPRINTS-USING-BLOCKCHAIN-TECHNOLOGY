import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the directory to save the images if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Load the dataset
df = pd.read_csv('C:\\pratham\\capstone\\steel_carbon_credit_prediction\\dataset.csv')

# Preprocessing the data
X = df.drop("Carbon Credits (tons)", axis=1)
y = df["Carbon Credits (tons)"]

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model using XGBoost
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

# Print the metrics
print("Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")
print(f"Explained Variance Score: {evs}")

# 1. Actual vs Predicted Values Plot (Scatter Plot)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Carbon Credits (tons)')
plt.ylabel('Predicted Carbon Credits (tons)')
plt.tight_layout()
plt.savefig('images/actual_vs_predicted.png')  # Save the plot
plt.close()

# 2. Residual Plot (Error Plot)
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Carbon Credits (tons)')
plt.ylabel('Residuals (Error)')
plt.tight_layout()
plt.savefig('images/residual_plot.png')  # Save the plot
plt.close()

# 3. Feature Importance Plot
xgb.plot_importance(model, importance_type='weight', max_num_features=10, height=0.5)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('images/feature_importance.png')  # Save the plot
plt.close()

# 4. Prediction Distribution vs Actual Distribution (Density Plot)
plt.figure(figsize=(8, 6))
sns.kdeplot(y_test, color='blue', label='Actual Values')
sns.kdeplot(y_pred, color='green', label='Predicted Values')
plt.title('Distribution of Actual vs Predicted Values')
plt.xlabel('Carbon Credits (tons)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('images/prediction_distribution.png')  # Save the plot
plt.close()

# 5. Learning Curve (Training and Test Errors)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
train_mean = -train_scores.mean(axis=1)
test_mean = -test_scores.mean(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Train Error', color='blue')
plt.plot(train_sizes, test_mean, label='Test Error', color='green')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.tight_layout()
plt.savefig('images/learning_curve.png')  # Save the plot
plt.close()

# Save the model and scaler using pickle
with open('model/model_xgb.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("XGBoost model trained and saved successfully.")

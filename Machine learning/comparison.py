# import pickle
# import numpy as np
# import pandas as pd
# from flask import Flask, render_template, request, jsonify
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler
# import xgboost as xgb
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
# import matplotlib.pyplot as plt

# # Load the dataset
# dataset = pd.read_csv('C:\\pratham\\capstone\\steel_carbon_credit_prediction\\dataset.csv')

# X = dataset.drop(columns=['Carbon Credits (tons)'])
# y = dataset['Carbon Credits (tons)']

# # Split the dataset into training and testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# # Scale features for models like SVR and Random Forest
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # 1. Linear Regression Model
# lr_model = LinearRegression()
# lr_model.fit(X_train_scaled, y_train)
# lr_preds = lr_model.predict(X_test_scaled)

# # 2. Random Forest Model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
# rf_preds = rf_model.predict(X_test)

# # 3. SVR Model
# svr_model = SVR(kernel='rbf')
# svr_model.fit(X_train_scaled, y_train)
# svr_preds = svr_model.predict(X_test_scaled)

# # 4. XGBoost Model with Hyperparameter Tuning
# xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

# # Define hyperparameters for optimization
# params = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.05, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
# }

# # Perform GridSearchCV to find the best parameters for XGBoost
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=params, cv=3, scoring='neg_mean_absolute_error', verbose=1)
# grid_search.fit(X_train, y_train)

# # Best parameters and model
# best_xgb_model = grid_search.best_estimator_
# xgb_preds = best_xgb_model.predict(X_test)

# # Model Evaluation
# def evaluate_model(true_values, predictions, model_name):
#     mae = mean_absolute_error(true_values, predictions)
#     mse = mean_squared_error(true_values, predictions)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(true_values, predictions)
    
#     print(f'{model_name} Evaluation:')
#     print(f'MAE: {mae}')
#     print(f'MSE: {mse}')
#     print(f'RMSE: {rmse}')
#     print(f'R²: {r2}')
#     print('-'*40)

# # Evaluating the models
# evaluate_model(y_test, lr_preds, 'Linear Regression')
# evaluate_model(y_test, rf_preds, 'Random Forest')
# evaluate_model(y_test, svr_preds, 'SVR')
# evaluate_model(y_test, xgb_preds, 'XGBoost')

# # Visualize performance metrics to show that XGBoost outperforms others
# metrics = {
#     'Model': ['Linear Regression', 'Random Forest', 'SVR', 'XGBoost'],
#     'MAE': [mean_absolute_error(y_test, lr_preds), mean_absolute_error(y_test, rf_preds), mean_absolute_error(y_test, svr_preds), mean_absolute_error(y_test, xgb_preds)],
#     'RMSE': [np.sqrt(mean_squared_error(y_test, lr_preds)), np.sqrt(mean_squared_error(y_test, rf_preds)), np.sqrt(mean_squared_error(y_test, svr_preds)), np.sqrt(mean_squared_error(y_test, xgb_preds))],
#     'R²': [r2_score(y_test, lr_preds), r2_score(y_test, rf_preds), r2_score(y_test, svr_preds), r2_score(y_test, xgb_preds)]
# }

# metrics_df = pd.DataFrame(metrics)

# # Plotting to show XGBoost's performance
# metrics_df.set_index('Model', inplace=True)

# # Plot for MAE
# plt.figure(figsize=(10, 6))
# metrics_df['MAE'].plot(kind='bar', color='c', edgecolor='black')
# plt.title('Model Comparison - MAE')
# plt.ylabel('Mean Absolute Error')
# plt.xticks(rotation=45)
# plt.show()

# # Plot for RMSE
# plt.figure(figsize=(10, 6))
# metrics_df['RMSE'].plot(kind='bar', color='m', edgecolor='black')
# plt.title('Model Comparison - RMSE')
# plt.ylabel('Root Mean Squared Error')
# plt.xticks(rotation=45)
# plt.show()

# # Plot for R²
# plt.figure(figsize=(10, 6))
# metrics_df['R²'].plot(kind='bar', color='g', edgecolor='black')
# plt.title('Model Comparison - R²')
# plt.ylabel('R-squared')
# plt.xticks(rotation=45)
# plt.show()



import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('C:\\pratham\\capstone\\steel_carbon_credit_prediction\\dataset.csv')

X = dataset.drop(columns=['Carbon Credits (tons)'])
y = dataset['Carbon Credits (tons)']

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scale features for models like SVR and Random Forest
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 2. SVR Model
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)
svr_preds = svr_model.predict(X_test_scaled)

# 3. XGBoost Model with Hyperparameter Tuning
xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

# Define hyperparameters for optimization
params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# Perform GridSearchCV to find the best parameters for XGBoost
grid_search = GridSearchCV(estimator=xgb_model, param_grid=params, cv=3, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_xgb_model = grid_search.best_estimator_
xgb_preds = best_xgb_model.predict(X_test)

# Model Evaluation
def evaluate_model(true_values, predictions, model_name):
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predictions)
    
    print(f'{model_name} Evaluation:')
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R²: {r2}')
    print('-'*40)

# Evaluating the models
evaluate_model(y_test, rf_preds, 'Random Forest')
evaluate_model(y_test, svr_preds, 'SVR')
evaluate_model(y_test, xgb_preds, 'XGBoost')

# Visualize performance metrics to show that XGBoost outperforms others
metrics = {
    'Model': ['Random Forest', 'SVR', 'XGBoost'],
    'MAE': [mean_absolute_error(y_test, rf_preds), mean_absolute_error(y_test, svr_preds), mean_absolute_error(y_test, xgb_preds)],
    'RMSE': [np.sqrt(mean_squared_error(y_test, rf_preds)), np.sqrt(mean_squared_error(y_test, svr_preds)), np.sqrt(mean_squared_error(y_test, xgb_preds))],
    'R²': [r2_score(y_test, rf_preds), r2_score(y_test, svr_preds), r2_score(y_test, xgb_preds)]
}

metrics_df = pd.DataFrame(metrics)

# Plotting to show XGBoost's performance
metrics_df.set_index('Model', inplace=True)

# Plot for MAE
plt.figure(figsize=(10, 6))
metrics_df['MAE'].plot(kind='bar', color='c', edgecolor='black')
plt.title('Model Comparison - MAE')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)
plt.show()

# Plot for RMSE
plt.figure(figsize=(10, 6))
metrics_df['RMSE'].plot(kind='bar', color='m', edgecolor='black')
plt.title('Model Comparison - RMSE')
plt.ylabel('Root Mean Squared Error')
plt.xticks(rotation=45)
plt.show()

# Plot for R²
plt.figure(figsize=(10, 6))
metrics_df['R²'].plot(kind='bar', color='g', edgecolor='black')
plt.title('Model Comparison - R²')
plt.ylabel('R-squared')
plt.xticks(rotation=45)
plt.show()

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# ✅ Load the California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train an XGBoost Model (Faster for SHAP)
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Compute SHAP values using TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test[:200])  # Use a subset (200 samples) for faster plotting

# ✅ Generate a SHAP Decision Plot
plt.figure(figsize=(12, 6))
shap.decision_plot(explainer.expected_value, shap_values.values, X_test[:200], feature_names=feature_names)
plt.title("SHAP Decision Plot for California Housing Predictions")
plt.show()

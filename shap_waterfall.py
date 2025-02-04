import shap
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# ✅ Force SHAP & PyTorch to use CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Load the California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train an XGBoost Model
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test[:200])  # Use a subset for performance

# ✅ Convert X_test to DataFrame
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# ✅ Select a single observation (Instance #5)
instance_index = 5
shap_single = shap_values[instance_index]

# ✅ Generate SHAP Waterfall Plot
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_single)
plt.title(f"SHAP Waterfall Plot for Observation #{instance_index}")
plt.show()

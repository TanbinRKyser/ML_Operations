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
shap_values = explainer(X_test[:200])  # Use a subset of 200 samples for speed

# Convert X_test to a DataFrame with feature names
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Select the 5th observation properly
instance_index = 5  # Choose the 5th prediction
shap_single = shap_values[instance_index:instance_index + 1]

# ✅ Generate SHAP Decision Plot for that single observation
plt.figure(figsize=(10, 5))
shap.decision_plot(explainer.expected_value, shap_single.values, X_test_df.iloc[instance_index:instance_index + 1], feature_names=feature_names)
plt.title(f"SHAP Decision Plot for Observation #{instance_index}")
plt.show()

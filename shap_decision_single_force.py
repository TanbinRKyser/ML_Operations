import shap
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from IPython.core.display import display

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
shap_values = explainer(X_test[:200])  # Use 200 samples for speed

# ✅ Convert X_test to DataFrame
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# ✅ Select a single observation (Instance #5)
instance_index = 5
shap_single = shap_values[instance_index]

# ✅ Generate SHAP Force Plot
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value, shap_single.values, X_test_df.iloc[instance_index])

# ✅ Display in Jupyter Notebook (if available)
# try:
#     display(force_plot)
# except:
print("Force plot could not be displayed in this environment. Saving as HTML...")
shap.save_html("shap_force_plot.html", force_plot)
print("SHAP force plot saved! Open 'shap_force_plot.html' in a browser.")

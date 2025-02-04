import shap
import xgboost
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load California Housing dataset (Alternative to Boston Housing)
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = xgboost.XGBRegressor().fit(X_train, y_train)

# Explain predictions using SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Visualizing feature importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

###"""abs
# 
# Interpreting the SHAP Summary Plot
"""
X-axis (SHAP value):

Represents how much each feature pushes the model output higher or lower.
Negative SHAP values → Lower predicted values.
Positive SHAP values → Higher predicted values.
Y-axis (Feature Names):

Lists the features in order of importance (most important feature at the top).
Color Scale (Feature Value Intensity):

Red (High Feature Value): Higher values of the feature.
Blue (Low Feature Value): Lower values of the feature.
Helps understand whether higher values increase or decrease the prediction.
Key Takeaways
✅ Most Important Feature → MedInc (Median Income)

Higher values (red) push predictions higher.
Lower values (blue) push predictions lower.
Suggests that higher median income increases housing prices.
✅ Other Key Features:

Latitude and Longitude show a geographical effect.
AveRooms (Average Rooms per House) and HouseAge also play a role.
Population and AveBedrms seem less impactful.
✅ Feature Impact Direction:

If red points are mostly on the right, the higher feature values increase the prediction.
If red points are mostly on the left, the higher feature values decrease the prediction."""
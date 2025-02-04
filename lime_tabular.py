# ✅ Import Necessary Libraries
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# ✅ Load the California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train a RandomForestRegressor (Good for LIME)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Create a LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train, 
    feature_names=feature_names, 
    mode="regression"
)

# ✅ Explain a Single Prediction (e.g., 5th test instance)
instance_index = 5  # Choose the 5th sample
instance = X_test[instance_index]
explanation = explainer.explain_instance(instance, model.predict)

# ✅ Save LIME Explanation as an HTML File (For Non-Jupyter Users)
explanation.save_to_file("lime_tabular_explanation.html")

print("LIME explanation saved! Open 'lime_explanation.html' in a browser to view it.")

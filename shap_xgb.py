import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Model prediction
#predictions = model.predict(X_test)

# Using SHAP for model interpretation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Ensure shap_values and feature_names are properly handled
feature_names = data.feature_names

for class_index in range(shap_values.shape[2]):
    print(f"SHAP summary plot for class {class_index}:")
    shap.summary_plot(shap_values[..., class_index], X_test, feature_names=feature_names)

# Choose a single instance to analyze
instance_index = 0  # Change this index for another instance
instance_shap_values = shap_values[instance_index, :, 0]  # SHAP values for instance 0, class 0

# Generate a bar plot
plt.figure(figsize=(10, 6))
plt.barh(feature_names, instance_shap_values)
plt.xlabel("SHAP value (impact on model output)")
plt.title(f"SHAP Values for Instance {instance_index} (Class 0)")
plt.grid(axis='x')
plt.tight_layout()
plt.show()
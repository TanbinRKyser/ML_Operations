import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train, 
    feature_names=feature_names, 
    mode="regression"
)

# 5th instance
instance_index = 5
explanation = explainer.explain_instance(X_test[instance_index], model.predict)


explanation.save_to_file("lime_explanation.html")

print("LIME explanation saved! Open 'lime_explanation.html' in your browser to view it.")


"""
LIME Explanation Interpretation for California Housing Dataset
This LIME (Local Interpretable Model-Agnostic Explanations) visualization explains how individual features influenced the predicted house price for a specific data instance.

1. Predicted Value
The model predicts 1.65 as the house price (in $100,000s, meaning ~$165,000).
The minimum possible price in the dataset is 0.56 (~$56,000).
The maximum possible price in the dataset is 4.98 (~$498,000).
2. Feature Contributions (Positive vs. Negative Impact)
Negative Features (Blue) → These features decreased the predicted price.
Positive Features (Orange) → These features increased the predicted price.
Top Negative Features (Lowering Price Prediction)
Longitude > -118.01 (-0.57 impact)
The location of the house (longitude) is pushing the price lower.
AveOccup ≤ 2.82 (-0.21 impact)
Fewer occupants per house leads to a lower price.
HouseAge ≤ 18.00 (-0.14 impact)
Newer houses seem to have a lower price in this case.
MedInc ≤ 4.71 (-0.13 impact)
A lower median income in the neighborhood negatively affects house price.
Top Positive Features (Increasing Price Prediction)
Latitude between 33.93 and ... (+0.36 impact)
The house’s latitude indicates a location where prices are higher.
Population > 1726 (+0.06 impact)
A more populated area has a small positive effect on price.
Average Rooms > 5.24 (+0.02 impact)
More rooms per house slightly increase price.
3. Feature Table
Shows the actual values of the features for this specific instance.
Longitude (-117.61) and Latitude (34.08) are the strongest indicators.
Other features like Average Occupancy (2.85) and Median Income (4.71) also played a role.
Key Takeaways
✅ Geographical Features Matter

Longitude has the largest negative effect, possibly indicating a location with lower house prices.
Latitude has a strong positive effect, likely showing a desirable area.
✅ Socioeconomic Features Influence Price

Lower median income pushes house prices down.
Higher occupancy per house slightly decreases price.
✅ Structural Features Have Smaller Impact

Average rooms per house only has a minor influence on price prediction.
"""
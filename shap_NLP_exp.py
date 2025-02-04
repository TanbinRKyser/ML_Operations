import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# ✅ Load dataset (Sentiment-based categories)
categories = ['rec.autos', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
X_text = newsgroups.data  # Text data
y = newsgroups.target  # Labels (0 or 1)

# ✅ Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X_text)

# ✅ Train a RandomForest Classifier
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Convert sparse TF-IDF matrices to dense arrays for SHAP
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# ✅ Use a small subset as background for SHAP
background = X_train_dense[:100]

# ✅ Compute SHAP values using valid background
explainer = shap.Explainer(model.predict_proba, background, max_evals=6000)

shap_values = explainer(X_test_dense[:20])  # Limit to 50 samples for speed

# ✅ Generate SHAP Summary Plot
plt.figure(figsize=(10, 5))
shap.summary_plot(shap_values, features=X_test_dense[:50], feature_names=vectorizer.get_feature_names_out())
plt.show()

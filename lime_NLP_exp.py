import lime
import lime.lime_text
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# ✅ Load dataset (Same as SHAP)
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

# ✅ Use LIME to explain a single prediction
lime_explainer = lime.lime_text.LimeTextExplainer(class_names=['Autos', 'Baseball'])
idx = 5  # Select a test instance to explain
sample_text = newsgroups.data[idx]
lime_exp = lime_explainer.explain_instance(sample_text, model.predict_proba, num_features=10)

# ✅ Show LIME Explanation in Notebook
# lime_exp.show_in_notebook(text=True)

# ✅ Save LIME Explanation as an HTML File
lime_exp.save_to_file("lime_NLP_explanation.html")
print("LIME explanation saved! Open 'lime_NLP_explanation.html' in a browser to view it.")

# generate_assets.py

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load Data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 2. Split into Reference (Training) and Current (Production)
# Reference = First 50% of data (Normal)
# Current = Last 50% of data (We will mess this up)
reference = df.iloc[:250]
current = df.iloc[250:].copy()

# 3. SABOTAGE THE DATA (Simulate Drift)
# We artificially increase the 'mean radius' feature in production data
# This simulates a sensor breaking or patient demographics changing
current['mean radius'] = current['mean radius'] + 5.0 

# 4. Train a dummy model
model = RandomForestClassifier(random_state=42)
X_train = reference.drop('target', axis=1)
y_train = reference['target']
model.fit(X_train, y_train)

# 5. Save everything
reference.to_csv("reference.csv", index=False)
current.to_csv("current.csv", index=False)
joblib.dump(model, "model.pkl")

print("✅ Assets generated: reference.csv, current.csv, model.pkl")
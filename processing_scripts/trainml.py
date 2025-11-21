import numpy as np
import joblib
import os
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# Output paths
MODEL_PATH = os.path.join('models', 'symptom_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

# Load real dataset (already includes Age and Gender)
DATA_PATH = os.path.join('processing_scripts', 'symptom_checker_dataset_5000_with_age_gender.csv')
df = pd.read_csv(DATA_PATH)

# Check if Age and Gender exist; if not, simulate them
if 'Age' not in df.columns:
    np.random.seed(42)
    df['Age'] = np.random.randint(20, 80, size=len(df))
if 'Gender' not in df.columns:
    np.random.seed(42)
    df['Gender'] = np.random.randint(0, 2, size=len(df))

# Separate features and label
X = df.drop('Diagnosis', axis=1).values  # 10 symptoms + Age + Gender = 12 features
y = df['Diagnosis'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_scaled, y)  # Full fit here (not partial_fit)

# Save model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ Model trained and saved using dataset with Age and Gender included.")

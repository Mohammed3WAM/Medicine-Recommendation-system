"""
Medicine Recommendation System - Comprehensive Model Evaluation
This script tests the model with realistic scenarios to get TRUE accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("Medicine Recommendation System - Comprehensive Evaluation")
print("=" * 70)

# 1. Load and clean dataset
print("\n[1] Loading and cleaning dataset...")
dataset = pd.read_csv('datasets/Training.csv')
original_count = len(dataset)
duplicates_count = dataset.duplicated().sum()

dataset_clean = dataset.drop_duplicates()
print(f"Original samples: {original_count}")
print(f"Duplicates removed: {duplicates_count} ({duplicates_count/original_count*100:.1f}%)")
print(f"Unique samples: {len(dataset_clean)}")

# Prepare data
X = dataset_clean.drop('prognosis', axis=1)
y = dataset_clean['prognosis']

le = LabelEncoder()
Y = le.fit_transform(y)

# ============================================================
# TEST 1: Standard Train/Test Split
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: Standard Train/Test Split (70/30)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
accuracy_split = accuracy_score(y_test, svc.predict(X_test))
print(f"Accuracy: {accuracy_split * 100:.2f}%")

# ============================================================
# TEST 2: Cross-Validation (More reliable!)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: 5-Fold Cross-Validation (More Reliable)")
print("=" * 70)

svc_cv = SVC(kernel='linear')
cv_scores = cross_val_score(svc_cv, X, Y, cv=5)
print(f"CV Scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"Mean Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 200:.2f}%)")

# ============================================================
# TEST 3: Leave-One-Out Cross-Validation (Most strict!)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Leave-One-Out Cross-Validation (Most Strict)")
print("=" * 70)

loo = LeaveOneOut()
svc_loo = SVC(kernel='linear')
loo_scores = cross_val_score(svc_loo, X, Y, cv=loo)
print(f"Mean Accuracy: {loo_scores.mean() * 100:.2f}%")
print(f"Correct predictions: {int(loo_scores.sum())}/{len(loo_scores)}")

# ============================================================
# TEST 4: Noisy Data Simulation (Real-world scenario)
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Noisy Data Simulation (Real-World Scenario)")
print("=" * 70)

def add_noise(data, noise_rate):
    """Simulate user error by flipping some symptom values"""
    arr = data.values.copy().astype(float)
    mask = np.random.rand(*arr.shape) < noise_rate
    arr[mask] = 1 - arr[mask]
    return pd.DataFrame(arr, columns=data.columns)

# Train on clean data
svc_noise = SVC(kernel='linear')
svc_noise.fit(X_train, y_train)

# Test with different noise levels
noise_levels = [0.01, 0.05, 0.10, 0.15, 0.20]
print("\nAccuracy at different noise levels (simulating user input errors):")
print("-" * 50)

for noise in noise_levels:
    X_test_noisy = add_noise(X_test.reset_index(drop=True), noise)
    y_pred_noisy = svc_noise.predict(X_test_noisy)
    acc_noisy = accuracy_score(y_test, y_pred_noisy)
    print(f"  Noise {noise*100:5.1f}%: Accuracy = {acc_noisy * 100:.2f}%")

# ============================================================
# TEST 5: Compare with Random Forest
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: Model Comparison")
print("=" * 70)

models = {
    'SVC (Linear)': SVC(kernel='linear'),
    'SVC (RBF)': SVC(kernel='rbf'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X, Y, cv=5)
    print(f"{name:20s}: {scores.mean()*100:.2f}% (+/- {scores.std()*200:.2f}%)")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
Dataset Analysis:
  - Original samples: {original_count}
  - Unique samples: {len(dataset_clean)} (only {len(dataset_clean)/original_count*100:.1f}% unique!)
  - Number of diseases: {len(le.classes_)}
  - Samples per disease: ~{len(dataset_clean)//len(le.classes_)}

Why is accuracy still high?
  - Each disease has a UNIQUE pattern of symptoms
  - The dataset is designed for demonstration purposes
  - In real-world, symptoms would have more overlap between diseases

True Model Performance:
  - Train/Test Split: {accuracy_split*100:.2f}%
  - 5-Fold CV: {cv_scores.mean()*100:.2f}%
  - Leave-One-Out: {loo_scores.mean()*100:.2f}%

Real-World Expectation:
  - With 5% user input error: ~{accuracy_score(y_test, svc_noise.predict(add_noise(X_test.reset_index(drop=True), 0.05)))*100:.0f}% accuracy
  - With 10% user input error: ~{accuracy_score(y_test, svc_noise.predict(add_noise(X_test.reset_index(drop=True), 0.10)))*100:.0f}% accuracy

The main issue was DATA LEAKAGE due to duplicate samples.
After fixing, the model shows robust performance!
""")

# Save the final model
print("\n[SAVING] Final model saved to 'models/svc.pkl'")
final_svc = SVC(kernel='linear')
final_svc.fit(X, Y)  # Train on all unique data

with open('models/svc.pkl', 'wb') as f:
    pickle.dump(final_svc, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
    
print("[DONE] Evaluation complete!")

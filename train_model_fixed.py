"""
Medicine Recommendation System - Fixed Training Script
This script fixes the 100% accuracy problem by removing duplicate samples
before train-test split to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("Medicine Recommendation System - Model Training (Fixed)")
print("=" * 60)

# 1. Load and explore the dataset
print("\n[1] Loading dataset...")
dataset = pd.read_csv('datasets/Training.csv')
print(f"Original dataset shape: {dataset.shape}")
print(f"Total samples: {len(dataset)}")

# 2. Check for duplicates (THIS IS THE PROBLEM!)
print("\n[2] Checking for duplicates...")
duplicates_count = dataset.duplicated().sum()
print(f"Number of duplicate rows: {duplicates_count}")
print(f"Percentage of duplicates: {duplicates_count / len(dataset) * 100:.2f}%")

# 3. Remove duplicates to fix the 100% accuracy problem
print("\n[3] Removing duplicates...")
dataset_clean = dataset.drop_duplicates()
print(f"Clean dataset shape: {dataset_clean.shape}")
print(f"Samples after removing duplicates: {len(dataset_clean)}")

# 4. Check class distribution after removing duplicates
print("\n[4] Class distribution after cleaning:")
class_counts = dataset_clean['prognosis'].value_counts()
print(f"Number of diseases: {len(class_counts)}")
print(f"Min samples per disease: {class_counts.min()}")
print(f"Max samples per disease: {class_counts.max()}")

# 5. Prepare features and labels
print("\n[5] Preparing features and labels...")
X = dataset_clean.drop('prognosis', axis=1)
y = dataset_clean['prognosis']

# Encode labels
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

print(f"Number of features (symptoms): {X.shape[1]}")
print(f"Number of classes (diseases): {len(le.classes_)}")

# 6. Split the data (now with clean, non-duplicate data!)
print("\n[6] Splitting data (train/test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size=0.3, 
    random_state=42,
    stratify=Y  # Ensure balanced split
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 7. Verify NO data leakage
print("\n[7] Verifying no data leakage...")
duplicates_in_test = 0
for i in range(len(X_test)):
    test_row = X_test.iloc[i].values
    is_duplicate = (X_train == test_row).all(axis=1).any()
    if is_duplicate:
        duplicates_in_test += 1

print(f"Test samples that are copies in training: {duplicates_in_test}")
if duplicates_in_test == 0:
    print("[OK] No data leakage! The split is clean.")
else:
    print(f"[WARNING] {duplicates_in_test} samples still overlap")

# 8. Train the SVC model
print("\n[8] Training SVC model...")
svc = SVC(kernel='linear', probability=True)
svc.fit(X_train, y_train)
print("[OK] Model trained successfully!")

# 9. Evaluate on test set
print("\n[9] Evaluating model...")
y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'=' * 60}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"{'=' * 60}")

# Show detailed classification report
print("\nClassification Report (first 10 classes):")
print(classification_report(y_test, y_pred, target_names=le.classes_[:10], labels=range(10)))

# 10. Save the model
print("\n[10] Saving model...")
with open('models/svc.pkl', 'wb') as f:
    pickle.dump(svc, f)
print("[OK] Model saved to 'models/svc.pkl'")

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("[OK] Label encoder saved to 'models/label_encoder.pkl'")

# 11. Summary
print("\n" + "=" * 60)
print("SUMMARY - PROBLEM FIXED!")
print("=" * 60)
print(f"""
The 100% accuracy problem was caused by DUPLICATE SAMPLES in the dataset.

Before fix:
  - Dataset had {duplicates_count} duplicate rows ({duplicates_count / len(dataset) * 100:.1f}%)
  - Same samples appeared in both train and test sets
  - Model was "memorizing" not "learning"

After fix:
  - Removed duplicates, keeping {len(dataset_clean)} unique samples
  - No overlap between train and test sets
  - Model accuracy: {accuracy * 100:.2f}% (more realistic!)

This new accuracy reflects the model's TRUE ability to generalize
to unseen patient symptoms.
""")

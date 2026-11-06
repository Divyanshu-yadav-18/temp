import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Load dataset (using Breast Cancer dataset as a binary classification example)
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Scale features (Critical: SVMs are sensitive to feature scales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Initialize the SVM model
svm_model = SVC(kernel="rbf", C=1.0, random_state=42)


# ==========================================
# Method 1: Using cross_val_score (Quickest)
# ==========================================
print("--- Method 1: Using cross_val_score ---")
scores = cross_val_score(svm_model, X_scaled, y, cv=5, scoring="accuracy")

print(f"Fold-wise accuracies: {np.round(scores, 4)}")
print(
    f"Mean Accuracy: {scores.mean():.4f} (+/- standard deviation: {scores.std() * 2:.4f})\n"
)


# ==========================================
# Method 2: Using KFold Explicitly (Detailed)
# ==========================================
print("--- Method 2: Using KFold Explicitly ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_index, test_index) in enumerate(kf.split(X_scaled)):
  X_train, X_test = X_scaled[train_index], X_scaled[test_index]
  y_train, y_test = y[train_index], y[test_index]

  # Train the model on the current fold's training data
  svm_model.fit(X_train, y_train)

  # Evaluate on the test fold
  accuracy = svm_model.score(X_test, y_test)
  fold_accuracies.append(accuracy)
  print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

print(f"\nAverage K-Fold Accuracy: {np.mean(fold_accuracies):.4f}")

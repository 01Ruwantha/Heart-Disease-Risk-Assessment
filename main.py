
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib  # For saving and loading the model

# Step 1: Load the dataset
dataset_path = "heart_disease_data.csv"
data = pd.read_csv(dataset_path)

# Step 2: Data exploration and preprocessing
# Display dataset information
print(data.info())
print(data.describe())
print(data.isnull().sum())  # Check for missing values

# Handle missing values (if any)
data = data.dropna()  # Example: drop rows with missing values

# Visualize target distribution
sns.countplot(x='target', data=data)
plt.title("Heart Disease Distribution")
plt.show()

# Visualize feature correlations
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Feature-target split
X = data.drop('target', axis=1)
y = data['target']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest model
rf_y_pred = rf_model.predict(X_test)
rf_y_prob = rf_model.predict_proba(X_test)[:, 1]

# Classification metrics
print("Random Forest - Confusion Matrix:\n", confusion_matrix(y_test, rf_y_pred))
print("\nRandom Forest - Classification Report:\n", classification_report(y_test, rf_y_pred))

# ROC-AUC for Random Forest
rf_roc_auc = roc_auc_score(y_test, rf_y_prob)
fpr, tpr, _ = roc_curve(y_test, rf_y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"Random Forest AUC = {rf_roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.legend()
plt.show()

# Step 5: Feature Importance
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances)
plt.title("Feature Importances - Random Forest")
plt.show()

# Step 6: Cross-validation for Random Forest
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5)
print(f"Random Forest - Cross-validation Accuracy: {cv_scores.mean():.2f}")

# Step 7: Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best Hyperparameters for Random Forest:", grid_search.best_params_)

# Step 8: Compare with other models
# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
lr_roc_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])

# Support Vector Machine
svc_model = SVC(probability=True)
svc_model.fit(X_train, y_train)
svc_roc_auc = roc_auc_score(y_test, svc_model.predict_proba(X_test)[:, 1])

# Comparison Plot
plt.figure(figsize=(10, 6))
plt.bar(['Random Forest', 'Logistic Regression', 'SVM'], [rf_roc_auc, lr_roc_auc, svc_roc_auc], color=['blue', 'orange', 'green'])
plt.title("Model Comparison - ROC-AUC Scores")
plt.ylabel("ROC-AUC Score")
plt.show()

# Save the final Random Forest model and scaler
joblib.dump(grid_search.best_estimator_, "heart_disease_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

#Feature selection visulizer for RFE and LASSO methods
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data1.csv")
# Assuming you have your dataset loaded into a DataFrame named 'data'
# Separate features (X) and target variable (y)
X = data.drop(columns=['Class'])  # Adjust 'target_column' to your target variable
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (if needed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Method 1: Feature Importance with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
print("Feature Importances:")
print(feature_importances)

# Method 2: Recursive Feature Elimination (RFE) with Logistic Regression
logreg = LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear')  # L1 regularization (LASSO)
rfe = RFE(estimator=logreg, n_features_to_select=10, step=1)
rfe.fit(X_train_scaled, y_train)

# Get selected features
selected_features_rfe = X.columns[rfe.support_]
print("Selected Features (RFE):")
print(selected_features_rfe)

# Method 3: L1 Regularization (LASSO) with Logistic Regression
logreg_l1 = LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear')  # L1 regularization (LASSO)
logreg_l1.fit(X_train_scaled, y_train)

# Get selected features
selected_features_lasso = X.columns[np.where(logreg_l1.coef_ != 0)[1]]
print("Selected Features (LASSO):")
print(selected_features_lasso)

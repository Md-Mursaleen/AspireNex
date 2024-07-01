import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=3'
df = pd.read_csv(r'c:\Users\md mursaleen\Downloads\creditcard.csv')

# Explore the dataset
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Define features and target variable
X = df.drop(columns=['Class'])
y = df['Class']

# Normalize the feature set
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train and evaluate Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
class_report_log_reg = classification_report(y_test, y_pred_log_reg)

print("Logistic Regression Model")
print("Accuracy:", accuracy_log_reg)
print("Confusion Matrix:\n", conf_matrix_log_reg)
print("Classification Report:\n", class_report_log_reg)

# Train and evaluate Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
class_report_tree = classification_report(y_test, y_pred_tree)

print("\nDecision Tree Model")
print("Accuracy:", accuracy_tree)
print("Confusion Matrix:\n", conf_matrix_tree)
print("Classification Report:\n", class_report_tree)

# Train and evaluate Random Forest model
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
y_pred_forest = random_forest.predict(X_test)
accuracy_forest = accuracy_score(y_test, y_pred_forest)
conf_matrix_forest = confusion_matrix(y_test, y_pred_forest)
class_report_forest = classification_report(y_test, y_pred_forest)

print("\nRandom Forest Model")
print("Accuracy:", accuracy_forest)
print("Confusion Matrix:\n", conf_matrix_forest)
print("Classification Report:\n", class_report_forest)

# Feature Importance Analysis using Random Forest
feature_importances = random_forest.feature_importances_
features = df.drop(columns=['Class']).columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Use only important features
important_features = importance_df[importance_df['Importance'] > 0.01]['Feature'].values
X_important = df[important_features]

# Normalize the important features set
X_important = scaler.fit_transform(X_important)

# Split the important features data into training and testing sets
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_important, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning using Grid Search for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train_imp, y_train_imp)

best_params_rf = grid_search_rf.best_params_
best_model_rf = grid_search_rf.best_estimator_

# Hyperparameter tuning using Grid Search for Decision Tree
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_dt = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid_dt, cv=3, n_jobs=-1, verbose=2)
grid_search_dt.fit(X_train_imp, y_train_imp)

best_params_dt = grid_search_dt.best_params_
best_model_dt = grid_search_dt.best_estimator_

# Evaluate the best models on the test dataset
y_pred_best_rf = best_model_rf.predict(X_test_imp)
accuracy_best_rf = accuracy_score(y_test_imp, y_pred_best_rf)
conf_matrix_best_rf = confusion_matrix(y_test_imp, y_pred_best_rf)
class_report_best_rf = classification_report(y_test_imp, y_pred_best_rf)

print("\nBest Random Forest Model with Grid Search")
print("Accuracy:", accuracy_best_rf)
print("Confusion Matrix:\n", conf_matrix_best_rf)
print("Classification Report:\n", class_report_best_rf)

y_pred_best_dt = best_model_dt.predict(X_test_imp)
accuracy_best_dt = accuracy_score(y_test_imp, y_pred_best_dt)
conf_matrix_best_dt = confusion_matrix(y_test_imp, y_pred_best_dt)
class_report_best_dt = classification_report(y_test_imp, y_pred_best_dt)

print("\nBest Decision Tree Model with Grid Search")
print("Accuracy:", accuracy_best_dt)
print("Confusion Matrix:\n", conf_matrix_best_dt)
print("Classification Report:\n", class_report_best_dt)

# Train Logistic Regression on important features
log_reg.fit(X_train_imp, y_train_imp)
y_pred_log_reg_imp = log_reg.predict(X_test_imp)
accuracy_log_reg_imp = accuracy_score(y_test_imp, y_pred_log_reg_imp)
conf_matrix_log_reg_imp = confusion_matrix(y_test_imp, y_pred_log_reg_imp)
class_report_log_reg_imp = classification_report(y_test_imp, y_pred_log_reg_imp)

print("\nLogistic Regression Model on Important Features")
print("Accuracy:", accuracy_log_reg_imp)
print("Confusion Matrix:\n", conf_matrix_log_reg_imp)
print("Classification Report:\n", class_report_log_reg_imp)

# Ensemble using Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('lr', log_reg),
    ('dt', best_model_dt),
    ('rf', best_model_rf)
], voting='hard')

voting_clf.fit(X_train_imp, y_train_imp)
y_pred_voting = voting_clf.predict(X_test_imp)

accuracy_voting = accuracy_score(y_test_imp, y_pred_voting)
conf_matrix_voting = confusion_matrix(y_test_imp, y_pred_voting)
class_report_voting = classification_report(y_test_imp, y_pred_voting)

print("\nVoting Classifier")
print("Accuracy:", accuracy_voting)
print("Confusion Matrix:\n", conf_matrix_voting)
print("Classification Report:\n", class_report_voting)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

# Normalize the 'Time' and 'Amount' features
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# Define features and target variable
X = df.drop(columns=['Class'])
y = df['Class']

# Normalize the feature set
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Logistic Regression model
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

# Train a Decision Tree model
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

# Train a Random Forest model
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

# Compare model performances
accuracies = {
    "Logistic Regression": accuracy_log_reg,
    "Decision Tree": accuracy_tree,
    "Random Forest": accuracy_forest
}

best_model_name = max(accuracies, key=accuracies.get)
best_model_accuracy = accuracies[best_model_name]

print("\nBest Model: ", best_model_name)
print("Accuracy: ", best_model_accuracy)

# Evaluate the best model on the test dataset
if best_model_name == "Logistic Regression":
    best_model = log_reg
elif best_model_name == "Decision Tree":
    best_model = decision_tree
else:
    best_model = random_forest

y_pred_best = best_model.predict(X_test)
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
class_report_best = classification_report(y_test, y_pred_best)

print("\nBest Model Evaluation on Test Data")
print("Confusion Matrix:\n", conf_matrix_best)
print("Classification Report:\n", class_report_best)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Convert TXT files to CSV
def convert_txt_to_csv(txt_file, csv_file, columns, delimiter=' ::: '):
    df = pd.read_csv(txt_file, delimiter=delimiter, engine='python', names=columns)
    df.to_csv(csv_file, index=False)

convert_txt_to_csv(r'c:\Users\md mursaleen\Downloads\train_data.txt', 'traindata.csv', ['ID', 'Title', 'Genre', 'Description'])
convert_txt_to_csv(r'c:\Users\md mursaleen\Downloads\test_data.txt', 'testdata.csv', ['ID', 'Title', 'Description'])
convert_txt_to_csv(r'c:\Users\md mursaleen\Downloads\test_data_solution.txt', 'testdata_solution.csv', ['ID', 'Genre'])

# Load the training and test data from CSV files
train_data = pd.read_csv('traindata.csv')
test_data = pd.read_csv('testdata.csv')
test_data_solution = pd.read_csv('testdata_solution.csv')

# Explore the dataset
print(train_data.head())
print(test_data.head())
print(train_data.info())
print(test_data.info())
print(train_data.describe())
print(test_data.describe())

# Check for missing values
print(train_data.isnull().sum())
print(test_data.isnull().sum())

# Vectorize the text data
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train = tfidf_vectorizer.fit_transform(train_data['Description'])
y_train = train_data['Genre']
X_test = tfidf_vectorizer.transform(test_data['Description'])
y_test = test_data_solution['Genre']

# Define models
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
naive_bayes = MultinomialNB()
svm = SVC(kernel='linear', class_weight='balanced')

# Train and evaluate Logistic Regression
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
class_report_log_reg = classification_report(y_test, y_pred_log_reg)

print("Logistic Regression Model")
print("Accuracy:", accuracy_log_reg)
print("Confusion Matrix:\n", conf_matrix_log_reg)
print("Classification Report:\n", class_report_log_reg)

# # Train and evaluate Naive Bayes
naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
class_report_nb = classification_report(y_test, y_pred_nb)

print("\nNaive Bayes Model")
print("Accuracy:", accuracy_nb)
print("Confusion Matrix:\n", conf_matrix_nb)
print("Classification Report:\n", class_report_nb)

# # Train and evaluate Support Vector Machine
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
class_report_svm = classification_report(y_test, y_pred_svm)

print("\nSupport Vector Machine Model")
print("Accuracy:", accuracy_svm)
print("Confusion Matrix:\n", conf_matrix_svm)
print("Classification Report:\n", class_report_svm)

# # Compare model performances
accuracies = {
    "Logistic Regression": accuracy_log_reg,
    "Naive Bayes": accuracy_nb,
    "Support Vector Machine": accuracy_svm
}

best_model_name = max(accuracies, key=accuracies.get)
best_model_accuracy = accuracies[best_model_name]

print("\nBest Model: ", best_model_name)
print("Accuracy: ", best_model_accuracy)

# # Evaluate the best model on the test dataset
if best_model_name == "Logistic Regression":
    best_model = log_reg
elif best_model_name == "Naive Bayes":
    best_model = naive_bayes
else:
    best_model = svm

y_pred_best = best_model.predict(X_test)
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
class_report_best = classification_report(y_test, y_pred_best)

print("\nBest Model Evaluation on Test Data")
print("Confusion Matrix:\n", conf_matrix_best)
print("Classification Report:\n", class_report_best)
# Credit Card Fraud Detection

This project aims to build and evaluate models to detect fraudulent credit card transactions. We use a dataset containing information about credit card transactions and experiment with various machine learning algorithms like Logistic Regression, Decision Trees and Random Forests to classify transactions as fraudulent or legitimate.

## Dataset

The dataset used in this project contains credit card transactions with the following features:

- `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1` to `V28`: Principal Component Analysis (PCA) components of the original features to protect confidentiality.
- `Amount`: Transaction amount.
- `Class`: Class label (1 for fraudulent transactions, 0 for legitimate transactions).

## Project Structure

The project is structured as follows:

- `creditcard.xlsx`: The dataset file containing credit card transactions.
- `fraud_detection.py`: The main script to preprocess the data, train models, and evaluate their performance.
- `README.md`: This file, providing an overview and instructions for the project.

## Preprocessing

1. **Normalization**: The `Time` and `Amount` features are normalized using `StandardScaler`.
2. **Splitting Data**: The dataset is split into training and testing sets using an 80-20 split ratio, with stratification to maintain class balance.

## Models

Three machine learning models are trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**

## Evaluation

The models are evaluated using the following metrics:

- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

The best model is determined based on the highest accuracy score and is further evaluated on the test dataset.

## Usage

To run the project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. **Install dependencies**:
    ```sh
    pip install pandas scikit-learn
    ```

3. **Run the script**:
    ```sh
    python fraud_detection.py
    ```

# Credit Card Fraud Detection

## Table of Contents
1. Introduction
2. Installation
3. Usage
4. Pipeline Description
5. Benefits to the Company
6. Future Work

## Introduction
This project aims to detect fraudulent credit card transactions using various machine learning models. The dataset used is highly imbalanced, so techniques like under-sampling were employed to balance the dataset. Multiple models were trained, and the best-performing model was selected and saved for deployment.

## Installation
To run this project, follow these steps:

1. **Download the dataset:**
    Download the `creditcard.csv` dataset and place it in the root directory of the project.

## Usage
1. **Run the end-to-end pipeline:**
    ```bash
    python credit_card_fraud_detection.py
    ```

2. **Predict using the trained model:**
    ```python
    from joblib import load
    import pandas as pd

    # Load the model
    model = load('best_rf_model.pkl')

    # Load new data
    new_data = pd.read_csv('new_data.csv')

    # Preprocess new data similarly as done in the pipeline
    # ...

    # Make predictions
    predictions = model.predict(new_data)
    print(predictions)
    ```

## Pipeline Description
The pipeline consists of the following steps:

1. **Data Collection:**
    - Loading the dataset from `creditcard.csv`.

2. **Exploratory Data Analysis (EDA):**
    - Checking data quality, handling missing values and outliers.
    - Normalizing 'Amount' and 'Time' columns.

3. **Balancing the Data:**
    - Using under-sampling to balance the dataset.

4. **Feature Engineering and Selection:**
    - Creating new features and selecting the most relevant ones.

5. **Train/Test Split:**
    - Splitting the data into training, validation, and test sets.

6. **Model Training and Evaluation:**
    - Training multiple models: Logistic Regression, Shallow Neural Network, Random Forest, Gradient Boosting, and LinearSVC.
    - Evaluating models using metrics such as Accuracy, Precision, Recall, F1-Score, and AUC-ROC.

7. **Hyperparameter Tuning:**
    - Using RandomizedSearchCV to find the best hyperparameters for Random Forest.

8. **Model Deployment:**
    - Saving the best model using joblib.
    - Loading the model for predictions.

## Benefits to the Company
1. **Improved Fraud Detection:**
    - This solution can significantly reduce the number of fraudulent transactions, saving the company from potential financial losses.

2. **Scalability:**
    - The pipeline is designed to be scalable, allowing for easy integration with the company's existing systems.

3. **Efficiency:**
    - Automating the detection process increases efficiency and reduces the need for manual review of transactions.

4. **Data-Driven Insights:**
    - The models provide insights into patterns of fraudulent activity, helping in the development of more robust fraud prevention strategies.

## Future Work
1. **Enhanced Feature Engineering:**
    - Explore additional features that could improve model performance.

2. **Real-time Fraud Detection:**
    - Implement real-time detection and alert systems.

3. **Model Ensemble:**
    - Combine multiple models to further improve accuracy and robustness.

4. **Continuous Learning:**
    - Implement mechanisms for the model to continuously learn from new data.

---

You can copy this text into your README file or any other document as needed.
The repository contains Upgrad Capstone Project 2 - FindDefault - Credit card fradulent transactions

# Credit-Risk-Benchmark

*Credit Risk Prediction*

**Background:**
Credit risk assessment helps financial institutions figure out the chances of a borrower failing to repay a loan or credit. Early identification of risky borrowers allows lenders to adjust terms or take action to avoid losses. Traditionally, banks used manual methods and financial ratios to assess credit risk. But now, machine learning models can make this process more accurate and efficient.

**Problem:**
We are given a dataset with borrower information, and our goal is to build a machine learning model to predict whether a borrower will default on a loan in the next two years. The target variable tells us if a borrower has defaulted (1) or not (0) within that time. The task is a binary classification, and we will predict one of the two outcomes:

* **0:** No serious delinquency (No risk)
* **1:** Serious delinquency (Risk of default)

To make this prediction, we will use different features like:

* **RevolvingUtilizationOfUnsecuredLines:** The borrower’s use of unsecured credit lines
* **DebtRatio:** The borrower’s debt-to-income ratio
* **MonthlyIncome:** The borrower’s monthly income
* **Age:** The borrower’s age
* **NumberOfTime30-59DaysPastDueNotWorse:** How often the borrower has been 30-59 days late on payments
* And other financial details

**Data:**
The dataset is taken from kaggle from https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset .
The dataset contains about 16714 records with 10 numerical features, such as income, age, credit usage, and payment history. The target variable is called **dlq_2yrs**, which shows whether a borrower defaulted or not in the next 2 years.

**Challenges:**

* **Outliers:** Some features, like **MonthlyIncome**, might have extreme values that could mess with the model’s predictions.
* **Feature Importance:** We need to figure out which features are most important for predicting default risk.
* **Model Evaluation:** We need to make sure the model doesn’t always predict the majority class (non-defaults) and misses the minority class (defaults).

---

## **Project Workflow**

1. **Data Reading**

   * Load the dataset using pandas.
   * Perform basic data inspection with methods like `head()`, `info()`, and `describe()` to check columns, data types, and missing values.

2. **Data Exploration**

   * Check for null values and handle them appropriately.
   * Evaluate unique values, value counts, and basic statistical summaries for the features.

3. **Data Visualization**

   * **Box Plots:** Detect outliers in numerical data.

5. **Feature and Target Selection**

   * Select the target variable: **dlq_2yrs**.
   * The features to be used for prediction:

     * RevolvingUtilizationOfUnsecuredLines
     * DebtRatio
     * MonthlyIncome
     * Age
     * NumberOfTime30-59DaysPastDueNotWorse
     * NumberOfOpenCreditLinesAndLoans
     * NumberOfTimes90DaysLate
     * NumberOfTime60-89DaysPastDueNotWorse
     * NumberOfDependents
     * NumberRealEstateLoansOrLines

6. **Feature Importance**

   * Use **Random Forest** to compute and display the importance of each feature.
   * Identify and rank the top features contributing to predictions.

7. **Outlier Removal**

   * Detect and remove outliers from numerical features (e.g., MonthlyIncome, Age) using **IQR (Interquartile Range)** to enhance data quality.

8. **SMOTE Application**

   * After removing outliers. The data becomes a bit imbalence
   *To  Handle the class imbalance in the target variable **SMOTE (Synthetic Minority Over-sampling Technique)** is used.

9. **Model Training**

   * Train four models:

     * **Support Vector Machine (SVM)**
     * **Random Forest Classifier**
     * **XGBoost Classifier**
     * **LightGBM Classifier**
   * Train each model on the **SMOTE-balanced data** and evaluate them on the test set.

10. **Model Evaluation**

   * **Classification Report:** Include performance metrics like precision, recall, f1-score, and support.
   * **Confusion Matrix:** Display true/false positives and negatives for each model to evaluate the performance.

---

## **About App**

#### **Project Overview:**

The **Credit Risk Benchmark App** is a machine learning-based web application designed to predict the likelihood of a borrower defaulting on a loan. The app uses a variety of borrower attributes such as income, credit utilization, payment history, and demographic information to predict the borrower’s risk.

#### **Key Features:**

* **Home Page:**

  * Provides an introduction to the Credit Risk Benchmark problem, including an overview of the dataset and features used for predicting loan default.
  * Details the solution workflow, including data cleaning, feature selection, class balancing, model training, and evaluation.

* **Predict Risk Page:**

  * Allows users to input borrower data (e.g., income, credit utilization, age, etc.).
  * Outputs a prediction about the likelihood of the borrower defaulting, along with a visual message indicating whether the borrower is predicted to default or repay the loan.

* **Joke Break Page:**

  * Provides a fun break with a collection of money, loan, and credit-related jokes.
  * Displays 1 jokes at a time with the option for the user to get new jokes by clicking a button.

#### **Streamlit Deployment:**

* **Navigation**: Sidebar with tabs for Home, Prediction, and Jokes.
* **Home**: The page shown about about credit risk problem and ans shows about the solution built.
* **Prediction Page**: A user-friendly interface for entering borrower details and receiving predictions.
* **Jokes Page**: A lighthearted section to refresh the user with money-related jokes.


#### **Files Included:**

1. **cred\_risk\_app.py** - The main Streamlit app file containing the logic for predictions and jokes.
2. **model.pkl** - The trained machine learning model used to make predictions.
3. **requirements.txt** - A text file with a list of required libraries (e.g., `streamlit`, `scikit-learn`, `xgboost`, etc.) for the app to function.
4. **credit.ip** - The file with all the data exploration, pre-processing and model training.

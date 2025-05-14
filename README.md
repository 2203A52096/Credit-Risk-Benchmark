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
The dataset is taken from kaggle from https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset
The dataset contains about 16714 records with 10 numerical features, such as income, age, credit usage, and payment history. The target variable is called **dlq_2yrs**, which shows whether a borrower defaulted or not in the next 2 years.

**Challenges:**

* **Outliers:** Some features, like **MonthlyIncome**, might have extreme values that could mess with the model’s predictions.
* **Feature Importance:** We need to figure out which features are most important for predicting default risk.
* **Model Evaluation:** We need to make sure the model doesn’t always predict the majority class (non-defaults) and misses the minority class (defaults).

---

**Project Workflow**

1. **Data Reading**

   * Load the dataset using pandas.
   * Perform basic data inspection with methods like `head()`, `info()`, and `describe()` to check columns, data types, and missing values.

2. **Data Exploration**

   * Check for null values and handle them appropriately.
   * Evaluate unique values, value counts, and basic statistical summaries for the features.

3. **Data Visualization**

   * **Histograms:** Visualize the distribution of numerical features to understand their spread.
   * **Box Plots:** Detect outliers in numerical data.
   * **Scatter Plots / Correlation Matrix:** Explore relationships between variables and their correlation.

4. **Feature and Target Selection**

   * Select the target variable: **SeriousDlqin2yrs**.
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

5. **Feature Importance**

   * Use models like **XGBoost** or **Random Forest** to compute and display the importance of each feature.
   * Identify and rank the top features contributing to predictions.

6. **Outlier Removal**

   * Detect and remove outliers from numerical features (e.g., MonthlyIncome, Age) using **IQR (Interquartile Range)** or **Z-score** to enhance data quality.

7. **SMOTE Application**

   * Handle the class imbalance in the target variable using **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples of the minority class, ensuring a balanced dataset.

8. **Model Training**

   * Train four models:

     * **Support Vector Machine (SVM)**
     * **Random Forest Classifier**
     * **XGBoost Classifier**
     * **LightGBM Classifier**
   * Train each model on the **SMOTE-balanced data** and evaluate them on the test set.

9. **Model Evaluation**

   * **Classification Report:** Include performance metrics like precision, recall, f1-score, and support.
   * **Confusion Matrix:** Display true/false positives and negatives for each model to evaluate the performance.

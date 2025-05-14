# Credit-Risk-Benchmark

**Credit Risk Prediction**

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
The dataset contains about 16714 records with 10 numerical features, such as income, age, credit usage, and payment history. The target variable is called **dlq_2yrs**, which shows whether a borrower defaulted or not in the next 2 years.

**Challenges:**

* **Class Imbalance:** There are fewer defaulters compared to non-defaulters, which can cause the model to favor predicting non-defaults and ignore the default cases.
* **Outliers:** Some features, like **MonthlyIncome**, might have extreme values that could mess with the model’s predictions.
* **Feature Importance:** We need to figure out which features are most important for predicting default risk.
* **Model Evaluation:** We need to make sure the model doesn’t always predict the majority class (non-defaults) and misses the minority class (defaults).

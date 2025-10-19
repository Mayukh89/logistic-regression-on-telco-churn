============================================================
Logistic Regression
============================================================

Description:
This script builds a logistic regression model to predict customer churn
(i.e., whether a telecom customer leaves the company). The data comes from a
public Telco Churn dataset.

Churn means a customer has stopped using a company's service.
- churn = 1 → customer left (churned)
- churn = 0 → customer stayed

The script follows these steps:
1. Import required libraries
2. Load dataset (from URL or local CSV)
3. Select the important columns (features + churn)
4. Create X (features) and y (target)
5. Standardize features for better model performance
6. Split data into training and testing sets
7. Train Logistic Regression model
8. Predict class labels (0/1) and probabilities
9. Inspect learned coefficients (feature importance)
10. Evaluate model performance using Log Loss
11. Plot and save feature coefficients graph


Output:
- Prints dataset info, shapes, and predictions
- Saves coefficient plot as `coefficients.png`
- Displays model log loss on the test data

# Logistic Regression on Telco Churn (compiled script)
# ----------------------------------------------------
# Exact flow:
# 1) Import libraries
# 2) Load dataset
# 3) Select columns (features + target)
# 4) Build X (features) and y (target)
# 5) Standardize features
# 6) Train/test split
# 7) Train LogisticRegression
# 8) Predict (labels and probabilities)
# 9) Inspect coefficients (plot + print)
# 10) Evaluate with log loss

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


def main():
    # ---------------------------
    # Load the dataset
    # ---------------------------
    # If you have a local CSV:
    # churn_df = pd.read_csv("ChurnData.csv")
    url = "https://raw.githubusercontent.com/IBM/skillsnetwork/master/coursera_ml/telecom_churn/ChurnData.csv"
    churn_df = pd.read_csv(url)

    print("\nLoaded data shape:", churn_df.shape)
    print("Columns:", list(churn_df.columns))
    print("\nHead:\n", churn_df.head())

    # ---------------------------
    # Keep only chosen columns
    # ---------------------------
    cols = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']
    missing = [c for c in cols if c not in churn_df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in the dataset: {missing}")
    churn_df = churn_df[cols]

    # ---------------------------
    # Build X (features) and y (target)
    # ---------------------------
    X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
    y = np.asarray(churn_df['churn'])

    print("\nFeature matrix X shape:", X.shape)
    print("Target vector y shape:", y.shape)
    print("Class balance (value counts):\n", pd.Series(y).value_counts())

    # ---------------------------
    # Standardize features
    # ---------------------------
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)

    # ---------------------------
    # Train/test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=4
    )
    print("\nTrain shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    # ---------------------------
    # Train Logistic Regression
    # ---------------------------
    LR = LogisticRegression().fit(X_train, y_train)

    # ---------------------------
    # Predictions
    # ---------------------------
    yhat = LR.predict(X_test)
    yhat_prob = LR.predict_proba(X_test)

    print("\nFirst 10 predicted labels:", yhat[:10])
    print("First 5 predicted probabilities:\n", yhat_prob[:5])

    # ---------------------------
    # Inspect coefficients
    # ---------------------------
    coef = pd.Series(
        LR.coef_[0],
        index=['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
    )
    print("\nCoefficients (sorted):\n", coef.sort_values())

    # Plot coefficients
    ax = coef.sort_values().plot(kind='barh', title='Feature Coefficients in Logistic Regression Churn Model')
    ax.set_xlabel('Coefficient Value')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig('coefficients.png', dpi=150)
    plt.close(fig)
    print("Saved coefficient plot to coefficients.png")

    # ---------------------------
    # Evaluate with log loss
    # ---------------------------
    ll = log_loss(y_test, yhat_prob)
    print("\nLog loss on test set:", ll)


if __name__ == "__main__":
    main()

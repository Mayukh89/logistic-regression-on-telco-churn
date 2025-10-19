# Logistic Regression on Telco Churn

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

    # If you have a local CSV:
    # churn_df = pd.read_csv("ChurnData.csv")
    url = "https://raw.githubusercontent.com/IBM/skillsnetwork/master/coursera_ml/telecom_churn/ChurnData.csv"
    churn_df = pd.read_csv(url)

    print("\nLoaded data shape:", churn_df.shape)
    print("Columns:", list(churn_df.columns))
    print("\nHead:\n", churn_df.head())

    cols = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']
    missing = [c for c in cols if c not in churn_df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in the dataset: {missing}")
    churn_df = churn_df[cols]

    X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
    y = np.asarray(churn_df['churn'])

    print("\nFeature matrix X shape:", X.shape)
    print("Target vector y shape:", y.shape)
    print("Class balance (value counts):\n", pd.Series(y).value_counts())


    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=4
    )
    print("\nTrain shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    
    LR = LogisticRegression().fit(X_train, y_train)

   
    yhat = LR.predict(X_test)
    yhat_prob = LR.predict_proba(X_test)

    print("\nFirst 10 predicted labels:", yhat[:10])
    print("First 5 predicted probabilities:\n", yhat_prob[:5])


    coef = pd.Series(
        LR.coef_[0],
        index=['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
    )
    print("\nCoefficients (sorted):\n", coef.sort_values())

  
    ax = coef.sort_values().plot(kind='barh', title='Feature Coefficients in Logistic Regression Churn Model')
    ax.set_xlabel('Coefficient Value')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig('coefficients.png', dpi=150)
    plt.close(fig)
    print("Saved coefficient plot to coefficients.png")

    ll = log_loss(y_test, yhat_prob)
    print("\nLog loss on test set:", ll)


if __name__ == "__main__":
    main()

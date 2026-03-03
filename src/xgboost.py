import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


def xgb_model(xtrain, ytrain, xtest, ytest):
    # Initialize and fit the XGBoost classifier
    xgb_classifier = XGBClassifier()
    xgb_classifier.fit(xtrain, ytrain)

    # Predict probabilities for the test set
    ypred_proba = xgb_classifier.predict_proba(xtest)[:, 1]

    # Convert probabilities to binary labels based on a threshold (0.5 used here)
    ypred_binary = (ypred_proba > 0.5).astype(int)

    # Calculate the ROC AUC score
    roc_auc = roc_auc_score(ytest, ypred_proba)

    # Calculate F1-score, Precision, and Recall
    f1 = f1_score(ytest, ypred_binary)
    precision = precision_score(ytest, ypred_binary)
    recall = recall_score(ytest, ypred_binary)

    # Print the scores
    print(f"Model ROC AUC (XGBoost) = {100 * roc_auc:.2f}%")
    print(f"Model F1-score (XGBoost) = {f1:.3f}")
    print(f"Model Precision (XGBoost) = {precision:.3f}")
    print(f"Model Recall (XGBoost) = {recall:.3f}")

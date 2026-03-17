import argparse
import os
import mlflow

from dataset import load_dataset, split_dataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def xgb_model(
    xtrain,
    ytrain,
    xtest,
    ytest,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    subsample=1.0,
    colsample_bytree=1.0,
    thres=0.5,
):
    # Initialize and fit the XGBoost classifier
    xgb_classifier = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
    )
    xgb_classifier.fit(xtrain, ytrain)

    # Predict probabilities for the test set
    ypred_proba = xgb_classifier.predict_proba(xtest)[:, 1]

    # Convert probabilities to binary labels based on a threshold (0.5 used here)
    ypred_binary = (ypred_proba > thres).astype(int)

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

    return roc_auc, f1, precision, recall, xgb_classifier.get_params()


def main():
    args = argument_parser()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["EXPERIMENT_NAME"])

    # Load dataset
    labels, features, homogenous = load_dataset(args.matrix, args.adjlist)

    # Split dataset into training and testing sets
    xtrain, xtest, ytrain, ytest, _, _ = split_dataset(features, labels)

    # Train and evaluate XGBoost model
    with mlflow.start_run():
        roc_auc, f1, precision, recall, params = xgb_model(
            xtrain,
            ytrain,
            xtest,
            ytest,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.lr,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            thres=args.thres,
        )
        mlflow.log_params(params)
        mlflow.log_metrics(
            {"roc_auc": roc_auc, "f1": f1, "precision": precision, "recall": recall}
        )


def argument_parser():
    parser = argparse.ArgumentParser(description="XGBoost Classifier")
    parser.add_argument(
        "--matrix",
        type=str,
        default="data/YelpChi.mat",
        help="the filename of the matrix file",
    )
    parser.add_argument(
        "--adjlist",
        type=str,
        default="data/yelp_home_adjlists.pickle",
        help="the filename of the adjacency list file",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100, help="number of boosting rounds"
    )
    parser.add_argument("--max-depth", type=int, default=6, help="maximum tree depth")
    parser.add_argument(
        "--lr", type=float, default=0.3, help="boosting learning rate (eta)"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="subsample ratio of training instances",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=1.0,
        help="subsample ratio of columns per tree",
    )
    parser.add_argument(
        "--thres", type=float, default=0.5, help="classification threshold"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

import argparse
import os
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from dataset import load_dataset, split_dataset

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def logistic_regression(xtrain, ytrain, xtest, ytest, thres=0.5, random_state=0):
    # Fit the logistic regression model
    clf = LogisticRegression(random_state=random_state).fit(xtrain, ytrain)

    # Predict probabilities
    ypred_proba = clf.predict_proba(xtest)[:, 1]

    # Convert probabilities to binary labels
    ypred_binary = (ypred_proba > thres).astype(int)

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(ytest, ypred_proba)

    # Calculate F1-score, Precision, and Recall
    f1 = f1_score(ytest, ypred_binary)
    precision = precision_score(ytest, ypred_binary)
    recall = recall_score(ytest, ypred_binary)

    print(f"Model ROC AUC (logistic regression) = {100 * roc_auc:.2f}%")
    print(f"Model F1-score (logistic regression) = {f1:.3f}")
    print(f"Model Precision (logistic regression) = {precision:.3f}")
    print(f"Model Recall (logistic regression) = {recall:.3f}")

    return roc_auc, f1, precision, recall


def main():
    args = argument_parser()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["EXPERIMENT_NAME"])

    # Load dataset
    labels, features, homogenous = load_dataset(args.matrix, args.adjlist)

    # Split dataset into training and testing sets
    xtrain, xtest, ytrain, ytest = split_dataset(features, labels)

    # Train and evaluate logistic regression model
    with mlflow.start_run():
        mlflow.log_params({"random_state": 0, "threshold": 0.5})
        roc_auc, f1, precision, recall = logistic_regression(
            xtrain, ytrain, xtest, ytest, args.thres, args.random_state
        )
        mlflow.log_metrics(
            {"roc_auc": roc_auc, "f1": f1, "precision": precision, "recall": recall}
        )


def argument_parser():
    parser = argparse.ArgumentParser(description="Logistic Regression")
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
        "--thres",
        type=float,
        default=0.5,
        help="classification threshold (default 0.5)",
    )
    parser.add_argument("--random-state", type=int, default=0, help="random seed")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

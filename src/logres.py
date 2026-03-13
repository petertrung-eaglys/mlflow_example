import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from dataset import load_dataset, split_dataset


def logistic_regression(xtrain, ytrain, xtest, ytest):
    # Fit the logistic regression model
    clf = LogisticRegression(random_state=0).fit(xtrain, ytrain)

    # Predict probabilities
    ypred_proba = clf.predict_proba(xtest)[:, 1]

    # Convert probabilities to binary labels
    ypred_binary = (ypred_proba > 0.5).astype(int)

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


def main():
    args = argument_parser()
    # Load dataset
    labels, features, homogenous = load_dataset(args.matrix, args.adjlist)

    # Split dataset into training and testing sets
    xtrain, xtest, ytrain, ytest = split_dataset(features, labels)

    # Train and evaluate logistic regression model
    logistic_regression(xtrain, ytrain, xtest, ytest)


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

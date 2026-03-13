import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from dataset import load_dataset, split_dataset


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 64),  # Input size is 32 to match the number of features
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    args = argument_parser()
    # Load dataset
    labels, features, homogenous = load_dataset(args.matrix, args.adjlist)

    # Split dataset into training and testing sets
    xtrain, xtest, ytrain, ytest = split_dataset(features, labels)

    # Convert data to PyTorch tensors
    xtrain = torch.FloatTensor(xtrain)
    ytrain = torch.LongTensor(ytrain)
    xtest = torch.FloatTensor(xtest)
    ytest = torch.LongTensor(ytest)

    # Initialize model, criterion, and optimizer
    model = MLP()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    epochs = args.epochs
    best_loss = float("inf")
    patience = args.patience
    counter = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(xtrain).squeeze()
        loss = criterion(output, ytrain.float())
        loss.backward()
        optimizer.step()

        # Evaluation
        with torch.no_grad():
            model.eval()
            test_output = model(xtest).squeeze()
            vloss = criterion(test_output, ytest.float())
            test_prob = torch.sigmoid(test_output)
            ypred_proba = test_prob.numpy()
            ypred_binary = (ypred_proba > 0.5).astype(int)

            roc_auc = roc_auc_score(ytest, ypred_proba)
            f1 = f1_score(ytest, ypred_binary)
            precision = precision_score(ytest, ypred_binary)
            recall = recall_score(ytest, ypred_binary)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {loss.item():.4f} | "
            f"Validation Loss: {vloss.item():.4f} | "
            f"ROC AUC: {roc_auc:.3f} | "
            f"F1-score: {f1:.3f} | "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f}"
        )
        # Early stopping
        if vloss.item() < best_loss:
            best_loss = vloss.item()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break


def argument_parser():
    parser = argparse.ArgumentParser(description="MLP Classifier")
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
        "--epochs", type=int, default=10000, help="number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=100, help="early stopping patience"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

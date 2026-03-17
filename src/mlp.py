import argparse
import logging
import os
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from dataset import load_dataset, split_dataset

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int]):
        super(MLP, self).__init__()
        sizes = [input_size] + hidden_sizes
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers += [nn.Linear(in_size, out_size), nn.ReLU()]
        layers.append(nn.Linear(sizes[-1], 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def main():
    args = argument_parser()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["EXPERIMENT_NAME"])

    logger.info("Loading dataset")
    labels, features, homogenous = load_dataset(args.matrix, args.adjlist)
    xtrain, xtest, ytrain, ytest, _, _ = split_dataset(features, labels)

    # Convert data to PyTorch tensors
    xtrain = torch.FloatTensor(xtrain)
    ytrain = torch.LongTensor(ytrain)
    xtest = torch.FloatTensor(xtest)
    ytest = torch.LongTensor(ytest)

    # Initialize model, criterion, and optimizer
    lr = args.lr
    input_size = xtrain.shape[1]
    model = MLP(input_size, args.hidden_sizes)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training the model
    epochs = args.epochs
    best_loss = float("inf")
    patience = args.patience
    counter = 0
    thres = args.thres

    logger.info("Device: training on cpu")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "patience": patience,
                "lr": lr,
                "threshold": thres,
                "hidden_sizes": args.hidden_sizes,
            }
        )

        logger.info("Starting training")
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            output = model(xtrain).squeeze()
            loss = criterion(output, ytrain.float())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                test_output = model(xtest).squeeze()
                vloss = criterion(test_output, ytest.float())
                test_prob = torch.sigmoid(test_output)
                ypred_proba = test_prob.numpy()
                ypred_binary = (ypred_proba > thres).astype(int)

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
            mlflow.log_metrics(
                {
                    "train_loss": loss.item(),
                    "val_loss": vloss.item(),
                    "roc_auc": roc_auc,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                },
                step=epoch,
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
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[64],
        help="sizes of hidden layers, e.g. --hidden-sizes 64 32",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate for the training",
    )
    parser.add_argument(
        "--thres",
        type=float,
        default=0.5,
        help="classification threshold (default 0.5)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

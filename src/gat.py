import os
import logging
import torch
import mlflow
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from dataset import load_dataset, split_dataset, graph_dataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_sizes: list[int], heads=1, dropout_p=0.0):
        super().__init__()
        torch.manual_seed(2022)
        self.dropout_p = dropout_p
        in_sizes = [num_features] + [h * heads for h in hidden_sizes[:-1]]
        self.convs = nn.ModuleList(
            [GATConv(in_size, h, heads, dropout=dropout_p) for in_size, h in zip(in_sizes, hidden_sizes)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(h * heads) for h in hidden_sizes]
        )
        self.out = GATConv(hidden_sizes[-1] * heads, 1, dropout=dropout_p)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.out(x, edge_index)

        return torch.sigmoid(x)


def main():
    args = argument_parser()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["EXPERIMENT_NAME"])

    logger.info("Loading dataset")
    labels, features, homogenous = load_dataset(args.matrix, args.adjlist)
    _, _, _, _, idxtrain, idxtest = split_dataset(features, labels)
    data = graph_dataset(homogenous, features, labels, idxtrain, idxtest)
    
    # Convert data to PyTorch tensors
    transform = NormalizeFeatures()
    transform(data)
    batch_size = args.batch_size
    loader = NeighborLoader(
        data.cpu(),
        # Sample 170 neighbors for each node for 2 iterations
        num_neighbors=[170]*3,
        # Use a batch size for sampling training nodes
        batch_size=batch_size,
        input_nodes=data.train_mask,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: training on %s", device)
    model = GAT(features.shape[1], hidden_sizes=args.hidden_sizes, heads=args.heads)
    model.to(device)

    lr = args.lr
    epochs = args.epochs
    best_loss = float("inf")
    patience = args.patience
    counter = 0
    thres = args.thres

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()

    with mlflow.start_run():
        mlflow.log_params({
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience, 
            "lr": lr, 
            "threshold": thres, 
            "hidden_sizes": args.hidden_sizes, 
            "heads": args.heads})

        logger.info("Starting training")
        for epoch in range(epochs):
            epoch_loss = 0.0
            for sampled_data in loader:
                sampled_data.to(device)
                model.train()
                optimizer.zero_grad()
                out = model(sampled_data)
                loss = criterion(out[sampled_data.train_mask], sampled_data.y[sampled_data.train_mask].reshape(-1, 1).float())
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()

            # Accumulate predictions across all batches before computing metrics
            epoch_vloss = 0.0
            all_true = []
            all_proba = []
            with torch.no_grad():
                for sampled_data in loader:
                    sampled_data.to(device)
                    model.eval()
                    ypred = model(sampled_data)

                    mask = sampled_data.test_mask
                    if mask.sum() == 0:
                        continue

                    vloss = criterion(ypred[mask], sampled_data.y[mask].reshape(-1, 1).float())
                    epoch_vloss += vloss.item()

                    # model output already has sigmoid applied — no need to re-apply
                    all_true.append(sampled_data.y[mask].cpu().numpy())
                    all_proba.append(ypred[mask].cpu().numpy())

            all_true = np.concatenate(all_true)
            all_proba = np.concatenate(all_proba)
            all_binary = (all_proba > thres).astype(int)

            roc_auc = roc_auc_score(all_true, all_proba)
            f1 = f1_score(all_true, all_binary)
            precision = precision_score(all_true, all_binary)
            recall = recall_score(all_true, all_binary)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Validation Loss: {epoch_vloss:.4f} | "
                f"ROC AUC: {roc_auc:.3f} | "
                f"F1-score: {f1:.3f} | "
                f"Precision: {precision:.3f} | "
                f"Recall: {recall:.3f}"
            )
            mlflow.log_metrics(
                {
                    "train_loss": epoch_loss,
                    "val_loss": epoch_vloss,
                    "roc_auc": roc_auc,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                },
                step=epoch,
            )

            # Early stopping
            if epoch_vloss < best_loss:
                best_loss = epoch_vloss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break


def argument_parser():
    parser = argparse.ArgumentParser(description="Graph Attention Network Classifier")
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
        "--batch-size", type=int, default=256, help="batch size for training"
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
        default=[32],
        help="sizes of hidden layers, e.g. --hidden-sizes 64 32",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate for the training",
    )
    parser.add_argument(
        "--heads", type=int, default=1, help="number of attention heads per GAT layer"
    )
    parser.add_argument(
        "--thres", type=float, default=0.5, help="classification threshold"
    )
    args = parser.parse_args()
    return args


if __name__=="__main__":
    main()

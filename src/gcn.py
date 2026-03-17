import os
import torch
import mlflow
import argparse
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from dataset import load_dataset, split_dataset, graph_dataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_sizes: list[int], seed=2022):
        super().__init__()
        torch.manual_seed(seed)
        sizes = [num_features] + hidden_sizes
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_size, out_size) for in_size, out_size in zip(sizes, sizes[1:])]
        )
        self.out = GCNConv(sizes[-1], 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.out(x, edge_index)

        return torch.sigmoid(x)


def main():
    args = argument_parser()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["EXPERIMENT_NAME"])

    # Load dataset
    labels, features, homogenous = load_dataset(args.matrix, args.adjlist)

    # Split dataset into training and testing sets
    _, _, _, _, idxtrain, idxtest = split_dataset(features, labels)
    data = graph_dataset(homogenous, features, labels, idxtrain, idxtest)

    # Convert data to PyTorch tensors
    transform = NormalizeFeatures()
    transform(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = GCN(features.shape[1], args.hidden_sizes)
    model.to(device)
    data_gpu = data.to(device)

    lr = args.lr
    epochs = args.epochs
    best_loss = float("inf")
    patience = args.patience
    counter = 0
    thres = args.thres

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()

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
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data_gpu)
            #for discussion on why masks is applied here, see: https://stackoverflow.com/questions/69019682/training-mask-not-used-in-pytorch-geometric-when-inputting-data-to-train-model
            #loss
            loss = criterion(out[data_gpu.train_mask], data.y[data_gpu.train_mask].reshape(-1,1).float())
            loss.backward()
            optimizer.step()

            # Evaluation
            with torch.no_grad():
                model.eval()
                ypred = model(data_gpu)
                vloss = criterion(ypred[data_gpu.test_mask], data.y[data_gpu.test_mask].reshape(-1,1).float())

                true = data.y[data_gpu.test_mask].clone().cpu().detach().numpy()
                pred = ypred[data_gpu.test_mask].clone().cpu().detach()
                test_prob = torch.sigmoid(pred)
                ypred_proba = test_prob.numpy()
                ypred_binary = (ypred_proba > thres).astype(int)

                roc_auc = roc_auc_score(true, ypred_proba)
                f1 = f1_score(true, ypred_binary)
                precision = precision_score(true, ypred_binary)
                recall = recall_score(true, ypred_binary)

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
    parser = argparse.ArgumentParser(description="Graph Convolutional Network Classifier")
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
    args = parser.parse_args()
    return args


if __name__=="__main__":
    main()

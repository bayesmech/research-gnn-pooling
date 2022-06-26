import torch
import torch_geometric as pyg

from stochpool.engine.trainer import train_graph_classification_inductive
from stochpool.models.mincut_pool import MinCutPooledConvolutionalNetwork
from stochpool.models.diff_pool import DiffPooledConvolutionalNetwork


def main(model: str, dataset: str, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == "proteins":
        df = pyg.datasets.TUDataset(
            "datasets/proteins",
            name="PROTEINS",
        )
    else:
        raise NotImplementedError(
            "This dataset has been been added to the CurvGN pipeline."
        )

    if model == "diffpool":
        model = DiffPooledConvolutionalNetwork(df.num_features, df.num_classes).to(device)
    elif model == "mincutpool":
        model = MinCutPooledConvolutionalNetwork(df.num_features, df.num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    train_graph_classification_inductive(
        df,
        model,
        optimizer,
        device,
        epochs=epochs,
    )

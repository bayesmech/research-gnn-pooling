import torch
import torch_geometric as pyg

from stochpool.engine.trainer import train_graph_classification_inductive
from stochpool.models.mincut_pool import MinCutPooledConvolutionalNetwork
from stochpool.models.diff_pool import DiffPooledConvolutionalNetwork
from stochpool.analyzers.wandb import WandBLogger


def main(model: str, dataset: str, epochs: int, use_wandb: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == "proteins":
        df = pyg.datasets.TUDataset(
            "datasets/tu_datasets",
            name="PROTEINS",
        )
    elif dataset == "frankenstein":
        df = pyg.datasets.TUDataset(
            "datasets/tu_datasets",
            name="FRANKENSTEIN",
        )
    elif dataset == "nci1":
        df = pyg.datasets.TUDataset(
            "datasets/tu_datasets",
            name="NCI1",
        )
    elif dataset == "nci109":
        df = pyg.datasets.TUDataset(
            "datasets/tu_datasets",
            name="NCI109",
        )
    elif dataset == "enzymes":
        df = pyg.datasets.TUDataset(
            "datasets/tu_datasets",
            name="ENZYMES",
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

    analyzer = WandBLogger(activated=use_wandb)

    train_graph_classification_inductive(
        data=df,
        model=model,
        optimizer=optimizer,
        device=device,
        analyzer=analyzer,
        epochs=epochs,
    )

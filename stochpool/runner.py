import argparse

import torch, torch.utils.data
import torch_geometric as pyg

from stochpool.engine.trainer import train_graph_classification_inductive
from stochpool.analyzers.wandb import WandBLogger

from stochpool.models.stoch_pool import StochPooledConvolutionalNetwork
from stochpool.models.mincut_pool import MinCutPooledConvolutionalNetwork
from stochpool.models.diff_pool import DiffPooledConvolutionalNetwork
from stochpool.models.asap_pool import ASAPooledConvolutionalNetwork


def main(args: argparse.Namespace, use_wandb: bool):
    device = torch.device("cuda")

    if args.dataset == "proteins":
        df = pyg.datasets.TUDataset(
            "datasets/tu_datasets",
            name="PROTEINS",
        )
    elif args.dataset == "frankenstein":
        df = pyg.datasets.TUDataset(
            "datasets/tu_datasets",
            name="FRANKENSTEIN",
        )
    elif args.dataset == "nci1":
        df = pyg.datasets.TUDataset(
            "datasets/tu_datasets",
            name="NCI1",
        )
    elif args.dataset == "nci109":
        df = pyg.datasets.TUDataset(
            "datasets/tu_datasets",
            name="NCI109",
        )
    elif args.dataset == "enzymes":
        df = pyg.datasets.TUDataset(
            "datasets/tu_datasets",
            name="ENZYMES",
        )
    else:
        raise NotImplementedError(
            "This dataset has been been added to the CurvGN pipeline."
        )

    if args.model == "diffpool":
        model = DiffPooledConvolutionalNetwork(
            in_channels=df.num_features,
            out_channels=df.num_classes,
            conv_channels=(
                8,
                8,
                16,
                16,
                32,
            ),
            n_clusters=(
                20,
                16,
                12,
                8,
                5,
            ),
            pool_after=2,
        ).to(device)
    elif args.model == "mincutpool":
        model = MinCutPooledConvolutionalNetwork(df.num_features, df.num_classes).to(
            device
        )
    elif args.model == "stochpool":
        model = StochPooledConvolutionalNetwork(
            in_channels=df.num_features,
            out_channels=df.num_classes,
            conv_channels=(
                8,
                8,
                16,
                16,
                32,
            ),
            n_clusters=(
                20,
                16,
                12,
                8,
                5,
            ),
            pool_after=2,
        ).to(device)
    elif args.model == "asapool":
        model = ASAPooledConvolutionalNetwork(
            in_channels=df.num_features,
            out_channels=df.num_classes,
            conv_channels=(
                8,
                8,
                16,
                16,
                32,
            ),
            n_clusters=(
                20,
                16,
                12,
                8,
                5,
            ),
            pool_after=2,
        ).to(device)
    else:
        raise NotImplementedError("This model is not supported.")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    analyzer = WandBLogger(activated=use_wandb)

    test_dataset_size = int(0.25 * len(df))
    train_dataset_size = len(df) - test_dataset_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        df,
        [train_dataset_size, test_dataset_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = pyg.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
    )
    test_loader = pyg.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        num_workers=16,
    )

    train_graph_classification_inductive(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        analyzer=analyzer,
        epochs=args.epochs,
        per_batch_iters=args.per_batch_iters,
        accumulate_grad_batches=args.accumulate_grad_batches,
        name=f"gnn-{args.model}-{args.dataset}",
    )

import argparse

import wandb

from stochpool.runner import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="diffpool",
        help="which exact model to use (diffpool, mincutpool, etc.)",
    )
    parser.add_argument(
        "--dataset",
        default="proteins",
        help="which dataset to use (proteins, enzymes, nci1, nci109, etc.)"
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="number of epochs to train for"
    )
    parser.add_argument(
        "--wandb",
        action="store_const",
        default=False,
        const=True,
        help="should wandb be used for logging the current run?"
    )
    args = parser.parse_args()

    if args.wandb:
        wandb.init(
            project="gnn-stochpool",
            name="gnn-stochpool-run",
            save_code=False,
            config=dict(
                dataset=args.dataset,
                model=args.model,
                epochs=args.epochs,
            ),
            resume=False,
        )
    main(args.model, args.dataset, args.epochs, use_wandb=args.wandb)

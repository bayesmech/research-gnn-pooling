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
        help="which dataset to use (proteins, enzymes, nci1, nci109, etc.)",
    )
    parser.add_argument(
        "--epochs", default=200, type=int, help="number of epochs to train for"
    )
    parser.add_argument(
        "--wandb",
        action="store_const",
        default=False,
        const=True,
        help="should wandb be used for logging the current run?",
    )
    parser.add_argument(
        "--per-batch-iters",
        default=1,
        type=int,
        help="Number of iterations for which to accumulate gradient of a single batch"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help="Number of batches to accumulate gradients over",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed for everything including datasets shuffle"
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
    main(args, use_wandb=args.wandb)

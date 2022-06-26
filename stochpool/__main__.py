import argparse

from stochpool.runner import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="diffpool",
        help="which exact model to use (diffpool, mincutpool, etc.)",
    )
    parser.add_argument(
        "--dataset", default="proteins", help="which dataset to use (sbm, cora)"
    )
    parser.add_argument(
        "--epochs", default=200, type=int, help="number of epochs to train for"
    )
    args = parser.parse_args()

    main(args.model, args.dataset, args.epochs)

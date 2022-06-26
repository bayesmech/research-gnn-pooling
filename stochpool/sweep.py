import wandb

from stochpool.runner import main


if __name__ == "__main__":
    wandb.init(
        project="gnn-stochpool",
        name="gnn-stochpool-sweep-1",
        save_code=False,
        resume=False,
    )
    main(
        model=wandb.config.model,
        dataset=wandb.config.dataset,
        epochs=wandb.config.epochs,
        use_wandb=True,
    )

import wandb


class WandBLogger:

    def __init__(self, activated=True):
        self.activated = activated

    def log(self, data: dict):
        if self.activated:
            wandb.log(data)

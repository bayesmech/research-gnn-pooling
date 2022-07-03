import typing

import tqdm.auto as tqdm
import torch_geometric as pyg
import torch, torch.nn.functional

from stochpool.analyzers.wandb import WandBLogger


def train_graph_classification_inductive(
    data: pyg.data.Dataset,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    analyzer: WandBLogger,
):
    train_loader = pyg.loader.DataLoader(data, batch_size=20)
    test_loader = pyg.loader.DataLoader(data[int(len(data) * 0.9) :], batch_size=20)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        correct = 0
        loss_all = 0
        examples = 0

        # noinspection PyTypeChecker
        with tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}") as iterator:
            for data in iterator:
                data = data.to(device)
                optimizer.zero_grad()
                pred, additional_loss = model(
                    data.x,
                    data.edge_index,
                    data.batch,
                    data.ptr,
                )
                loss = (
                    torch.nn.functional.nll_loss(pred, data.y.view(-1))
                    + additional_loss
                )
                print(loss)
                loss.backward()
                loss_all += data.y.size(0) * float(loss)
                optimizer.step()
                correct += int(pred.max(dim=1)[1].eq(data.y.view(-1)).sum())
                examples += data.y.size(0)
                iterator.set_postfix(
                    loss=loss_all / examples, accuracy=correct / examples
                )
        train_loss = loss_all / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        with torch.no_grad():
            model.eval()
            correct, loss_all, examples = 0, 0, 0
            # noinspection PyTypeChecker
            with tqdm.tqdm(test_loader, desc=f"Testing Epoch {epoch}") as iterator:
                for data in iterator:
                    data = data.to(device)
                    pred, additional_loss = model(
                        data.x,
                        data.edge_index,
                        data.batch,
                        data.ptr,
                    )
                    loss = (
                        torch.nn.functional.nll_loss(pred, data.y.view(-1))
                        + additional_loss
                    )
                    loss_all += data.y.size(0) * float(loss)
                    correct += int(pred.max(dim=1)[1].eq(data.y.view(-1)).sum())
                    examples += data.y.size(0)
                    iterator.set_postfix(
                        loss=loss_all / examples, accuracy=correct / examples
                    )

        val_loss = loss_all / len(test_loader.dataset)
        val_acc = correct / len(test_loader.dataset)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        analyzer.log(
            {
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )

    return history

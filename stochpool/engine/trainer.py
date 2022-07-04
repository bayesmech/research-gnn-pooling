import typing

import tqdm.auto as tqdm
import torch_geometric as pyg
import torch, torch.nn.functional

from stochpool.analyzers.wandb import WandBLogger
from stochpool.utils.train_utils import AverageMeter


def train_graph_classification_inductive(
    train_loader: pyg.data.DataLoader,
    test_loader: pyg.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    analyzer: WandBLogger,
    per_batch_iters: int,
    accumulate_grad_batches: int,
    seed: int,
):

    optimizer.zero_grad()
    global_step = 0
    for epoch in range(epochs):
        model.train()

        accuracy_all = AverageMeter()
        loss_all = AverageMeter()
        classification_loss_all = AverageMeter()
        additional_loss_all = AverageMeter()

        # noinspection PyTypeChecker
        with tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}") as iterator:
            for data in iterator:
                # For accumulate gradient batches
                global_step += 1

                data = data.to(device, non_blocking=True)

                iter_loss = AverageMeter()
                iter_pred = AverageMeter()

                iter_classification_loss = AverageMeter()
                iter_additional_loss = AverageMeter()

                # To better estimate the gradients for a single batch
                for _ in range(per_batch_iters):
                    pred, additional_loss = model(
                        data.x,
                        data.edge_index,
                        data.batch,
                        data.ptr,
                    )
                    classification_loss = torch.nn.functional.nll_loss(
                        pred, data.y.view(-1)
                    )
                    loss = classification_loss + additional_loss
                    loss.backward()

                    iter_loss.update(loss.item(), n=1)
                    iter_pred.update(pred.detach(), n=1)

                    iter_classification_loss.update(classification_loss.item(), n=1)
                    iter_additional_loss.update(additional_loss.item(), n=1)

                loss_all.update(iter_loss.avg, n=data.y.size(0))
                acc = (
                    (iter_pred.avg.argmax(dim=1) == data.y.view(-1))
                    .float()
                    .mean()
                    .item()
                )
                accuracy_all.update(acc, n=data.y.size(0))

                classification_loss_all.update(
                    iter_classification_loss.avg, n=data.y.size(0)
                )
                additional_loss_all.update(iter_additional_loss.avg, n=data.y.size(0))

                if global_step % accumulate_grad_batches == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                iterator.set_postfix(loss=loss_all.avg, accuracy=accuracy_all.avg)

        train_loss = loss_all.avg
        train_acc = accuracy_all.avg

        val_accuracy_all = AverageMeter()
        val_loss_all = AverageMeter()
        val_classification_loss_all = AverageMeter()
        val_additional_loss_all = AverageMeter()

        with torch.no_grad():
            model.eval()
            # noinspection PyTypeChecker
            with tqdm.tqdm(test_loader, desc=f"Testing Epoch {epoch}") as iterator:
                for data in iterator:
                    data = data.to(device, non_blocking=True)

                    iter_loss = AverageMeter()
                    iter_pred = AverageMeter()

                    iter_classification_loss = AverageMeter()
                    iter_additional_loss = AverageMeter()

                    # Running to get a better estimate of the predictions
                    for _ in range(per_batch_iters):
                        pred, additional_loss = model(
                            data.x,
                            data.edge_index,
                            data.batch,
                            data.ptr,
                        )
                        classification_loss = torch.nn.functional.nll_loss(
                            pred, data.y.view(-1)
                        )
                        loss = classification_loss + additional_loss

                        iter_loss.update(loss.item(), n=1)
                        iter_pred.update(pred.detach(), n=1)

                        iter_additional_loss.update(additional_loss.item(), n=1)
                        iter_classification_loss.update(classification_loss.item(), n=1)

                    val_loss_all.update(iter_loss.avg, n=data.y.size(0))
                    acc = (
                        (iter_pred.avg.argmax(dim=1) == data.y.view(-1))
                        .float()
                        .mean()
                        .item()
                    )
                    val_accuracy_all.update(acc, n=data.y.size(0))

                    val_additional_loss_all.update(
                        iter_additional_loss.avg, n=data.y.size(0)
                    )
                    val_classification_loss_all.update(
                        iter_classification_loss.avg, n=data.y.size(0)
                    )

                    # correct += int(pred.max(dim=1)[1].eq(data.y.view(-1)).sum())
                    # examples += data.y.size(0)

                    iterator.set_postfix(
                        loss=val_loss_all.avg, accuracy=val_accuracy_all.avg
                    )

        val_loss = val_loss_all.avg
        val_acc = val_accuracy_all.avg

        analyzer.log(
            {
                "train_acc": train_acc,
                "train_loss": train_loss,
                "train_classification_loss": classification_loss_all.avg,
                "train_additional_loss": additional_loss_all.avg,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "val_classification_loss": val_classification_loss_all.avg,
                "val_additional_loss": val_additional_loss_all.avg,
            }
        )

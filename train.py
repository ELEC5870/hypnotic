#!/usr/bin/env python3

from contextlib import nullcontext
import itertools
import os
import random

import git
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from dataset import ParquetRDDataset
from model import LaudeAndOstermannPlusScalars


def seed_prngs(random_seed):
    if not random_seed:
        random.seed(0)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        generator.manual_seed(0)
        np.random.seed(0)


def image_transform(pu):
    pu = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )(pu)
    pu -= pu.mean()
    std = pu.std()
    if std != 0:
        pu /= std
    return pu


def dataloaders():
    filter = (pc.field("w") == args.width) & (pc.field("h") == args.height)
    training_dataset = ParquetRDDataset(
        args.image_path,
        args.parquet_path,
        filter=filter,
        transform=image_transform,
        target_transform=target_transform,
        deterministic=not args.random_seed,
    )
    testing_size = len(training_dataset) // 4
    training_dataset.limit = len(training_dataset) - testing_size
    testing_dataset = ParquetRDDataset(
        args.image_path,
        args.parquet_path,
        filter=filter,
        transform=image_transform,
        offset=training_dataset.limit,
        deterministic=not args.random_seed,
    )

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=None,
        batch_sampler=None,
        generator=generator,
    )

    testing_dataloader = DataLoader(
        testing_dataset,
        batch_size=None,
        batch_sampler=None,
        generator=generator,
    )

    return (training_dataloader, testing_dataloader)


def tb_write_meta():
    # log hyperparameters
    writer.add_hparams(
        {
            "image_path": args.image_path,
            "parquet_path": args.parquet_path,
            "dataset_size": len(training_dataloader),
            "learning_rate": args.learning_rate,
            "loss_fn": args.loss_function,
        },
        {},
        # @TODO: Fix this - at the moment it seems to break all other monitoring.
        run_name=".",  # Don't append an extra timestamp to the run name
    )

    # log git revision
    repo = git.Repo(search_parent_directories=True)
    diff = repo.git.diff(repo.head.object.hexsha)
    writer.add_text("git", f"`{repo.head.object.hexsha}`\n```\n{diff}\n```")

    # dataset info
    (x_image, x_scalars), y = next(iter(training_dataloader))
    log(f"{len(training_dataloader)=}")
    log(f"{x_image.shape=} {x_image.dtype=}")
    log(f"{x_scalars.shape=} {x_scalars.dtype=}")
    log(f"{y.shape=} {y.dtype=}")
    log(f"{args.random_seed=}")
    img_grid = torchvision.utils.make_grid(x_image)
    writer.add_image("input sample", img_grid)
    writer.flush()


def train(dataloader, model, loss_fn, optimizer, epoch, profile=False):
    model.train()

    log()
    num_batches = len(dataloader.dataset)
    data = tqdm(dataloader, desc=f"training epoch {epoch}", disable=args.quiet)
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(run_dir),
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    )
    with profiler if profile else nullcontext() as profiler:
        for i, ((x_image, x_scalars), y) in enumerate(data):
            if y.shape[0] == 0:
                continue

            x_image = x_image.to(device, non_blocking=True)
            x_scalars = x_scalars.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # compute loss
            pred = model(x_image, x_scalars)
            loss = loss_fn(pred, y)

            # backpropagation
            loss.backward()
            optimizer.step()

            # monitoring
            writer.add_scalar("training_loss", loss, epoch * num_batches + i)
            if profiler is not None:
                profiler.step()


def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(include_values=False)
    labels = ["" for _ in range(67)]
    labels[0] = "0"
    labels[18] = "18"
    labels[50] = "50"
    plt.gca().set_xticklabels(labels)
    plt.gca().set_yticklabels(labels)
    return plt.gcf()


def test(dataloader, target_transform, model, loss_fn, epoch):
    model.eval()

    num_samples = 0
    num_batches = len(dataloader)
    data = tqdm(dataloader, desc=f"testing epoch {epoch}", disable=args.quiet)
    test_losses, correct, rd_costs = [], 0, []
    cm = np.zeros((67, 67), dtype=np.int64)

    with torch.no_grad():
        for (x_image, x_scalars), y in data:
            if y.shape[0] == 0:
                continue

            x_image = x_image.to(device, non_blocking=True)
            x_scalars = x_scalars.to(device, non_blocking=True)
            if target_transform is not None:
                tx_y = target_transform(y).to(device, non_blocking=True)
            else:
                tx_y = y.to(device, non_blocking=True)
            num_samples += x_image.shape[0]

            pred = model(x_image, x_scalars)

            # update metrics
            loss = loss_fn(pred, tx_y)
            test_losses.append(loss)
            correct += correct_fn(pred, tx_y)
            cm += confusion_matrix(
                sel_fn(tx_y).cpu(), sel_fn(pred).cpu(), labels=range(67)
            )
            rd_costs.append(rd_cost_fn(pred, y))

    # @NOTE: This is a mean of means, which is a bit wrong
    mean_test_loss = sum(test_losses) / num_batches
    writer.add_scalar("testing_loss", mean_test_loss, epoch)
    writer.add_histogram(
        "testing_loss_distribution",
        torch.stack(test_losses),
        epoch,
    )

    correct /= num_samples
    writer.add_scalar("accuracy", correct, epoch)

    cm = plot_confusion_matrix(cm)
    writer.add_figure("confusion_matrix", cm, epoch)

    mean_rd_cost = torch.cat(rd_costs).mean()
    writer.add_scalar("rd_cost", mean_rd_cost, epoch)

    log(
        f"loss: {mean_test_loss:.4f}, accuracy: {100 * correct:.2f}%, RD cost vs optimal: {100 * mean_rd_cost:.2f}%"
    )


if __name__ == "__main__":
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("parquet_path", type=str)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data-loader-workers", type=int, default=2)
    parser.add_argument("--random-seed", action="store_true")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--loss-function", default="mse", choices=["mse", "crossentropy"])
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--profile", "-p", action="store_true")
    args = parser.parse_args()
    # fmt: on

    def log(*a, **kw):
        if not args.quiet:
            print(*a, **kw)

    # set up pytorch
    log(f"pytorch {torch.__version__}")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    log(f"using {device} device")

    # set up random number generators
    generator = torch.Generator()
    seed_prngs(args.random_seed)

    # set up tensorboard
    writer = SummaryWriter()
    run_dir = writer.log_dir
    log(f"writing logs to {run_dir}")

    # define evaluation functions
    if args.loss_function == "mse":
        target_transform = torch.log
        loss_fn = nn.MSELoss()
        sel_fn = lambda pred: pred.argmin(1)
    elif args.loss_function == "crossentropy":
        target_transform = lambda y: y.argmin(1)
        loss_fn = nn.CrossEntropyLoss()
        sel_fn = lambda pred: pred.argmax(1)
    rd_cost_fn = (
        lambda pred, y: (y[torch.arange(len(y)), sel_fn(pred).cpu()] - y.min(1).values)
        / y.min(1).values
    )
    correct_fn = lambda pred, y: (sel_fn(pred) == sel_fn(y)).float().sum().item()

    # load dataset
    training_dataloader, testing_dataloader = dataloaders()
    (x_image_example, x_scalars_example), y_example = next(iter(training_dataloader))

    # define model
    model = LaudeAndOstermannPlusScalars(
        pu_size=(x_image_example.shape[2], x_image_example.shape[3]),
        num_scalars=x_scalars_example.shape[1],
    )
    model = model.to(device)

    # training hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        log(f"resuming from epoch {start_epoch}")

    tb_write_meta()

    # train and test loop
    epochs = range(args.epochs) if args.epochs > 0 else itertools.count(1)
    for t in epochs:
        train(training_dataloader, model, loss_fn, optimizer, t, args.profile)
        test(testing_dataloader, target_transform, model, loss_fn, t)

        torch.save(
            {
                "epoch": t,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(run_dir, f"checkpoint.pt"),
        )
        writer.flush()

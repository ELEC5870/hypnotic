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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("parquet_path", type=str)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data-loader-workers", type=int, default=2)
    parser.add_argument("--random-seed", action="store_true")
    parser.add_argument("--resume", type=str)
    parser.add_argument(
        "--loss-function", default="mse", choices=["mse", "crossentropy"]
    )
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--profile", "-p", action="store_true")
    args = parser.parse_args()

    def log(msg):
        if not args.quiet:
            print(msg)

    log(f"pytorch {torch.__version__}")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    log(f"using {device} device")

    generator = torch.Generator()

    if not args.random_seed:
        print("disabling random seed")
        random.seed(0)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        generator.manual_seed(0)
        np.random.seed(0)

    writer = SummaryWriter()
    run_dir = writer.log_dir
    log(f"writing logs to {run_dir}")

    if args.loss_function == "mse":
        target_transform = torch.log
        loss_fn = nn.MSELoss()
        sel_fn = lambda pred: pred.argmin(1)
    elif args.loss_function == "crossentropy":
        target_transform = lambda y: y.argmin(1)
        loss_fn = nn.CrossEntropyLoss()
        sel_fn = lambda pred: pred.argmax(1)

    # rd_cost_fn = (
    #     lambda pred, y: y[torch.arange(len(y)), sel_fn(pred)]
    #     - y[torch.arange(len(y)), sel_fn(y)]
    # )
    correct_fn = lambda pred, y: (sel_fn(pred) == sel_fn(y)).float().sum().item()

    # load data
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    # @TODO: Split into training and testing datasets
    filter = (pc.field("w") == 16) & (pc.field("h") == 16)
    dataset = ParquetRDDataset(
        args.image_path,
        args.parquet_path,
        filter=filter,
        transform=transform,
        target_transform=target_transform,
        deterministic=not args.random_seed,
    )
    # [testing_dataset, training_dataset] = torch.utils.data.random_split(
    #     dataset,
    #     [0.25, 0.75],
    # )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        generator=generator,
    )

    (x_image, x_scalars), y = next(iter(dataloader))
    print(f"{len(dataset)=}")
    print(f"{x_image.shape=} {x_image.dtype=}")
    print(f"{x_scalars.shape=} {x_scalars.dtype=}")
    print(f"{y.shape=} {y.dtype=}")
    img_grid = torchvision.utils.make_grid(x_image)
    writer.add_image("input sample", img_grid)
    writer.flush()

    # define model
    model = LaudeAndOstermannPlusScalars(
        pu_size=(x_image.shape[2], x_image.shape[3]), num_scalars=x_scalars.shape[1]
    )
    model = model.to(device)

    # training hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        testing_loss = checkpoint["testing_loss"]
        log(f"resuming from epoch {start_epoch}, testing loss {testing_loss}")

    # log hyperparameters
    writer.add_hparams(
        {
            "image_path": args.image_path,
            "parquet_path": args.parquet_path,
            "dataset_size": len(dataset),
            "learning_rate": args.learning_rate,
            "loss_fn": args.loss_function,
        },
        {},
        # @TODO: Fix this - at the moment it seems to break all other monitoring.
        run_name=".",  # Don't append an extra timestamp to the run name
    )
    writer.flush()

    # log git revision
    repo = git.Repo(search_parent_directories=True)
    diff = repo.git.diff(repo.head.object.hexsha)
    writer.add_text("git", f"`{repo.head.object.hexsha}`\n```\n{diff}\n```")
    writer.flush()
    del repo

    def train(dataloader, model, loss_fn, optimizer, epoch, profile=False):
        print()
        size = len(dataloader.dataset)
        model.train()
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
                writer.add_scalar("training_loss", loss, epoch * size + i)
                if profiler is not None:
                    profiler.step()

    def test(dataloader, model, loss_fn, epoch):
        size = 0
        num_batches = len(dataloader)
        model.eval()
        data = tqdm(dataloader, desc=f"testing epoch {epoch}", disable=args.quiet)
        test_losses, correct = [], 0
        cm = np.zeros((67, 67), dtype=np.int64)
        with torch.no_grad():
            for (x_image, x_scalars), y in data:
                x_image = x_image.to(device, non_blocking=True)
                x_scalars = x_scalars.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                pred = model(x_image, x_scalars)

                loss = loss_fn(pred, y)
                test_losses.append(loss)
                correct += correct_fn(pred, y)
                size += x_image.shape[0]

                cm += confusion_matrix(
                    sel_fn(y).cpu(), sel_fn(pred).cpu(), labels=range(67)
                )
        correct /= size
        mean_test_loss = sum(test_losses) / size
        writer.add_scalar("testing_loss", mean_test_loss, epoch)
        writer.add_histogram(
            "testing_loss_distribution",
            torch.stack(test_losses),
            epoch,
        )
        writer.add_scalar("accuracy", correct, epoch)
        log(f"loss: {mean_test_loss:.4f}, accuracy: {100 * correct:.2f}%")
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(include_values=False)
        labels = ["" for _ in range(67)]
        labels[0] = "0"
        labels[18] = "18"
        labels[50] = "50"
        plt.gca().set_xticklabels(labels)
        plt.gca().set_yticklabels(labels)
        writer.add_figure("confusion_matrix", plt.gcf(), epoch)
        return mean_test_loss

    if args.epochs > 0:
        epochs = range(args.epochs)
    else:
        epochs = itertools.count(1)

    for t in epochs:
        train(dataloader, model, loss_fn, optimizer, t, args.profile)
        loss = test(dataloader, model, loss_fn, t)
        torch.save(
            {
                "epoch": t,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "testing_loss": loss,
            },
            os.path.join(run_dir, f"checkpoint.pt"),
        )
        writer.flush()

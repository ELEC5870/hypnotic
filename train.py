#!/usr/bin/env python3

from contextlib import nullcontext
import itertools
import os
import random

import git
import numpy as np
import pyarrow.compute as pc
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
from torchvision.transforms import v2 as transforms
from torchvision.models import resnet50
from tqdm import tqdm

from dataset import ParquetRDDataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("parquet_path", type=str)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data-loader-workers", type=int, default=2)
    parser.add_argument("--random-seed", action="store_true")
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

    # load data
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    dataset = ParquetRDDataset(
        args.image_path,
        args.parquet_path,
        filter=(pc.field("w") == 16) & (pc.field("h") == 16),
        transform=transform,
        target_transform=lambda y: y.argmin(1),
        deterministic=not args.random_seed,
    )
    print(f"{len(dataset)=}")
    training_dataset, testing_dataset = torch.utils.data.random_split(
        dataset,
        [0.75, 0.25],
    )
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=None,
        batch_sampler=None,
        generator=generator,
        shuffle=True,
        # num_workers=args.data_loader_workers,
        # pin_memory=device != "cpu",
    )
    testing_dataloader = DataLoader(
        testing_dataset,
        batch_size=None,
        batch_sampler=None,
        generator=generator,
        shuffle=True,
        # num_workers=args.data_loader_workers,
        # pin_memory=device != "cpu",
    )

    x, y = next(iter(training_dataloader))
    print(f"{x.shape=} {x.dtype=}")
    print(f"{y.shape=} {y.dtype=}")
    img_grid = torchvision.utils.make_grid(x)
    writer.add_image("input sample", img_grid)
    writer.flush()

    # define model
    model = resnet50(weights=None)
    model = model.to(device)

    # training hyperparameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # log hyperparameters
    writer.add_hparams(
        {
            "image_path": args.image_path,
            "parquet_path": args.parquet_path,
            "dataset_size": len(dataset),
            "learning_rate": args.learning_rate,
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
            for i, (x, y) in enumerate(data):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                # compute loss
                pred = model(x)
                loss = loss_fn(pred, y)

                # backpropagation
                loss.backward()
                optimizer.step()

                # monitoring
                writer.add_scalar("training_loss", loss, epoch * size + i)
                if profiler is not None:
                    profiler.step()

                # ensure memory is freed
                del pred, loss, x, y

    def test(dataloader, model, loss_fn, epoch):
        size = 0
        num_batches = len(dataloader)
        model.eval()
        data = tqdm(dataloader, desc=f"testing epoch {epoch}", disable=args.quiet)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in data:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                size += x.shape[0]
        test_loss /= num_batches
        correct /= size
        writer.add_scalar("testing_loss", test_loss, epoch)
        writer.add_scalar("accuracy", correct, epoch)
        log(f"loss: {test_loss:.4f}, accuracy: {100 * correct:.2f}%")
        return test_loss

    if args.epochs > 0:
        epochs = range(args.epochs)
    else:
        epochs = itertools.count(1)

    for t in epochs:
        train(training_dataloader, model, loss_fn, optimizer, t, args.profile)
        loss = test(testing_dataloader, model, loss_fn, t)
        torch.save(
            {
                "epoch": t,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "testing_loss": loss,
            },
            os.path.join(run_dir, f"checkpoint_{t}.pt"),
        )
        writer.flush()

#!/usr/bin/env python3

from contextlib import nullcontext
import copy
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
from model import Custom

NUM_MODES = 67
BATCH_SIZE = 32
VALIDATE = False
DOWNSAMPLING_FACTOR = 0.5


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


def optimal_mode_distribution(dataset):
    optimal_mode_frequencies = [0] * NUM_MODES
    for batch in dataset.to_batches():
        costs = np.stack(batch["cost"].to_numpy(zero_copy_only=False))
        optimal_modes = costs.argmin(1)
        for mode, count in zip(*np.unique(optimal_modes, return_counts=True)):
            optimal_mode_frequencies[mode] += count
    return optimal_mode_frequencies


def dataloaders():
    training_dataset = ParquetRDDataset(
        args.image_path,
        args.training_data_path,
        transform=image_transform,
        target_transform=target_transform,
        deterministic=not args.random_seed,
        batch_size=BATCH_SIZE,
    )
    training_dataloader = DataLoader(
        training_dataset,
        generator=generator,
        batch_size=None,
    )

    testing_dataset = ParquetRDDataset(
        args.image_path,
        args.testing_data_path,
        deterministic=not args.random_seed,
        batch_size=BATCH_SIZE,
    )
    testing_dataloader = DataLoader(
        testing_dataset,
        generator=generator,
        batch_size=None,
    )

    return (training_dataloader, testing_dataloader)


def image_grid(images):
    fig, axes = plt.subplots(BATCH_SIZE // 8, 8, figsize=(16, 16))
    for i, (ax, image) in enumerate(zip(axes.flat, images)):
        image = image[0]
        width, height = image.shape
        ax.set_title(str(i))
        ax.grid()
        ax.set_xticks(range(width))
        ax.set_xticklabels([])
        ax.set_yticks(range(height))
        ax.set_yticklabels([])
        ax.imshow(image, cmap="gray", aspect="equal", extent=(0, width, 0, height))
    return fig, axes


def tb_write_meta():
    # log hyperparameters
    writer.add_hparams(
        {
            "image_path": args.image_path,
            "training_data_path": args.training_data_path,
            "testing_data_path": args.testing_data_path,
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
    log(f"{x_image_example.shape=}")
    log(f"{x_scalars_example.shape=}")
    log(f"{y_example.shape=}")
    log(f"{args.random_seed=}")
    fig, axes = image_grid(x_image_example)
    for ax, scalars, costs in zip(axes.flat, x_scalars_example, y_example):
        optimal_mode = costs.argmin().item()
        lagrange = scalars[0]
        mpm = [int(scalars[i]) for i in range(6, 6 + 6)]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(
            (xlim[0] + xlim[1]) / 2,
            (ylim[0] + ylim[1]) / 2,
            str(optimal_mode),
            bbox={"boxstyle": "round", "facecolor": "white"},
            ha="center",
            va="center",
        )
        ax.text(
            (xlim[0] + xlim[1]) / 2,
            -1,
            f"λ={lagrange:.2f}\n{mpm}",
            ha="center",
            va="center",
        )
    writer.add_figure("input sample", fig)
    writer.flush()


def train(dataloader, model, loss_fn, optimizer, scheduler, epoch, profile=False):
    model.train()

    log()
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
            optimizer.zero_grad(set_to_none=True)

            x_image = x_image.to(device)
            x_scalars = x_scalars.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            # validate input
            if VALIDATE:
                if x_image.isnan().any() or x_image.isinf().any():
                    raise RuntimeError(f"x_image is {x_image}")
                if x_scalars.isnan().any() or x_scalars.isinf().any():
                    raise RuntimeError(f"x_scalars is {x_scalars}")
                if y.isnan().any() or y.isinf().any() or (y < 0).any():
                    raise RuntimeError(f"y is {y}")

            # compute loss
            pred = model(x_image, x_scalars)
            loss = loss_fn(pred, y)

            # backpropagation
            loss.backward()
            optimizer.step()

            # monitoring
            writer.add_scalar("training_loss", loss, batch_idx)
            if profiler is not None:
                profiler.step()
            batch_idx += 1


def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(include_values=False)
    disp.im_.set_clim(0, 1)
    labels = ["" for _ in range(67)]
    labels[0] = "0"
    labels[18] = "18"
    labels[50] = "50"
    plt.gca().set_xticklabels(labels)
    plt.gca().set_yticklabels(labels)
    return plt.gcf()


def test(
    dataloader, image_transform, target_transform, model, loss_fn, scheduler, epoch
):
    model.eval()

    num_samples = 0
    data = tqdm(dataloader, desc=f"testing epoch {epoch}", disable=args.quiet)
    test_losses, correct, rd_costs = [], 0, []
    cm = np.zeros((67, 67), dtype=np.int64)

    with torch.no_grad():
        for (x_image, x_scalars), y in data:
            x_image = x_image.to(device)
            if image_transform is not None:
                x_image = torch.stack([image_transform(xi) for xi in x_image]).to(
                    device
                )
            else:
                x_image = x_image.to(device)
            x_scalars = x_scalars.to(device)
            if target_transform is not None:
                tx_y = torch.stack([target_transform(yi) for yi in y]).to(device)
            else:
                tx_y = y.to(device)
            num_samples += x_image.shape[0]

            # validate input
            if VALIDATE:
                if x_image.isnan().any() or x_image.isinf().any():
                    raise RuntimeError(f"x_image is {x_image}")
                if x_scalars.isnan().any() or x_scalars.isinf().any():
                    raise RuntimeError(f"x_scalars is {x_scalars}")
                if y.isnan().any() or y.isinf().any() or (y < 0).any():
                    raise RuntimeError(f"y is {y}")

            pred = model(x_image, x_scalars)

            loss = loss_fn(pred, tx_y)
            if loss.isnan().any() or loss.isinf().any() or (loss < 0).any():
                raise RuntimeError(f"loss is {loss}")

            # update metrics
            test_losses.append(loss)
            correct += correct_fn(pred, tx_y)
            cm += confusion_matrix(
                sel_fn(tx_y).cpu(), sel_fn(pred).cpu(), labels=range(67)
            )
            rd_costs.append(rd_cost_fn(pred, y))

    # @NOTE: This is a mean of means, which is a bit wrong
    mean_test_loss = torch.stack(test_losses).mean()
    writer.add_scalar("testing_loss", mean_test_loss, epoch)
    writer.add_histogram(
        "testing_loss_distribution",
        torch.stack(test_losses),
        epoch,
    )
    plt.hist(
        (torch.stack(test_losses) / sum(test_losses)).cpu().numpy(),
        bins=100,
        cumulative=True,
    )
    writer.add_figure("testing_loss_distribution (cumulative)", plt.gcf(), epoch)

    correct /= num_samples
    writer.add_scalar("accuracy", correct, epoch)

    cm_sums = cm.sum(axis=1)[:, np.newaxis]
    cm = np.divide(cm.astype("float"), cm_sums, where=cm_sums >= 1)
    cm = plot_confusion_matrix(cm)
    writer.add_figure("confusion_matrix", cm, epoch)

    mean_rd_cost = torch.cat(rd_costs).mean()
    writer.add_scalar("rd_cost", mean_rd_cost, epoch)

    example_pred = model(
        image_transform(x_image_example).to(device),
        x_scalars_example.to(device),
    )
    example_loss = [
        loss_fn(p, target_transform(y).to(device))
        for p, y in zip(example_pred, y_example)
    ]
    example_rd_cost = rd_cost_fn(example_pred, y_example)

    fig, axes = image_grid(x_image_example)
    for ax, pred, loss, cost in zip(
        axes.flat, example_pred, example_loss, example_rd_cost
    ):
        xlim = ax.get_xlim()
        ax.text(
            (xlim[0] + xlim[1]) / 2,
            -1,
            f"pred: {int(sel_fn(pred[None, :]))}\nloss: {loss:.2f}\n+RD: {cost:.2%}",
            ha="center",
            va="center",
        )
    writer.add_figure("output example", fig, epoch)

    if scheduler is not None:
        scheduler.step(mean_test_loss)

    log(
        f"loss: {mean_test_loss:.4f}, accuracy: {100 * correct:.2f}%, RD cost vs optimal: {100 * mean_rd_cost:.2f}%"
    )


if __name__ == "__main__":
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("training_data_path", type=str)
    parser.add_argument("testing_data_path", type=str)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data-loader-workers", type=int, default=2)
    parser.add_argument("--random-seed", action="store_true")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--loss-function", default="crossentropy", choices=["mse", "crossentropy"])
    parser.add_argument("--temperature", type=float, default=10)
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
        target_transform = None
        training_loss_fn = nn.MSELoss()
        testing_loss_fn = copy.deepcopy(training_loss_fn)
        sel_fn = lambda pred: pred.argmin(1)
    elif args.loss_function == "crossentropy":
        target_transform = lambda y: (
            -(y - y.mean()) / y.std() * args.temperature
        ).softmax(0)
        training_loss_fn = nn.CrossEntropyLoss()
        testing_loss_fn = copy.deepcopy(training_loss_fn)
        sel_fn = lambda pred: pred.argmax(1)
    rd_cost_fn = (
        lambda pred, y: (y[torch.arange(len(y)), sel_fn(pred).cpu()] - y.min(1).values)
        / y.min(1).values
    )
    correct_fn = lambda pred, y: (sel_fn(pred) == sel_fn(y)).float().sum().item()

    # load dataset
    training_dataloader, testing_dataloader = dataloaders()

    # set up downsampling & upweighting
    mode_freqs = optimal_mode_distribution(training_dataloader.dataset.dataset)
    plt.bar(range(len(mode_freqs)), mode_freqs)
    plt.xlabel("optimal mode")
    plt.ylabel("frequency")
    writer.add_figure("mode_frequencies", plt.gcf())
    sampling_weights = [
        (min(mode_freqs) / freq) ** DOWNSAMPLING_FACTOR for freq in mode_freqs
    ]
    training_dataloader.dataset.mode_weights = sampling_weights
    training_loss_fn.weight = torch.tensor(
        [1 / weight for weight in sampling_weights]
    ).to(device)

    # Get example batch
    (x_image_example, x_scalars_example), y_example = next(iter(testing_dataloader))
    x_image_example, x_scalars_example, y_example = (
        x_image_example,
        x_scalars_example,
        y_example,
    )

    # define model
    model = Custom(num_scalars=x_scalars_example.shape[1]).to(device)
    model = torch.jit.trace(
        model,
        (image_transform(x_image_example).to(device), x_scalars_example.to(device)),
    )

    # training hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)

    start_epoch = 0
    if args.resume:
        model = torch.jit.load(os.path.join(args.resume, "model.pt"))
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint.pt"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        log(f"resuming from epoch {start_epoch}")

    tb_write_meta()

    # train and test loop
    epochs = (
        range(start_epoch, args.epochs)
        if args.epochs > 0
        else itertools.count(start_epoch)
    )
    batch_idx = 0
    for t in epochs:
        train(
            training_dataloader,
            model,
            training_loss_fn,
            optimizer,
            scheduler,
            t,
            args.profile,
        )
        test(
            testing_dataloader,
            image_transform,
            target_transform,
            model,
            testing_loss_fn,
            scheduler,
            t,
        )

        checkpoint_data = {
            "epoch": t,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(
            checkpoint_data,
            os.path.join(run_dir, f"checkpoint.pt"),
        )
        model.save(os.path.join(run_dir, f"model.pt"))
        writer.flush()

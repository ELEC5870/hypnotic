import random

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import tqdm

RESPONSES = ["cost", "dist", "fracBits"]


def prepare_batch(batch, predictors, reduction, pbar):
    table = pa.Table.from_batches([batch])
    rows = len(table)

    # @TODO: Check this is actually doing what I think this is doing - are the two lists
    #        associated with one other?
    table = table.group_by(predictors).aggregate(
        [(col, "list") for col in RESPONSES + ["intra_mode"]]
    )

    sample = random.sample(range(len(table)), int(len(table) * (1 - reduction)))
    table = table.take(sample)

    batches = table.to_batches()
    if len(batches) > 1:
        raise Exception("More than one batch")
    pbar.update(rows)
    return batches[0]


def prepare_dataset(dataset, output_path, reduction, quiet=True):
    columns = dataset.schema.names
    predictors = [c for c in columns if c not in RESPONSES + ["intra_mode"]]
    batch_size = 131072
    batches = dataset.to_batches(batch_size=batch_size)
    pbar = tqdm.tqdm(total=dataset.count_rows(), disable=quiet)

    schema = dataset.schema
    for c in RESPONSES:
        schema = schema.remove(schema.get_field_index(c))
        schema = schema.insert(
            0, pa.field(f"{c}_list", pa.list_(pa.float32()), nullable=False)
        )
    schema = schema.remove(schema.get_field_index("intra_mode"))
    schema = schema.insert(
        0, pa.field("intra_mode_list", pa.list_(pa.int32()), nullable=False)
    )

    ds.write_dataset(
        map(lambda b: prepare_batch(b, predictors, reduction, pbar), batches),
        output_path,
        format="parquet",
        schema=schema,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--reduction", type=float, default=0.9)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    partitioning = pa.dataset.partitioning(
        flavor="filename",
        schema=pa.schema([("sequence", pa.string())]),
    )
    dataset = ds.dataset(
        args.input,
        format="parquet",
        partitioning=partitioning,
        exclude_invalid_files=True,
    )

    prepare_dataset(dataset, args.output, args.reduction, args.quiet)

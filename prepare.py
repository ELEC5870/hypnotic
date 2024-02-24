from itertools import chain
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

RESPONSES = ["cost", "dist", "fracBits"]
META_SCHEMA = pa.schema(
    [
        ("sequence", pa.string()),
        ("qp", pa.uint8()),
    ]
)


def prepare_batch(batch, meta, reduction):
    table = pa.Table.from_batches([batch])

    for key, value in meta.items():
        field = META_SCHEMA.field(key)
        table = table.append_column(field, pa.array([value] * len(batch), field.type))

    sample = []
    for i in range(len(batch)):
        if random.random() >= reduction:
            sample.append(i)
    if not sample:
        return pa.RecordBatch.from_pylist([], schema=table.schema)
    table = table.take(sample)

    batches = table.to_batches()
    if len(batches) > 1:
        raise ValueError("Expected a single batch")
    batch = batches[0]
    return batch


def prepare_dataset(input_path, output_path, reduction, quiet=True):
    if os.path.isdir(input_path):
        filepaths = [
            os.path.join(dirpath, filename)
            for dirpath, _, filenames in os.walk(input_path)
            for filename in filenames
        ]
    else:
        filepaths = [input_path]

    def get_meta(filepath):
        meta = {}
        name = os.path.splitext(os.path.basename(filepath))[0]
        sequence, qp = name.rsplit("_", 1)
        meta["sequence"] = sequence
        meta["qp"] = int(qp)
        return meta

    schema = pq.read_schema(filepaths[0])
    for field in META_SCHEMA:
        schema = schema.append(field)
    ds.write_dataset(
        chain(
            *(
                map(
                    lambda b: prepare_batch(b, get_meta(filepath), reduction),
                    ds.dataset(filepath, format="parquet").to_batches(),
                )
                for filepath in filepaths
            )
        ),
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

    prepare_dataset(args.input, args.output, args.reduction, args.quiet)

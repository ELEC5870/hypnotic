#!/usr/bin/env python3

# Copyright (c) 2023 Frank Plowman
#
# This project is licensed under the terms of the MIT license. For details, see LICENSE

"""This utility converts the raw output of the RD search into a parquet dataset."""

import os
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

RD_LINE_PATTERN = re.compile(
    r"IntraCost T "
    r"x=(\d+),"
    r"y=(\d+),"
    r"w=(\d+),"
    r"h=(\d+),"
    r"cost=([\d.e+-]+),"
    r"dist=([\d.e+-]+),"
    r"fracBits=([\d.e+-]+),"
    r"lambda=([\d.e+-]+),"
    r"modeId=(\d+),"
    r"ispMod=([\d-]+),"
    r"multiRefIdx=(\d+),"
    r"mipFlag=(\d+),"
    r"lfnstIdx=(\d+),"
    r"mtsFlag=(\d+),"
    r"mpm_pred0=(\d+),"
    r"mpm_pred1=(\d+),"
    r"mpm_pred2=(\d+),"
    r"mpm_pred3=(\d+),"
    r"mpm_pred4=(\d+),"
    r"mpm_pred5=(\d+)"
)

RD_SCHEMA = pa.schema(
    [
        pa.field("x", pa.uint16()),
        pa.field("y", pa.uint16()),
        pa.field("w", pa.uint16()),
        pa.field("h", pa.uint16()),
        pa.field("cost", pa.float32()),
        pa.field("dist", pa.float32()),
        pa.field("fracBits", pa.float32()),
        pa.field("lambda", pa.float32()),
        pa.field("intra_mode", pa.uint8()),
        pa.field("isp_mode", pa.int8()),
        pa.field("multi_ref_idx", pa.uint8()),
        pa.field("mip_flag", pa.bool_()),
        pa.field("lfnst_idx", pa.uint8()),
        pa.field("mts_flag", pa.uint8()),
        pa.field("mpm0", pa.uint8()),
        pa.field("mpm1", pa.uint8()),
        pa.field("mpm2", pa.uint8()),
        pa.field("mpm3", pa.uint8()),
        pa.field("mpm4", pa.uint8()),
        pa.field("mpm5", pa.uint8()),
    ]
)


def _match_to_dict(match):
    return {
        "x": np.uint16(match.group(1)),
        "y": np.uint16(match.group(2)),
        "w": np.uint16(match.group(3)),
        "h": np.uint16(match.group(4)),
        "cost": np.float32(match.group(5)),
        "dist": np.float32(match.group(6)),
        "fracBits": np.float32(match.group(7)),
        "lambda": np.float32(match.group(8)),
        "intra_mode": np.uint8(match.group(9)),
        "isp_mode": np.int8(match.group(10)),
        "multi_ref_idx": np.uint8(match.group(11)),
        "mip_flag": np.bool_(match.group(12) != "0"),
        "lfnst_idx": np.uint8(match.group(13)),
        "mts_flag": np.uint8(match.group(14)),
        "mpm0": np.uint8(match.group(15)),
        "mpm1": np.uint8(match.group(16)),
        "mpm2": np.uint8(match.group(17)),
        "mpm3": np.uint8(match.group(18)),
        "mpm4": np.uint8(match.group(19)),
        "mpm5": np.uint8(match.group(20)),
    }


# This is a very rough estimate of the number of partitions required to
# achieve a target partition size. It is based on the observation that
# the total size after serialisation is typically 4% of the size of the
# raw trace.
def _size_to_partitions(file_size, target_partition_size_mb=32768):
    return int(np.ceil(0.04 * file_size / (target_partition_size_mb * 1024 * 1024)))


class RowReader:
    """Iterator which reads rows from a raw trace file."""

    def __init__(self, path):
        # We can't use a context manager here because the lifetime of the file extends
        # beyond the scope of this function
        # pylint: disable-next=consider-using-with
        self.file = open(path, "r", encoding="utf-8")

    def __iter__(self):
        return self

    def __next__(self):
        line = self.file.readline()
        if line == "":
            raise StopIteration

        match = RD_LINE_PATTERN.match(line)
        if not match:
            return self.__next__()

        return _match_to_dict(match)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Close the underlying file."""
        self.file.close()


def row_size(schema):
    """Return the size of a row in bytes."""
    return sum(
        field.type.bit_width // 8
        for field in schema
        if field.type.bit_width is not None
    )


class RowGroupReader:
    """Iterator which groups rows from a raw trace file into groups,
    each approximately `row_group_size` bytes."""

    def __init__(self, path, schema, row_group_size=33554432):
        self.row_reader = RowReader(path)
        self.row_group_nrows = (row_group_size) // row_size(schema)
        self.schema = schema

    def __iter__(self):
        return self

    def __next__(self):
        rows = [row for _, row in zip(range(self.row_group_nrows), self.row_reader)]
        if len(rows) == 0:
            raise StopIteration
        columns = [pa.array([row[field.name] for row in rows]) for field in self.schema]
        row_group = pa.RecordBatch.from_arrays(columns, schema=self.schema)
        return row_group

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Close the underlying `RowReader`"""
        self.row_reader.close()


# Pylint shouldn't count named arguments with default values towards the
# too-many-arguments limit IMO.
# pylint: disable-next=too-many-arguments
def rd_dump_to_parquet(
    input_path,
    output_path,
    schema=RD_SCHEMA,
    partitions=None,
    row_group_size=33554432,
    quiet=True,
):
    """Convert a raw trace file to a parquet dataset."""

    if partitions is None:
        partitions = _size_to_partitions(os.path.getsize(input_path))

    with tqdm(
        total=os.path.getsize(input_path), disable=quiet, unit="B", unit_scale=True
    ) as pbar:
        with pq.ParquetWriter(output_path, schema) as writer:
            with RowGroupReader(input_path, schema, row_group_size) as reader:
                for row_group in reader:
                    writer.write_batch(row_group, row_group_size=row_group_size)
                    pbar.update(reader.row_reader.file.tell() - pbar.n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input file")
    parser.add_argument("output", help="Path to output file")
    parser.add_argument(
        "--num-parquet-files",
        help="Number of parquet files",
        type=int,
    )
    parser.add_argument(
        "--row-group-size",
        help="Row group size (KiB)",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--quiet",
        help="Suppress progress bar",
        action="store_true",
    )
    args = parser.parse_args()

    rd_dump_to_parquet(
        args.input,
        args.output,
        RD_SCHEMA,
        partitions=args.num_parquet_files,
        row_group_size=args.row_group_size * 1024,
        quiet=args.quiet,
    )

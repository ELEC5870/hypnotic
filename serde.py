#!/usr/bin/env python3

disclaimer = """
Copyright (c) 2023 Frank Plowman

This project is licensed under the terms of the MIT license. For details, see LICENSE
"""

desc = """
This is a utility for converting raw rate-distortion data from VTM into a variety of formats
for easier handling. At the moment, the tool supports (de)serialising:
* Raw data (.rd)
* Pickle (.pickle)
* Petastorm (.petastorm)
* Parquet (.parquet)
"""

from contextlib import nullcontext
import os
import re
import unittest
from tqdm import tqdm
import warnings

from common import Area, Entry


RD_LINE_PATTERN = re.compile(
    r"IntraCost T \[x=(\d+),y=(\d+),w=(\d+),h=(\d+)\] ([\d.]+) \((\d+),([\d-]+),(\d+),(\d+),(\d+),(\d+),\[(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\]\)"
)


def deserialise_rd_line(line):
    if (match := RD_LINE_PATTERN.match(line)) is not None:
        return Entry(
            Area(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
            ),
            float(match.group(5)),
            int(match.group(6)),
            int(match.group(7)),
            int(match.group(8)),
            match.group(9) != "0",
            int(match.group(10)),
            int(match.group(11)),
            (
                int(match.group(12)),
                int(match.group(13)),
                int(match.group(14)),
                int(match.group(15)),
                int(match.group(16)),
                int(match.group(17)),
            ),
        )


def deserialise_rd(path, quiet=True):
    data = []
    file_size = os.path.getsize(path)

    pbar = tqdm(desc="Deserialising", total=file_size, unit="B", unit_scale=True)
    with pbar if not quiet else nullcontext():
        with open(path, "r") as f:
            while (line := f.readline()) != "":
                if (entry := deserialise_rd_line(line)) is not None:
                    data.append(entry)

                if pbar is not None:
                    pbar.update(len(line))

    return data


def deserialise_pickle(path):
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def deserialise_petastorm(path):
    from petastorm import make_reader

    url = f"file://{os.path.abspath(path)}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        with make_reader(url) as reader:
            return [
                Entry(
                    Area(
                        row.x,
                        row.y,
                        row.w,
                        row.h,
                    ),
                    row.cost,
                    row.intra_mode,
                    row.isp_mode,
                    row.multi_ref_idx,
                    row.mip_flag,
                    row.lfnst_idx,
                    row.mts_flag,
                    (row.mpm0, row.mpm1, row.mpm2, row.mpm3, row.mpm4, row.mpm5),
                )
                for row in reader
            ]


def deserialise_parquet(path, quiet=True):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    table = table.to_batches()
    if not quiet:
        table = tqdm(table, desc="Deserialising")

    data = []
    for batch in table:
        data += [
            Entry(
                Area(
                    row["x"],
                    row["y"],
                    row["w"],
                    row["h"],
                ),
                row["cost"],
                row["intra_mode"],
                row["isp_mode"],
                row["multi_ref_idx"],
                row["mip_flag"],
                row["lfnst_idx"],
                row["mts_flag"],
                (
                    row["mpm0"],
                    row["mpm1"],
                    row["mpm2"],
                    row["mpm3"],
                    row["mpm4"],
                    row["mpm5"],
                ),
            )
            for row in batch.to_pylist()
        ]

        # data += [
        #     for row in batch]

    # pbar = tqdm(desc="Deserialising")
    # with pbar if not quiet else nullcontext():

    # if not quiet:
    #     pbar = tqdm(table, desc="Deserialising")
    # else:
    #     pbar = table.to_pylist()

    return data


def deserialise(path, format=None, quiet=True):
    if format is None:
        format = os.path.splitext(path.rstrip("/"))[1][1:]

    if format == "rd":
        data = deserialise_rd(path, quiet)
    elif format == "pickle":
        data = deserialise_pickle(path)
    elif format == "petastorm":
        data = deserialise_petastorm(path)
    elif format == "parquet":
        data = deserialise_parquet(path, quiet)
    else:
        raise Exception("Unknown format")

    return data


def serialise_rd(data, path):
    with open(path, "w") as f:
        for entry in data:
            f.write(
                f"IntraCost T [x={entry.area.x},y={entry.area.y},w={entry.area.w},h={entry.area.h}] {entry.cost} ({entry.intra_mode},{entry.isp_mode},{entry.multi_ref_idx},{int(entry.mip_flag)},{entry.lfnst_idx},{entry.mts_flag},[{','.join(str(mode) for mode in entry.mpm)}])\n"
            )


def serialise_pickle(data, path):
    import pickle

    with open(path, "wb") as f:
        pickle.dump(data, f)


def serialise_petastorm(data, path, partitions=1, cores=2, quiet=True):
    import numpy as np
    from petastorm.codecs import ScalarCodec, NdarrayCodec
    from petastorm.etl.dataset_metadata import materialize_dataset
    from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
    from pyspark.sql import SparkSession
    from pyspark.sql.types import FloatType, IntegerType

    output_url = f"file://{os.path.abspath(path)}"

    # fmt: off
    schema = Unischema(
        "RDSchema",
        [
            UnischemaField("x",             np.uint16,  (),   ScalarCodec(IntegerType()), False),
            UnischemaField("y",             np.uint16,  (),   ScalarCodec(IntegerType()), False),
            UnischemaField("w",             np.uint16,  (),   ScalarCodec(IntegerType()), False),
            UnischemaField("h",             np.uint16,  (),   ScalarCodec(IntegerType()), False),
            UnischemaField("cost",          np.float32, (),   ScalarCodec(FloatType()),   False),
            UnischemaField("intra_mode",    np.uint8,   (),   ScalarCodec(IntegerType()), False),
            UnischemaField("isp_mode",      np.int8,    (),   ScalarCodec(IntegerType()), False),
            UnischemaField("multi_ref_idx", np.uint8,   (),   ScalarCodec(IntegerType()), False),
            UnischemaField("mip_flag",      np.bool_,   (),   ScalarCodec(IntegerType()), False),
            UnischemaField("lfnst_idx",     np.uint8,   (),   ScalarCodec(IntegerType()), False),
            UnischemaField("mts_flag",      np.uint8,   (),   ScalarCodec(IntegerType()), False),
            UnischemaField("mpm0",          np.uint8,   (),   ScalarCodec(IntegerType()), False),
            UnischemaField("mpm1",          np.uint8,   (),   ScalarCodec(IntegerType()), False),
            UnischemaField("mpm2",          np.uint8,   (),   ScalarCodec(IntegerType()), False),
            UnischemaField("mpm3",          np.uint8,   (),   ScalarCodec(IntegerType()), False),
            UnischemaField("mpm4",          np.uint8,   (),   ScalarCodec(IntegerType()), False),
            UnischemaField("mpm5",          np.uint8,   (),   ScalarCodec(IntegerType()), False),
        ],
    )
    # fmt: on

    def row_generator(entry):
        return {
            schema.x.name: np.uint16(entry.area.x),
            schema.y.name: np.uint16(entry.area.y),
            schema.w.name: np.uint16(entry.area.w),
            schema.h.name: np.uint16(entry.area.h),
            schema.cost.name: np.float32(entry.cost),
            schema.intra_mode.name: np.uint8(entry.intra_mode),
            schema.isp_mode.name: np.int8(entry.isp_mode),
            schema.multi_ref_idx.name: np.uint8(entry.multi_ref_idx),
            schema.mip_flag.name: np.bool_(entry.mip_flag),
            schema.lfnst_idx.name: np.uint8(entry.lfnst_idx),
            schema.mts_flag.name: np.uint8(entry.mts_flag),
            schema.mpm0.name: np.uint8(entry.mpm[0]),
            schema.mpm1.name: np.uint8(entry.mpm[1]),
            schema.mpm2.name: np.uint8(entry.mpm[2]),
            schema.mpm3.name: np.uint8(entry.mpm[3]),
            schema.mpm4.name: np.uint8(entry.mpm[4]),
            schema.mpm5.name: np.uint8(entry.mpm[5]),
        }

    spark = (
        SparkSession.builder.config("spark.driver.memory", "2g")
        .master(f"local[{cores}]")
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    if not quiet:
        data = tqdm(data, desc="Serialising")

    # Suppress FutureWarning from petastorm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        with materialize_dataset(spark, output_url, schema, row_group_size_mb=128):
            rows_rdd = (
                sc.parallelize(data, cores)
                .map(row_generator)
                .map(lambda x: dict_to_spark_row(schema, x))
            )

            spark.createDataFrame(rows_rdd, schema.as_spark_schema()).coalesce(
                partitions
            ).write.mode("overwrite").parquet(output_url)


def serialise_parquet(data, path, quiet=True):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds

    fields = [
        ("x", [entry.area.x for entry in data], pa.uint16()),
        ("y", [entry.area.y for entry in data], pa.uint16()),
        ("w", [entry.area.w for entry in data], pa.uint16()),
        ("h", [entry.area.h for entry in data], pa.uint16()),
        ("cost", [entry.cost for entry in data], pa.float32()),
        ("intra_mode", [entry.intra_mode for entry in data], pa.uint8()),
        ("isp_mode", [entry.isp_mode for entry in data], pa.int8()),
        ("multi_ref_idx", [entry.multi_ref_idx for entry in data], pa.uint8()),
        ("mip_flag", [entry.mip_flag for entry in data], pa.bool_()),
        ("lfnst_idx", [entry.lfnst_idx for entry in data], pa.uint8()),
        ("mts_flag", [entry.mts_flag for entry in data], pa.uint8()),
        ("mpm0", [entry.mpm[0] for entry in data], pa.uint8()),
        ("mpm1", [entry.mpm[1] for entry in data], pa.uint8()),
        ("mpm2", [entry.mpm[2] for entry in data], pa.uint8()),
        ("mpm3", [entry.mpm[3] for entry in data], pa.uint8()),
        ("mpm4", [entry.mpm[4] for entry in data], pa.uint8()),
        ("mpm5", [entry.mpm[5] for entry in data], pa.uint8()),
    ]

    pbar = tqdm(desc="Serialising", total=len(fields) + 2)
    with pbar if not quiet else nullcontext():
        columns = []
        names = []
        for name, val, type in fields:
            columns.append(pa.array(val, type=type))
            names.append(name)
            if not quiet:
                pbar.update()

        table = pa.table(
            columns,
            names,
        )
        if not quiet:
            pbar.update()

        ds.write_dataset(
            table, path, format="parquet", existing_data_behavior="overwrite_or_ignore"
        )
        if not quiet:
            pbar.update()


def serialise(data, path, format=None, quiet=True):
    if format is None:
        format = os.path.splitext(path)[1][1:]

    if format == "rd":
        serialise_rd(data, path)
    elif format == "pickle":
        serialise_pickle(data, path)
    elif format == "petastorm":
        serialise_petastorm(data, path, quiet=quiet)
    elif format == "parquet":
        serialise_parquet(data, path, quiet=quiet)
    else:
        raise Exception("Unknown format")


class Tests(unittest.TestCase):
    # fmt: off
    test_data = [
        Entry(
            Area(0, 1, 2, 3),
            4.0,
            5, 6, 7, False, 8, 9, (10, 11, 12, 13, 14, 15)
        ),
        Entry(
            Area(16, 17, 18, 19),
            20.0,
            21, 22, 23, True, 24, 25, (26, 27, 28, 29, 30, 31),
        ),
    ]
    # fmt: on

    def test_rd(self):
        import tempfile

        # @TODO: refactor to use a buffer instead of a temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pickle")
            serialise_rd(self.test_data, path)
            data = deserialise_rd(path)

        self.assertEqual(len(data), len(self.test_data))
        self.assertEqual(data, self.test_data)

    def test_pickle(self):
        import tempfile

        # @TODO: refactor to use a buffer instead of a temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pickle")
            serialise_pickle(self.test_data, path)
            data = deserialise_pickle(path)

        self.assertEqual(len(data), len(self.test_data))
        self.assertEqual(data, self.test_data)

    def test_petastorm(self):
        import tempfile
        from collections import Counter

        # @TODO: refactor to use a buffer instead of a temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.petastorm")
            serialise_petastorm(self.test_data, path)
            data = deserialise_petastorm(path)

        self.assertEqual(len(data), len(self.test_data))
        # Order is not preserved, so use a multiset (Counter).
        self.assertEqual(Counter(data), Counter(self.test_data))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=disclaimer,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input", help="Input path")
    parser.add_argument("output", help="Output path")
    parser.add_argument("--format", "-f", help="Output format")
    parser.add_argument("--time", "-t", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    if args.time:
        import time

    if args.format is None:
        args.format = os.path.splitext(args.output)[1][1:]

    # parse input
    if args.time:
        start = time.time()
    data = deserialise(args.input, quiet=args.quiet)
    if args.time:
        end = time.time()
        print(f"Deserialised in {end - start:.2f}s")

    print(f"Read {len(data)} entries")

    # write output
    if args.time:
        start = time.time()
    serialise(data, args.output, format=args.format, quiet=args.quiet)
    if args.time:
        end = time.time()
        print(f"Serialised in {end - start:.2f}s")

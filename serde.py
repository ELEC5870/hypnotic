#!/usr/bin/env python3

disclaimer = """
Copyright (c) 2023 Frank Plowman

This project is licensed under the terms of the MIT license. For details, see LICENSE
"""

desc = """
This is a utility for converting raw rate-distortion data from VTM into a variety of formats for easier handling. At the moment, the tool supports (de)serialising:
* Raw data (`.rd`)
* Pickle (`.pickle`)
* Petastorm (`.petastorm`)
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


def deserialise_rd(path, quiet=False):
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

        with make_reader(url, reader_pool_type="process") as reader:
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
                    tuple(row.mpm),
                )
                for row in reader
            ]


def deserialise(path, format=None):
    if format is None:
        format = os.path.splitext(path)[1][1:].rstrip("/")

    if format == "rd":
        data = deserialise_rd(path)
    elif format == "pickle":
        data = deserialise_pickle(path)
    elif format == "petastorm":
        data = deserialise_petastorm(path)
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


def serialise_petastorm(data, path, num_parquet_files=10, cores=2, quiet=False):
    import numpy as np
    from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
    from petastorm.etl.dataset_metadata import materialize_dataset
    from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
    from pyspark.sql import SparkSession
    from pyspark.sql.types import FloatType, IntegerType

    output_url = f"file://{os.path.abspath(path)}"

    # fmt: off
    RDSchema = Unischema(
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
            UnischemaField("mpm",           np.uint8,   (6,), NdarrayCodec(),             False),
        ],
    )
    # fmt: on

    def row_generator(entry):
        return {
            RDSchema.x.name: np.uint16(entry.area.x),
            RDSchema.y.name: np.uint16(entry.area.y),
            RDSchema.w.name: np.uint16(entry.area.w),
            RDSchema.h.name: np.uint16(entry.area.h),
            RDSchema.cost.name: np.float32(entry.cost),
            RDSchema.intra_mode.name: np.uint8(entry.intra_mode),
            RDSchema.isp_mode.name: np.int8(entry.isp_mode),
            RDSchema.multi_ref_idx.name: np.uint8(entry.multi_ref_idx),
            RDSchema.mip_flag.name: np.bool_(entry.mip_flag),
            RDSchema.lfnst_idx.name: np.uint8(entry.lfnst_idx),
            RDSchema.mts_flag.name: np.uint8(entry.mts_flag),
            RDSchema.mpm.name: np.asarray(entry.mpm, dtype=np.uint8),
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

        with materialize_dataset(spark, output_url, RDSchema, row_group_size_mb=128):
            rows_rdd = (
                sc.parallelize(data, cores)
                .map(row_generator)
                .map(lambda x: dict_to_spark_row(RDSchema, x))
            )

            spark.createDataFrame(rows_rdd, RDSchema.as_spark_schema()).coalesce(
                num_parquet_files
            ).write.mode("overwrite").parquet(output_url)


def serialise(data, path, format=None):
    if format is None:
        format = os.path.splitext(path)[1][1:]

    if format == "rd":
        serialise_rd(data, path)
    elif format == "pickle":
        serialise_pickle(data, path)
    elif format == "petastorm":
        serialise_petastorm(data, path)
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

    parser = argparse.ArgumentParser(description=desc, epilog=disclaimer)
    parser.add_argument("input", help="Input file")
    parser.add_argument("output", help="Output file")
    parser.add_argument("--format", "-f", help="Output file format")
    parser.add_argument("--time", "-t", action="store_true")
    args = parser.parse_args()

    if args.time:
        import time

    if args.format is None:
        args.format = os.path.splitext(args.output)[1][1:]

    # parse input
    if args.time:
        start = time.time()
    data = deserialise(args.input)
    if args.time:
        end = time.time()
        print(f"Deserialised in {end - start:.2f}s")

    print(f"Read {len(data)} entries")

    # write output
    if args.time:
        start = time.time()
    serialise(data, args.output, format=args.format)
    if args.time:
        end = time.time()
        print(f"Serialised in {end - start:.2f}s")

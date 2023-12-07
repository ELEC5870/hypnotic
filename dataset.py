from itertools import islice
import os
import random
import unittest

from more_itertools import consume
from PIL import Image
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import torch
from torchvision.transforms import v2 as transforms

NUM_INTRA_MODES = 67


class ParquetRDDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        image_path,
        parquet_path,
        limit=None,
        offset=0,
        filter=None,
        transform=None,
        target_transform=None,
        deterministic=False,
    ):
        if os.path.isdir(image_path):
            self.image = {}
            for path in os.listdir(image_path):
                if path.endswith(".bmp"):
                    sequence = (
                        os.path.basename(path).replace(".bmp", "").replace("_", "-")
                    )
                    self.image[sequence] = Image.open(os.path.join(image_path, path))
        else:
            sequence = (
                os.path.basename(image_path).replace(".bmp", "").replace("_", "-")
            )
            self.image = {sequence: Image.open(os.path.expanduser(image_path))}

        partitioning = ds.partitioning(
            flavor="filename", schema=pa.schema([("sequence", pa.string())])
        )
        self.dataset = ds.dataset(
            parquet_path, format="parquet", partitioning=partitioning
        )
        self.limit = limit
        self.offset = offset
        self.filter = filter
        self.transform = transform
        self.target_transform = target_transform
        self.deterministic = deterministic
        self.predictors = self.dataset.schema.names
        for col in ["intra_mode", "cost", "dist", "fracBits"]:
            try:
                self.predictors.remove(col)
            # Ignore if col is not present
            except ValueError:
                pass

        self.len = sum(1 for _ in self.dataset.to_batches(filter=self.filter))

    def __iter__(self):
        iter = self.dataset.to_batches(filter=self.filter)
        consume(iter, self.offset)
        if self.limit:
            iter = islice(iter, self.limit)
        return map(self._modify_batch, iter)

    def __len__(self):
        capacity = self.len - self.offset
        return min(capacity, self.limit) if self.limit else capacity

    def _modify_batch(self, batch):
        batch = pa.Table.from_batches([batch])
        batch = batch.group_by(
            self.predictors,
            use_threads=not self.deterministic,
        ).aggregate([("intra_mode", "list"), ("cost", "list")])
        # @TODO: This could be made less strict.
        #        Rows with more than 67 intra modes can likely be included. Duplicate
        #        costs for a given intra mode could be averaged or chosen from at
        #        random.
        batch = batch.filter(
            pc.list_value_length(pc.field("intra_mode_list")) == NUM_INTRA_MODES
        )
        batch = batch.to_pylist()

        images = []
        scalars = []
        targets = []
        for i, row in enumerate(batch):
            try:
                pu = self.image[row["sequence"]].crop(
                    (row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"])
                )
            except KeyError:
                print(self.image)
                raise KeyError(row["sequence"])
            if self.transform:
                pu = self.transform(pu)
            images.append(pu)

            scalars.append(
                torch.tensor(
                    [
                        row["isp_mode"],
                        row["multi_ref_idx"],
                        row["mip_flag"],
                        row["lfnst_idx"],
                        row["mts_flag"],
                        row["mpm0"],
                        row["mpm1"],
                        row["mpm2"],
                        row["mpm3"],
                        row["mpm4"],
                        row["mpm5"],
                    ]
                )
            )

            costs = [[] for _ in range(NUM_INTRA_MODES)]
            for intra_mode, cost in zip(row["intra_mode_list"], row["cost_list"]):
                costs[intra_mode].append(cost)
            costs = [random.choice(cost) for cost in costs]
            targets.append(costs)

        targets = torch.tensor(targets)
        if self.target_transform:
            targets = self.target_transform(targets)

        return (torch.stack(images), torch.stack(scalars)), targets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("parquet_path", type=str)
    args = parser.parse_args()

    dataset = ParquetRDDataset(
        args.image_path,
        args.parquet_path,
        filter=(pc.field("w") == 16) & (pc.field("h") == 16),
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        ),
    )
    input, target = dataset[0]
    print(f"{input.shape=}")
    print(f"{target.shape=}")


class Tests(unittest.TestCase):
    def test(self):
        import os
        import tempfile

        from serialise import rd_dump_to_parquet

        # 4x4 black .bmp file
        IMAGE_TEST_DATA = bytes.fromhex(
            "424dba000000000000008a0000007c000000040000000400000001001800000000003000000"
            "0000000000000000000000000000000000000ff0000ff0000ff000000000000ff424752738f"
            "c2f52851b81e151e85eb01333333136666662666666606999999093d0ad703285c8f3200000"
            "000000000000000000004000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
        )

        RD_TEST_DATA = """
IntraCost T [x=0,y=1,w=2,h=3] 0.0 (0,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 1.0 (1,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 2.0 (2,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 3.0 (3,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 4.0 (4,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 5.0 (5,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 6.0 (6,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 7.0 (7,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 8.0 (8,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 9.0 (9,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 10.0 (10,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 11.0 (11,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 12.0 (12,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 13.0 (13,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 14.0 (14,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 15.0 (15,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 16.0 (16,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 17.0 (17,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 18.0 (18,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 19.0 (19,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 20.0 (20,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 21.0 (21,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 22.0 (22,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 23.0 (23,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 24.0 (24,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 25.0 (25,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 26.0 (26,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 27.0 (27,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 28.0 (28,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 29.0 (29,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 30.0 (30,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 31.0 (31,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 32.0 (32,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 33.0 (33,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 34.0 (34,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 35.0 (35,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 36.0 (36,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 37.0 (37,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 38.0 (38,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 39.0 (39,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 40.0 (40,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 41.0 (41,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 42.0 (42,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 43.0 (43,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 44.0 (44,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 45.0 (45,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 46.0 (46,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 47.0 (47,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 48.0 (48,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 49.0 (49,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 50.0 (50,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 51.0 (51,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 52.0 (52,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 53.0 (53,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 54.0 (54,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 55.0 (55,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 56.0 (56,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 57.0 (57,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 58.0 (58,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 59.0 (59,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 60.0 (60,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 61.0 (61,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 62.0 (62,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 63.0 (63,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 64.0 (64,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 65.0 (65,-1,0,0,0,0,[4,5,6,7,8,9]) 
IntraCost T [x=0,y=1,w=2,h=3] 66.0 (66,-1,0,0,0,0,[4,5,6,7,8,9]) 
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "data.bmp")
            with open(image_path, "wb") as f:
                f.write(IMAGE_TEST_DATA)

            # Write raw rows in a random order
            rd_data = RD_TEST_DATA.splitlines()
            random.shuffle(rd_data)
            rd_data = "\n".join(rd_data)
            rd_path = os.path.join(tmpdir, "data.rd")
            with open(rd_path, "w") as f:
                f.write(rd_data)

            parquet_path = os.path.join(tmpdir, "data.parquet")
            rd_dump_to_parquet(rd_path, parquet_path)

            dataset = ParquetRDDataset(
                image_path,
                parquet_path,
                transform=transforms.ToImage(),
            )

            image, target = dataset[0]
            self.assertTrue(image.count_nonzero() == 0)
            for i, cost in enumerate(target[0]):
                self.assertEqual(cost, i)

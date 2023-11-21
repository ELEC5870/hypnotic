import random
import unittest

from PIL import Image
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torchvision.transforms import v2 as transforms

NUM_INTRA_MODES = 67


class ParquetRDDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, parquet_path, filter=None, transform=None):
        self.image = Image.open(image_path)
        self.pq_file = pq.ParquetFile(parquet_path)
        self.filter = filter
        self.transform = transform

    def __len__(self):
        return self.pq_file.num_row_groups

    def __getitem__(self, idx):
        row_group = self.pq_file.read_row_group(idx)
        if self.filter is not None:
            row_group = row_group.filter(self.filter)
        row_group = row_group.group_by(
            [
                "x",
                "y",
                "w",
                "h",
                "isp_mode",
                "multi_ref_idx",
                "mip_flag",
                "lfnst_idx",
                "mts_flag",
            ]
        ).aggregate([("intra_mode", "list"), ("cost", "list")])
        # @TODO: This could be made less strict.
        #        Rows with more than 67 intra modes can likely be included. Duplicate
        #        costs for a given intra mode could be averaged or chosen from at
        #        random.
        row_group = row_group.filter(
            pc.list_value_length(pc.field("intra_mode_list")) == NUM_INTRA_MODES
        )
        row_group = row_group.to_pylist()

        inputs = []
        targets = []
        for i, row in enumerate(row_group):
            pu = self.image.crop(
                (row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"])
            )
            if self.transform:
                pu = self.transform(pu)
            inputs.append(pu)

            costs = [[] for _ in range(NUM_INTRA_MODES)]
            for intra_mode, cost in zip(row["intra_mode_list"], row["cost_list"]):
                costs[intra_mode].append(cost)
            costs = [random.choice(cost) for cost in costs]
            targets.append(costs)

        return torch.stack(inputs), torch.tensor(targets)


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

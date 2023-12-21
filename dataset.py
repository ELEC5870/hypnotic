from itertools import islice
import os
import random
import unittest

from more_itertools import consume
from PIL import Image
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torchvision.transforms import v2 as transforms

NUM_INTRA_MODES = 67


class ParquetRDDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        image_path,
        parquet_path,
        filter=None,
        transform=None,
        target_transform=None,
        deterministic=False,
    ):
        self.images = self._load_images(image_path)
        self.dataset = self._load_dataset(parquet_path)
        self.predictors = self._get_predictors(self.dataset.schema)
        self.filter = filter
        self.transform = transform
        self.target_transform = target_transform
        self.deterministic = deterministic

    def _load_image(self, path):
        sequence = os.path.splitext(os.path.basename(path))[0]
        image = Image.open(os.path.expanduser(path))
        return sequence, image

    def _load_images(self, path):
        images = {}
        if os.path.isdir(path):
            for sub_path in os.listdir(path):
                if sub_path.endswith(".bmp"):
                    self._load_image(os.path.join(path, sub_path))
                    sequence, image = self._load_image(os.path.join(path, sub_path))
                    images[sequence] = image
        else:
            sequence, image = self._load_image(os.path.join(path))
            images[sequence] = image
        return images

    def _load_dataset(self, path):
        dataset = ds.dataset(
            path,
            format="parquet",
        )
        return dataset

    def _get_predictors(self, schema):
        predictors = schema.names
        for col in ["intra_mode", "cost", "dist", "fracBits"]:
            try:
                predictors.remove(col)
            # Ignore if col is not present
            except ValueError:
                pass
        return predictors

    def _get_batches(self):
        filter = pc.list_value_length(pc.field("intra_mode_list")) == NUM_INTRA_MODES
        if self.filter is not None:
            filter &= self.filter
        return self.dataset.to_batches(filter=filter, batch_size=1024)

    def __iter__(self):
        batches = self._get_batches()
        for batch in batches:
            for row in batch.to_pylist():
                yield self._modify_row(row)

    def _modify_row(self, row):
        # @TODO: Remove this.
        #        Make this substitution when preparing the dataset.
        sequence = row["sequence"].replace("-", "_")
        image = self.images[sequence].crop(
            (row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"])
        )
        if self.transform:
            image = self.transform(image)

        scalars = torch.tensor(
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

        targets = torch.full((NUM_INTRA_MODES,), torch.nan)
        for intra_mode, cost in zip(row["intra_mode_list"], row["cost_list"]):
            targets[intra_mode] = cost
        if targets.isnan().any():
            raise ValueError(f"NaN cost(s) in row {row}")
        if self.target_transform:
            targets = self.target_transform(targets)

        return (image, scalars), targets


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
        from prepare import prepare_dataset

        # 4x4 black .bmp file
        IMAGE_TEST_DATA = bytes.fromhex(
            "424dba000000000000008a0000007c000000040000000400000001001800000000003000000"
            "0000000000000000000000000000000000000ff0000ff0000ff000000000000ff424752738f"
            "c2f52851b81e151e85eb01333333136666662666666606999999093d0ad703285c8f3200000"
            "000000000000000000004000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
        )

        RD_TEST_DATA = """
IntraCost T x=0,y=1,w=2,h=3,cost=0.0,dist=0.0,fracBits=0.0,lambda=1.0,modeId=66,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=1.0,dist=1.0,fracBits=1.0,lambda=1.0,modeId=65,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=2.0,dist=2.0,fracBits=2.0,lambda=1.0,modeId=64,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=3.0,dist=3.0,fracBits=3.0,lambda=1.0,modeId=63,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=4.0,dist=4.0,fracBits=4.0,lambda=1.0,modeId=62,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=5.0,dist=5.0,fracBits=5.0,lambda=1.0,modeId=61,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=6.0,dist=6.0,fracBits=6.0,lambda=1.0,modeId=60,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=7.0,dist=7.0,fracBits=7.0,lambda=1.0,modeId=59,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=8.0,dist=8.0,fracBits=8.0,lambda=1.0,modeId=58,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=9.0,dist=9.0,fracBits=9.0,lambda=1.0,modeId=57,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=10.0,dist=10.0,fracBits=10.0,lambda=1.0,modeId=56,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=11.0,dist=11.0,fracBits=11.0,lambda=1.0,modeId=55,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=12.0,dist=12.0,fracBits=12.0,lambda=1.0,modeId=54,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=13.0,dist=13.0,fracBits=13.0,lambda=1.0,modeId=53,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=14.0,dist=14.0,fracBits=14.0,lambda=1.0,modeId=52,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=15.0,dist=15.0,fracBits=15.0,lambda=1.0,modeId=51,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=16.0,dist=16.0,fracBits=16.0,lambda=1.0,modeId=50,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=17.0,dist=17.0,fracBits=17.0,lambda=1.0,modeId=49,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=18.0,dist=18.0,fracBits=18.0,lambda=1.0,modeId=48,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=19.0,dist=19.0,fracBits=19.0,lambda=1.0,modeId=47,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=20.0,dist=20.0,fracBits=20.0,lambda=1.0,modeId=46,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=21.0,dist=21.0,fracBits=21.0,lambda=1.0,modeId=45,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=22.0,dist=22.0,fracBits=22.0,lambda=1.0,modeId=44,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=23.0,dist=23.0,fracBits=23.0,lambda=1.0,modeId=43,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=24.0,dist=24.0,fracBits=24.0,lambda=1.0,modeId=42,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=25.0,dist=25.0,fracBits=25.0,lambda=1.0,modeId=41,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=26.0,dist=26.0,fracBits=26.0,lambda=1.0,modeId=40,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=27.0,dist=27.0,fracBits=27.0,lambda=1.0,modeId=39,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=28.0,dist=28.0,fracBits=28.0,lambda=1.0,modeId=38,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=29.0,dist=29.0,fracBits=29.0,lambda=1.0,modeId=37,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=30.0,dist=30.0,fracBits=30.0,lambda=1.0,modeId=36,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=31.0,dist=31.0,fracBits=31.0,lambda=1.0,modeId=35,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=32.0,dist=32.0,fracBits=32.0,lambda=1.0,modeId=34,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=33.0,dist=33.0,fracBits=33.0,lambda=1.0,modeId=33,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=34.0,dist=34.0,fracBits=34.0,lambda=1.0,modeId=32,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=35.0,dist=35.0,fracBits=35.0,lambda=1.0,modeId=31,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=36.0,dist=36.0,fracBits=36.0,lambda=1.0,modeId=30,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=37.0,dist=37.0,fracBits=37.0,lambda=1.0,modeId=29,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=38.0,dist=38.0,fracBits=38.0,lambda=1.0,modeId=28,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=39.0,dist=39.0,fracBits=39.0,lambda=1.0,modeId=27,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=40.0,dist=40.0,fracBits=40.0,lambda=1.0,modeId=26,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=41.0,dist=41.0,fracBits=41.0,lambda=1.0,modeId=25,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=42.0,dist=42.0,fracBits=42.0,lambda=1.0,modeId=24,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=43.0,dist=43.0,fracBits=43.0,lambda=1.0,modeId=23,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=44.0,dist=44.0,fracBits=44.0,lambda=1.0,modeId=22,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=45.0,dist=45.0,fracBits=45.0,lambda=1.0,modeId=21,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=46.0,dist=46.0,fracBits=46.0,lambda=1.0,modeId=20,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=47.0,dist=47.0,fracBits=47.0,lambda=1.0,modeId=19,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=48.0,dist=48.0,fracBits=48.0,lambda=1.0,modeId=18,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=49.0,dist=49.0,fracBits=49.0,lambda=1.0,modeId=17,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=50.0,dist=50.0,fracBits=50.0,lambda=1.0,modeId=16,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=51.0,dist=51.0,fracBits=51.0,lambda=1.0,modeId=15,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=52.0,dist=52.0,fracBits=52.0,lambda=1.0,modeId=14,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=53.0,dist=53.0,fracBits=53.0,lambda=1.0,modeId=13,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=54.0,dist=54.0,fracBits=54.0,lambda=1.0,modeId=12,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=55.0,dist=55.0,fracBits=55.0,lambda=1.0,modeId=11,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=56.0,dist=56.0,fracBits=56.0,lambda=1.0,modeId=10,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=57.0,dist=57.0,fracBits=57.0,lambda=1.0,modeId=9,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=58.0,dist=58.0,fracBits=58.0,lambda=1.0,modeId=8,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=59.0,dist=59.0,fracBits=59.0,lambda=1.0,modeId=7,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=60.0,dist=60.0,fracBits=60.0,lambda=1.0,modeId=6,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=61.0,dist=61.0,fracBits=61.0,lambda=1.0,modeId=5,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=62.0,dist=62.0,fracBits=62.0,lambda=1.0,modeId=4,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=63.0,dist=63.0,fracBits=63.0,lambda=1.0,modeId=3,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=64.0,dist=64.0,fracBits=64.0,lambda=1.0,modeId=2,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=65.0,dist=65.0,fracBits=65.0,lambda=1.0,modeId=1,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
IntraCost T x=0,y=1,w=2,h=3,cost=66.0,dist=66.0,fracBits=66.0,lambda=1.0,modeId=0,ispMod=-1,multiRefIdx=0,mipFlag=0,lfnstIdx=0,mtsFlag=0,mpm_pred0=4,mpm_pred1=5,mpm_pred2=6,mpm_pred3=7,mpm_pred4=8,mpm_pred5=9 
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

            parquet_path = os.path.join(tmpdir, "data_.parquet")
            rd_dump_to_parquet(rd_path, parquet_path)

            partitioning = ds.partitioning(
                flavor="filename", schema=pa.schema([("sequence", pa.string())])
            )
            dataset = ds.dataset(
                parquet_path, format="parquet", partitioning=partitioning
            )
            prepared_path = os.path.join(tmpdir, "prepared.parquet")
            prepare_dataset(dataset, prepared_path, 0)

            dataset = ParquetRDDataset(
                image_path,
                prepared_path,
                transform=transforms.ToImage(),
            )

            (image, scalars), target = next(iter(dataset))
            self.assertTrue(image.count_nonzero() == 0)
            for mode_id, cost in enumerate(target):
                self.assertEqual(cost, NUM_INTRA_MODES - mode_id - 1)

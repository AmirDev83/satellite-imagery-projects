#!/usr/bin/env python

""""""
import argparse
import math
import os
import time

import numpy as np
import rasterio
import torch
import tqdm
from torch.utils.data import DataLoader


from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets import stack_samples, RasterDataset
from torchgeo.trainers import SemanticSegmentationTask


def preprocess2(sample):
    if "image" in sample:
        sample["image"] = (sample["image"] / 255.0).float()  # inputs are normalized to [0, 1]
    if "mask" in sample:
        sample["mask"] = sample["mask"].squeeze().long()
    return sample


class SingleRasterDataset(RasterDataset):
    def __init__(self, fn, transforms = None):
        self.filename_regex = os.path.basename(fn)
        super().__init__(root=os.path.dirname(fn), transforms=transforms)

def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-model-checkpoint",
        required=True,
        type=str,
        help="model checkpoint (.ckpt format)",
        metavar="CKPT",
    )
    parser.add_argument(
        "--input-image-fn",
        required=True,
        type=str,
        help="input imagery as a geotiff",
        metavar="GEOTIFF",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--output-dir",
        type=str,
        help="directory to write prediction tiles to",
    )
    group.add_argument(
        "--output-fn",
        type=str,
        help="filename to write prediction tiles to (defaults to name of input file)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrites the output tiles if they exist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print stuff",
    )

    parser.add_argument(
        "--gpu",
        required=False,
        type=int,
        help="GPU id to use for inference, CPU is used if not set",
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=2048,
        help="Size of patch to use for inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=64,
        help=(
            "Number of pixels to throw away from each side of the patch after"
            + " inference"
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of workers to use in the dataloader",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Flag to prevent printing stuff",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    # Sanity checks
    assert os.path.exists(args.input_model_checkpoint)
    assert args.input_model_checkpoint.endswith(".ckpt")
    assert os.path.exists(args.input_image_fn)
    assert args.input_image_fn.endswith(".tif") or args.input_image_fn.endswith(".vrt")
    assert int(math.log(args.patch_size, 2)) == math.log(args.patch_size, 2)
    stride = args.patch_size - args.padding * 2

    if args.output_fn is None:
        os.makedirs(args.output_dir, exist_ok=True)
        output_soft_predictions_fn = os.path.join(
            args.output_dir,
            os.path.basename(args.input_image_fn).replace(
                ".tif", "_predictions-soft.tif"
            ),
        )
        output_hard_predictions_fn = os.path.join(
            args.output_dir,
            os.path.basename(args.input_image_fn).replace(".tif", "_predictions.tif"),
        )
    else:
        assert args.output_fn.endswith(".tif")
        output_dir = os.path.dirname(args.output_fn)
        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
        output_hard_predictions_fn = args.output_fn.replace(".tif", "_predictions.tif")

    if not args.overwrite:
        assert not os.path.exists(output_hard_predictions_fn)

    device = torch.device(
        f"cuda:{args.gpu}"
        if (args.gpu is not None) and torch.cuda.is_available()
        else "cpu"
    )

    # Load task and data
    tic = time.time()
    task = SemanticSegmentationTask.load_from_checkpoint(args.input_model_checkpoint)
    task.freeze()
    model = task.model
    model = model.eval().to(device)

    dataset = SingleRasterDataset(
        args.input_image_fn,
        transforms=preprocess2
    )
    sampler = GridGeoSampler(dataset, size=args.patch_size, stride=stride)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=stack_samples,
    )
    if args.verbose:
        print(
            "Finished loading checkpoint and setting up dataset in"
            f" {time.time()-tic:0.2f} seconds"
        )

    # Run inference
    tic = time.time()
    with rasterio.open(args.input_image_fn) as f:
        input_height, input_width = f.shape
        profile = f.profile
        transform = profile["transform"]

    if args.verbose:
        print(f"Input size: {input_height} x {input_width}")
    assert args.patch_size <= input_height
    assert args.patch_size <= input_width
    output = np.zeros((input_height, input_width), dtype=np.uint8)

    if args.quiet:
        dl_enumerator = dataloader
    else:
        dl_enumerator = tqdm.tqdm(dataloader)

    for batch in dl_enumerator:
        images = batch["image"].to(device)
        bboxes = batch["bbox"]
        with torch.inference_mode():
            predictions = task(images)
            predictions = predictions.argmax(axis=1).cpu().numpy()

        for i in range(len(bboxes)):
            bb = bboxes[i]

            left, top = ~transform * (bb.minx, bb.maxy)
            right, bottom = ~transform * (bb.maxx, bb.miny)
            left, right, top, bottom = int(np.round(left)), int(np.round(right)), int(np.round(top)), int(np.round(bottom))
            assert right - left == args.patch_size
            assert bottom - top == args.patch_size

            output[top+args.padding:bottom-args.padding, left+args.padding:right-args.padding] = predictions[i][args.padding:-args.padding, args.padding:-args.padding]

    if args.verbose:
        print(f"Finished running model in {time.time()-tic:0.2f} seconds")

    # Save predictions
    tic = time.time()
    profile["driver"] = "GTiff"
    profile["count"] = 1
    profile["dtype"] = "uint8"
    profile["compress"] = "lzw"
    profile["predictor"] = 2
    profile["nodata"] = 0
    profile["blockxsize"] = 512
    profile["blockysize"] = 512
    profile["tiled"] = True
    profile["interleave"] = "pixel"

    with rasterio.open(output_hard_predictions_fn, "w", **profile) as f:
        f.write(output, 1)

    if args.verbose:
        print(f"Finished saving predictions in {time.time()-tic:0.2f} seconds")


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
import argparse
import os
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from pathlib import Path

import Patcher
import SlideScan

PATH = "C:\\..."
PATH_MASKS = "C:\\..."
PATH_PATCHES = "C:\\..."
PATH_LOGGING_FILE = "C:\\..."

parser = argparse.ArgumentParser()

# general arguments
parser.add_argument("path", default=PATH, help="path to dataset")
parser.add_argument("description", help="path to the description file, labelling each sample as train/test/val")
parser.add_argument("--maskdir", default=PATH_MASKS, help="directory for the masks")
parser.add_argument("--patchesdir", default=PATH_PATCHES,
                    help="directory to save the files with correct coordinates")
parser.add_argument("--logfile", default=PATH_LOGGING_FILE,
                    help="file for saving some logging information")
parser.add_argument("--workers", default=2, type=int, help="number of workers for (optional) parallel processing")
parser.add_argument("--savestyle", default="png", type=str, help="whether to save as png/h5")
parser.add_argument("--fixedcenters", action="store_true", help="whether to generate images at precomputed centers")
parser.add_argument("--centerpath", default=None, type=str, help="the path to the file specifying the centers "
                                                                 "(in native resolution) for each slide")
# tiling specific options
parser.add_argument("--tilesize", type=int, default=224, help="tile size in pixels")
parser.add_argument("--tilelevel", type=int, default=2, help="level for the tiling process")
parser.add_argument("--tilelevelsvs", type=int, default=2, help="level for the tiling process")
parser.add_argument("--stride", type=int, default=224, help="stride for tiling process")
# contain too much area outside the image
parser.add_argument("--blackthreshold", type=int, default=80,
                    help="tiles with an average across all color channels smaller than this are considered as invalid")
parser.add_argument("--padding", type=bool, default=True,
                    help="whether to use padding")
# contain too much background
parser.add_argument("--whitethreshold", type=int, default=15,
                    help="tiles with an average saturation smaller than this are considered invalid")

# contour-specific options
parser.add_argument("--seglevel", type=int, default=6,
                    help="level for the segmentation process (blurring, thresholding, area thresholding)")
parser.add_argument("--seglevelsvs", type=int, default=2,
                    help="level for the segmentation process (blurring, thresholding, area thresholding)")
parser.add_argument("--vizlevel", type=int, default=5,
                    help="level for the visualizing the results of the tissue segmentation")
parser.add_argument("--vizlevelsvs", type=int, default=2,
                    help="level for the visualizing the results of the tissue segmentation")
parser.add_argument("--blocksize", default=3, help="block size for adaptive thresholding. Must be an odd number")
parser.add_argument("--c", default=2, help="constant c for adaptive thresholding")
parser.add_argument("--maxval", default=255, help="maximum possible value of the individual color channels")
parser.add_argument("--kernelsize", default=16, help="size of circular kernel for morphological opening")
parser.add_argument("--erosionsize", default=15, help="size of circular kernel for erosion")
parser.add_argument("--minareatissue", default=200,
                    help="minimal size of an area to be considered in multiples of 512 pixels in the original image")
parser.add_argument("--maxareahole", default=15,
                    help="maximal size of a hole to be considered in multiples of 512 pixels in the original image")
parser.add_argument("--maxnholes", default=15,
                    help="only the maxnholes largest holes per contour are considered for contouring")


def patch_file(file, description, args):
    try:
        if "Set" in description.columns.values:
            dataset = str(description[description["Path"] == file]["Set"].values[0])
        else:
            dataset = "processed"
    except:
        print("Did not find file {}".format(file))
        # print(description[description["Path"] == file])
        return 0, -1, -1, -1, -1, -1, -1

    if file.endswith(".ndpi"):
        file_base = file.strip(".ndpi")
    elif file.endswith(".svs"):
        file_base = file.strip(".svs")
    else:
        raise NotImplementedError("Filetype not supported. Only use .svs and .ndpi.")
    file_path = os.path.join(args.path, file)
    contour_path = os.path.join(args.maskdir, file_base + "_contours.jpeg")
    mask_path = os.path.join(args.maskdir, file_base + "_mask.jpeg")
    patches_path = os.path.join(args.patchesdir, file_base + "_patches.csv")

    # read center file
    center_file = None
    if args.fixedcenters:
        center_file = pd.read_csv(args.centerpath)
        # subset
        center_file = center_file[center_file["File"] == file_path]

    if args.savestyle == "png":
        h5_path = None
        png_path = os.path.join(args.raw_dir, dataset, file_base)
        Path(png_path).mkdir(parents=True, exist_ok=True)
    else:
        h5_path = os.path.join(args.patchesdir, file_base + "_patches.h5")
        png_path = None

    scan = SlideScan.SlideScan(file_path)
    if file.endswith(".ndpi"):
        # does not have any meaning, if center_file is not none.
        scan.computeMask(level=args.seglevel, save_path=contour_path, maxval=args.maxval,
                         kernel_size=args.kernelsize, min_area_tissue=args.minareatissue,
                         max_area_hole=args.maxareahole,
                         max_n_holes=args.maxnholes, ref_patch_size=args.tilesize, blocksize=args.blocksize, c=args.c,
                         erosionsize=args.erosionsize, viz_level=args.vizlevel)
        # construct object
        patcher = Patcher.Patcher(scan, tile_size=args.tilesize, level=args.tilelevel, stride=args.stride,
                                  save_path=patches_path, black_threshold=args.blackthreshold,
                                  white_threshold=args.whitethreshold, use_black_padding=args.padding,
                                  save_style=args.savestyle, h5_path=h5_path, png_path=png_path)
    else:  # has to be svs, since we already checked against unimplemented filetpyes above
        scan.computeMask(level=args.seglevelsvs, save_path=contour_path, maxval=args.maxval,
                         kernel_size=args.kernelsize, min_area_tissue=args.minareatissue,
                         max_area_hole=args.maxareahole,
                         max_n_holes=args.maxnholes, ref_patch_size=args.tilesize, blocksize=args.blocksize, c=args.c,
                         erosionsize=args.erosionsize, viz_level=args.vizlevelsvs)
        # construct object
        patcher = Patcher.Patcher(scan, tile_size=args.tilesize, level=args.tilelevelsvs, stride=args.stride,
                                  save_path=patches_path, black_threshold=args.blackthreshold,
                                  white_threshold=args.whitethreshold, use_black_padding=args.padding,
                                  save_style=args.savestyle, h5_path=h5_path, png_path=png_path)
    counter = patcher.get_patches(center_file=center_file, ret=False)
    if file.endswith(".ndpi"):
        patcher.visualize(mask_path, loadlevel=args.vizlevel)
        segmentation_downsample = scan.slide.level_downsamples[args.seglevel]
        tiling_downsample = scan.slide.level_downsamples[args.tilelevel]
    else:
        patcher.visualize(mask_path, loadlevel=args.vizlevelsvs)
        segmentation_downsample = scan.slide.level_downsamples[args.seglevelsvs]
        tiling_downsample = scan.slide.level_downsamples[args.tilelevelsvs]
    # additional information on slide
    available_downsamples = scan.slide.level_downsamples
    dimensions = scan.slide.dimensions
    vendor = scan.vendor
    objective_power = scan.slide.properties["openslide.objective-power"]
    return counter, segmentation_downsample, tiling_downsample, available_downsamples, dimensions, vendor, objective_power


def parallel_caller(file, description, args):
    if file.endswith(".ndpi") or file.endswith(".svs"):
        print("Processing file {}.".format(file))
        counter, segmentation_downsample, tiling_downsample, available_downsamples, dimensions, vendor, objective_power = patch_file(
            file, description, args)
        return [counter, segmentation_downsample, tiling_downsample, available_downsamples, dimensions, vendor,
                objective_power]
    else:
        print("File {} excluded, because filetype is not implemented.".format(file))


if __name__ == '__main__':
    # read cmd arguments
    args = parser.parse_args()

    # make sure that the paths exist
    Path(args.maskdir).mkdir(parents=True, exist_ok=True)
    Path(args.patchesdir).mkdir(parents=True, exist_ok=True)
    print("Created path " + args.maskdir)
    print("Created path " + args.patchesdir)
    if args.savestyle == "png":
        args.raw_dir = os.path.join(args.patchesdir, "raw")
        Path(args.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.raw_dir, "train")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.raw_dir, "test")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.raw_dir, "val")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.raw_dir, "Eval")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.raw_dir, "processed")).mkdir(parents=True, exist_ok=True)

    # iterate over files
    description = pd.read_csv(args.description)
    # only usable files
    description = description.loc[description["Usable"] == 1]

    results = Parallel(n_jobs=args.workers)(delayed(parallel_caller)(file, description, args) for file in description["Path"].values)
    results = [r for r in results if r is not None]
    results = pd.DataFrame(results, columns=["Num. Tiles", "Segmentation Downsample", "Tiling Downsample",
                                             "All Available Downsamples", "Img. Dimensions", "Vendor",
                                             "Objective Power"])
    # exclude discarded files
    results = results.loc[results["Num. Tiles"] != 0]

    if not len(results["Tiling Downsample"].unique()) == 1:
        print("Data has been sampled on different downsample values. Verify resolution.")

    counts = results["Num. Tiles"].values

    print("Dataset contained {} slides".format(len(counts)))
    print("Average number of extracted tiles: {}".format(np.mean(counts)))
    print("Standard deviation of no. extracted tiles: {}".format(np.std(counts)))
    print("Median number of extracted tiles: {}".format(np.median(counts)))
    print("Minimum number of extracted tiles: {}".format(np.min(counts)))
    print("Maximum number of extracted tiles: {}".format(np.max(counts)))

    print(results)
    results.to_csv(args.logfile)

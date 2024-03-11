import argparse
import os
import gc

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DistDatPar
import time
import datetime
from pathlib import Path
import HistogramMatcher

import LazyHistoDataset
from Encoder import Encoder
import FDAMatcher

parser = argparse.ArgumentParser()
# general config
parser.add_argument('rawdata', type=str, help='path to the directory with the train/val/test folders')
parser.add_argument('patchingdescrdir', type=str, help='path to the directory with the .csv description file')
parser.add_argument('descriptionfile', metavar='DIR2', help='description file for the dataset')
parser.add_argument('--workers', default=8, type=int, help='no. of workers used for preprocessing and data loading')
#
# preprocessing
parser.add_argument('--modelpath', default="/models", type=str, help="path to models for SimSiam checkpointing")
parser.add_argument('--encodeddir', default="/encoded/", type=str,
                    help='where to store the encoded files. Should be an empy directory')

parser.add_argument("--tempfilename", default="temp.csv", type=str, help="tempfile used by simsiam for communication"
                                                                         "There, we look for the meand and stds for"
                                                                         "normalization")
parser.add_argument('--prepbatchsize', default=256, type=int, help='batch size for preprocessing (encoding) the tiles')

# # simsiam specific configs:
parser.add_argument('--num', default=50, type=int, help='num. layers in resnet model')
parser.add_argument('--dim', default=2048, type=int, help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int, help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true', help='Fix learning rate for the predictor')
parser.add_argument('--pdl', '--prediction-layers', default=2, type=int, metavar='PD',
                    help='layers in prediction MLP', dest='pdl')
parser.add_argument('--pjl', '--projection-layers', default=3, type=int, metavar='PJ',
                    help='layers in projection MLP', dest='pjl')
# misc.
parser.add_argument('--encodertyp', default="simsiam", type=str, choices=["simsiam", "resnet"],
                    help="whether to use simsiam or a (pretrained) resnet as encoder")
parser.add_argument("--lastavg", action="store_true", help="whether to use the output after the last avg layer for the"
                                                           "pretrained resnet.")
parser.add_argument("--withmlp", action="store_true", help="whether to use the mlp head for encoding as well"
                                                           "(SimSiam only)")
parser.add_argument("--amp", action="store_true", help="whether to use AMP")
parser.add_argument("--nhwc", action="store_true", help="whether to work with channels last")
parser.add_argument("--histograms", action="store_true", help="whether to work with histogram correction")
parser.add_argument("--fda", action="store_true", help="whether to work with FDA")
parser.add_argument("--nreferences", type=int, help="whether to work with histogram correction")
parser.add_argument('--referencerawdata', help='path to the directory with the train/val/test folders')
parser.add_argument('--referencepatchingdescrdir', help='path to the directory with the .csv description file')
parser.add_argument('--referencedescriptionfile', help='description file for the dataset')
parser.add_argument("--beta", default=0.01, type=float, help="beta for FDA")


def load_checkpoint(model, args, device_id):
    highest_epoch = -1
    for file in os.listdir(args.modelpath):
        if file.endswith(".pt"):
            file_base = file.strip(".pt")
            epoch = int(file_base)
            highest_epoch = max(epoch, highest_epoch)
    if highest_epoch == -1:
        raise Exception("Didn't find any checkpoint file. Continue with new model.")
    else:
        print("Using checkpoint from epoch {}.".format(highest_epoch))
    model_path = os.path.join(args.modelpath, str(highest_epoch) + ".pt")
    state_dict = torch.load(model_path, map_location="cuda:{}".format(device_id))
    model.load_state_dict(state_dict["model_state_dict"])
    return model, state_dict["epoch"]


def worker_fct(encoder, args):
    image_dim = 224

    # we should always normalize the input with the correct mean and std.
    temp_df = pd.read_csv(args.tempfilename, index_col=0)
    args.mean = (temp_df.iloc[0]["Mean_r"], temp_df.iloc[0]["Mean_g"], temp_df.iloc[0]["Mean_b"])
    args.std = (temp_df.iloc[0]["Std_r"], temp_df.iloc[0]["Std_g"], temp_df.iloc[0]["Std_b"])
    print("Will use mean of {} and std of {} for normalization.".format(args.mean, args.std))

    if args.histograms:
        ex1 = HistogramMatcher.get_examples(args.nreferences, args.referencedescriptionfile,
                                            args.referencepatchingdescrdir, args.referencerawdata)
        ex2 = HistogramMatcher.get_examples(args.nreferences, args.descriptionfile,
                                            args.patchingdescrdir, args.rawdata)
        augmentations = [HistogramMatcher.HistogramMatcher(ex1, ex2)]
    elif args.fda:
        ex1 = HistogramMatcher.get_examples(args.nreferences, args.referencedescriptionfile, args.referencepatchingdescrdir, args.referencerawdata)
        augmentations = [FDAMatcher.FDAMatcher(ex1, args.beta)]
    else:
        augmentations = []

    augmentations.extend(
        [transforms.CenterCrop(image_dim), transforms.ToTensor(), transforms.Normalize(args.mean, args.std)])
    transform = transforms.Compose(augmentations)

    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=10))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    n_gpus = torch.cuda.device_count()
    device_id = rank % n_gpus
    print("Initialized worker process on rank {}/{} and GPU {}/{}".format(rank + 1, world_size, device_id + 1, n_gpus))

    if args.encodertyp == "simsiam":
        encoder.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder.model)
    encoder.model = encoder.model.to(device_id)
    encoder.model = DistDatPar(encoder.model, device_ids=[device_id])

    if args.encodertyp == "simsiam":
        encoder.model, simsiam_epoch = load_checkpoint(encoder.model, args, device_id)
        print("Loaded Checkpoint on rank {}/{} and GPU {}/{} with model being on GPU {}".format(rank + 1,
                                                                                                world_size,
                                                                                                device_id + 1,
                                                                                                n_gpus,
                                                                                                encoder.model.device))

    encoder.model.eval()

    description_file = pd.read_csv(args.descriptionfile)

    for idx in description_file.index.values:
        if idx % world_size == rank:
            time1 = time.perf_counter()
            slideset = description_file.loc[idx]["Set"]
            slidename = description_file.loc[idx]["Path"]
            slidebase = ".".join(slidename.split(".")[:-1])
            csv_description_name = slidebase + "_patches.csv"

            csv_path = os.path.join(args.patchingdescrdir, csv_description_name)
            total_raw_path = os.path.join(args.rawdata, slideset, slidebase)
            dataset = LazyHistoDataset.H5SlideDataset(csv_file=csv_path, filebase=slidebase,
                                                      raw_directory=total_raw_path, directory_encoded=args.encodeddir,
                                                      transform=transform)
            numenc = encode_slide(encoder, device_id, dataset, args)
            time2 = time.perf_counter()
            print("Slide-idx.: {}/{}\t\tRank: {}\tNum. tiles: {}\t\tRuntime: {}\t\tSlide: {}".format(
                idx, description_file.shape[0], rank, numenc, time2 - time1, slidename))
            del slidename
            gc.collect()
    dist.barrier()


def encode_slide(encoder: Encoder, device_id: int, dataset: LazyHistoDataset.H5SlideDataset, args: argparse.Namespace):
    bag_transformed = []
    loader = data.DataLoader(dataset, batch_size=args.prepbatchsize, num_workers=args.workers,
                             drop_last=False)
    for i, (_, _, imgs) in enumerate(loader):
        if args.nhwc:
            imgs = imgs.to(memory_format=torch.channels_last)
        imgs = imgs.cuda(device_id, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            imgs = encoder.forward(imgs)
        imgs = imgs.cpu().detach().numpy().astype(np.float32)
        bag_transformed.append(imgs)
    bag_transformed = np.vstack(bag_transformed)
    dataset.store(bag_transformed)
    return len(bag_transformed)


def main():
    args = parser.parse_args()
    # create target path for encoded files, if it does not exist already
    Path(args.encodeddir).mkdir(parents=True, exist_ok=True)
    model = Encoder(args)
    if args.nhwc:
        model = model.to(memory_format=torch.channels_last)
    print("Encoded images will have {} features".format(model.outputsize))
    worker_fct(model, args)


if __name__ == '__main__':
    main()

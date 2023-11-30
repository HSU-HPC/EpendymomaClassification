import argparse
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DistDatPar
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

import LARC

import CustomAugmentations
import CustomSimSiam

from LazyHistoDataset import SlowH5Set

parser = argparse.ArgumentParser()
# general config
parser.add_argument('data', metavar='DIR', help='path to dataset/patches')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency in minibatches (default: 10)')
parser.add_argument('--checkpoint-freq', default=50, type=int, help='checkpoint frequency (default: 50).')
parser.add_argument('--workers', default=None, type=int, help='Number of workers for data preprocessing')
parser.add_argument('--logfile', default="log.csv", type=str, help='File for logging')
parser.add_argument('--modelpath', default="/models", type=str, help="path to models for checkpointing")

# learning specific
parser.add_argument('--num', default=50, type=int, help='num. layers in resnet model')
parser.add_argument('--epochs', default=100, type=int, help='number of total ')
parser.add_argument('--batch-size', default=512, type=int, help='mini-batch size ')
parser.add_argument('--learning-rate', default=0.05, type=float, help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument("--warmup", default=0, type=int, help="number of epochs for warmup")

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int, help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int, help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true', help='Fix learning rate for the predictor')
parser.add_argument('--pdl', '--prediction-layers', default=2, type=int, metavar='PD',
                    help='layers in prediction MLP', dest='pdl')
parser.add_argument('--pjl', '--projection-layers', default=3, type=int, metavar='PJ',
                    help='layers in projection MLP', dest='pjl')

# misc.
parser.add_argument("--normestim", default=25, type=int, help="number of batches to estimate mean and std")
parser.add_argument("--resume", default=False, type=bool, help="number of batches to estimate mean and std")
parser.add_argument("--tempfilename", default="temp.csv", type=str, help="tempfile used for communication")
parser.add_argument("--amp", action="store_true", help="whether to use AMP (automatic mixed precision)")
parser.add_argument("--nhwc", action="store_true", help="whether to use channel last")
parser.add_argument("--larc", action="store_true", help="whether to use LARC optimizer")
parser.add_argument("--hf", action="store_true", help="whether to use h5 files")
parser.add_argument("--trusteta", type=float, default=0.001, help="trust coefficient eta for the lars optimizer")
parser.add_argument("--workertempfilename", default="tempworkers.csv", type=str,
                    help="tempfile used for communication of the optimal number of workers")


def training_dummy(model: CustomSimSiam.SimSiam, dataloader: torch.utils.data.DataLoader, criterion: callable,
                   device_id: int, args: argparse.Namespace, rank: int, rep: int = 5):
    """
    Returns the theoretical training time (dataloading + forward pass + backward pass) for the model with the prescribed
    number of workers
    :param rank: the rank
    :param args: Namespace of cmd-arguments
    :param device_id: the local device id
    :param criterion: the loss function to minnimize
    :param model: the encoder (here: CustomSimSiam)
    :param dataloader: the dataloader to use
    :param rep: the number of epochs to train
    :return: timediff
    """
    dist.barrier()  # make sure that all processes start at the same time
    time_start = time.perf_counter()  # start time measurement
    for currep in range(rep):
        for i, ((aug1, aug2), _) in enumerate(dataloader):
            if args.nhwc:
                aug1 = aug1.to(memory_format=torch.channels_last)
                aug2 = aug2.to(memory_format=torch.channels_last)
            aug1 = aug1.cuda(device_id, non_blocking=True)  # move to GPU
            aug2 = aug2.cuda(device_id, non_blocking=True)  # move to GPU
            # if args.amp, we'l use autocast to perform some of the computations in half-precision
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                p1, p2, z1, z2 = model(aug1_batch=aug1, aug2_batch=aug2)  # forward pass
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5  # compute loss
            if rank == 0 and i % 25 == 0:
                print("Rank 0, Workers {}, Batch {}/{}, Current Repetition {}".format(dataloader.num_workers, i,
                                                                                      len(dataloader), currep))
            loss.backward()  # backward pass
    dist.barrier()  # make process wait, until everything is finished
    time_end = time.perf_counter()  # stop time measurement
    return time_end - time_start


def get_optimal_num_workers(train_dataset: torch.utils.data.Dataset,
                            world_size: int, rank: int, device_id: int,
                            model: CustomSimSiam.SimSiam, args: argparse.Namespace, criterion: callable,
                            min_workers: int = 2,
                            max_workers: int = 10, step: int = 1):
    """
    Determines the optimal number of workers for fast training
    by measuring the training time on 1 epoch
    :param criterion: criterion to optimize
    :param args: cmd-arguments
    :param model: the model to train
    :param device_id: the device id for this rank
    :param rank: this rank
    :param world_size: the world size
    :param train_dataset: the dataset
    :param min_workers: minimal number of workers
    :param max_workers: maximal number of workers
    :param step: step size
    :return: best worker
    """
    # lists to store pairs of workers- and timing information
    worker_list = []
    time_list = []

    # we don't need the entire big dataset for estimating runtime --> only the first 10% of the images
    train_subset = torch.utils.data.Subset(train_dataset, np.arange(int(len(train_dataset) * 0.1)))
    # construct respective distributed sampler
    train_subset_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)

    # 1st epoch should be excluded, since cudn.benchmark will select the correct algorithm
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size,
                                               num_workers=(min_workers + max_workers) // 2, pin_memory=True,
                                               drop_last=True,
                                               sampler=train_subset_sampler, persistent_workers=True,
                                               worker_init_fn=seed_worker)
    training_dummy(model, train_loader, criterion, device_id, args, rank, rep=1)  # train for the 1st time

    # iterate over valid range of workers
    for workers in range(min_workers, max_workers + 1, step):
        # append corresponding worker
        worker_list.append(workers)
        # create corresponding data loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=workers, pin_memory=True, drop_last=True,
                                                   sampler=train_subset_sampler, persistent_workers=True,
                                                   worker_init_fn=seed_worker)
        # time for 1 epoch
        workertime = training_dummy(model, train_loader, criterion, device_id, args, rank)
        # append measured time
        time_list.append(workertime)
        # print recorded time with rank 0
        if rank == 0:
            print(
                "Number of workers: {} \t\tTraining time for 5 epochs on 10% of the dataset: {}".format(
                    workers, round(workertime, 3)))
    # rank 0 stores the results, the results from the other processes are ignored
    if rank == 0 and not os.path.exists(args.workertempfilename):
        worker_df = pd.DataFrame({"workers": worker_list, "timeperepoch": time_list})
        worker_df.to_csv(args.workertempfilename)
    return


def seed_worker(worker_id):
    """
    Seed data loader workers individually. Function from the official pytorch documentation at
    https://pytorch.org/docs/stable/notes/randomness.html
    :param worker_id: ignored
    :return: None
    """
    worker_seed = torch.initial_seed() % 2 ** 32  # seed for external libraries in this worker
    np.random.seed(worker_seed)  # seed numpy
    random.seed(worker_seed)  # seed python's default RNG


def worker_fct(model, args):
    """
    The main worker function.
    * sets up the distributed computations
    * optionally loads a checkpoint
    * loads all pre-computed parameters, such as for normalization and the data
    :param model: model
    :param args: the cmd arguments
    :return: nothing
    """

    ####################################################################################################################
    # Distributed Setup + Setup
    ####################################################################################################################
    # initializes the distributed group, with nccl backend
    dist.init_process_group("nccl")
    # rank of the current process
    rank = dist.get_rank()
    # number of processes participating in the job
    world_size = dist.get_world_size()
    # devices available on this node
    n_gpus = torch.cuda.device_count()
    # the device id for this process
    device_id = rank % n_gpus
    print("Initialized worker process on rank {}/{} and GPU {}/{}".format(rank + 1, world_size, device_id + 1, n_gpus))
    # print cmd-arguments for debugging
    if rank == 0:
        print(args)
    # convert to sync batch norm --> over all mini-batches of the process group
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # move the model to the gpu
    model = model.to(device_id)
    # make use of DDP
    model = DistDatPar(model, device_ids=[device_id])
    # construct the optimizer on each rank
    optimizer = build_optimizer(model, args)
    start_epoch = 0  # the start epoch, might be altered later on, if we start from a checkpoint
    # gradient scaler to avoid floor-to-zero problems with AMP
    # will become no-ops when args.amp is False
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    # if we're supposed to look for old checkpoints
    if args.resume:
        print("Loading checkpoint")
        model, optimizer, scaler, start_epoch = load_checkpoint(model, optimizer, scaler, args, device_id)
        # verify here, that the each worker process loads the model to the respective gpu.
        # Following the DDP documentation, we would have needed to specify a map_location.
        # We see, however, that this is not necessary, if the script has been restarted (model.device is
        # in {0,1} for our 2 GPU system).
        print("Finished loading checkpoint on rank {}/{} and GPU {}/{} with model being on GPU {}".format(rank + 1,
                                                                                                          world_size,
                                                                                                          device_id + 1,
                                                                                                          n_gpus,
                                                                                                          model.device))
    print("Finished building/loading model on rank {}/{} and GPU {}/{}".format(rank + 1, world_size, device_id + 1,
                                                                               n_gpus))

    ####################################################################################################################
    # Augmentation setup
    ####################################################################################################################
    # histo-datasets can also be rotated and flipped vertically
    blur = True
    verticalflip = True
    rotate = verticalflip
    image_dim = 224
    # compute means/stds required for normalization, if the values are not already logged somewhere
    if not os.path.exists(args.tempfilename):
        args.mean, args.std = get_norm_values(args, image_dim, verbose=(rank == 0))
        print("Estimated mean of {} and std of {} on rank {}/{}".format(args.mean, args.std, rank + 1, world_size))
    # store the results from rank 0
    if rank == 0 and not os.path.exists(args.tempfilename):
        temp_df = pd.DataFrame({"Mean_r": [args.mean[0]], "Mean_g": [args.mean[1]], "Mean_b": [args.mean[2]],
                                "Std_r": [args.std[0]], "Std_g": [args.std[1]], "Std_b": [args.std[2]]})
        temp_df.to_csv(args.tempfilename)
    dist.barrier()  # make sure that all processes are here (in esp. wait for rank 0 to write out the results)
    # all processes now read the results from rank 0
    temp_df = pd.read_csv(args.tempfilename, index_col=0)
    args.mean = (temp_df.iloc[0]["Mean_r"], temp_df.iloc[0]["Mean_g"], temp_df.iloc[0]["Mean_b"])
    args.std = (temp_df.iloc[0]["Std_r"], temp_df.iloc[0]["Std_g"], temp_df.iloc[0]["Std_b"])
    print("After communication, got mean of {} and std of {} on rank {}/{}".format(args.mean, args.std, rank + 1,
                                                                                   world_size))

    # compose transformations
    transform = CustomAugmentations.get_twoway_transforms(blur=blur, verticalflip=verticalflip, rotate=rotate,
                                                          mean=args.mean, std=args.std, image_dim=image_dim)

    ####################################################################################################################
    # Data loading setup
    ####################################################################################################################
    # we want to distribute the data evenly between the processes --> compute local batch size
    args.batch_size = int(args.batch_size / world_size)
    if args.hf:
        train_dataset = SlowH5Set(root=args.data, transform=transform)
    else:
        train_dataset = ImageFolder(root=args.data, transform=transform)

    # build datasampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    cudnn.benchmark = True  # set benchmarking to 0. Will slow down the first epoch of training
    criterion = nn.CosineSimilarity(dim=1).cuda(device_id)  # define loss as in the simsiam publication

    if args.workers is None and not os.path.exists(args.workertempfilename):
        # determine optimal number of workers
        if rank == 0:
            print("args.workers was not set.")
            print("Determining optimal nuber of workers.")
        # iterate over the possible number of workers and write down the respective training time per epoch
        get_optimal_num_workers(train_dataset, world_size, rank, device_id, model, args, criterion)
        optimizer.zero_grad()  # just to be sure, that the loss is not accumulated from the function above
        # now we can load the worker file in the next step
    if args.workers is None and os.path.exists(args.workertempfilename):
        worker_df = pd.read_csv(args.workertempfilename, index_col=0)
        best_index = worker_df.idxmin()["timeperepoch"]  # get index with minimal runtime
        args.workers = int(worker_df.iloc[best_index]["workers"])  # overwrite number of workers to use
    print("After communication, chose to start {} worker processes per process".format(args.workers))

    # construct data loader.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.workers, pin_memory=True, drop_last=True,
                                               sampler=train_sampler, persistent_workers=True,
                                               worker_init_fn=seed_worker)
    ####################################################################################################################
    # Training
    ####################################################################################################################
    dist.barrier()  # wait until all processes can start with training
    if rank == 0:
        # print some supplementary information
        print("Dataset contains {} images in total.".format(len(train_dataset)))
        print("This corresponds to {} minibatches.".format(len(train_dataset) // (args.batch_size * world_size)))
    # iterate over the epochs, starting at 1 and ending at args.epoch
    for epoch in range(start_epoch + 1, args.epochs + 1):
        time1 = np.array([time.perf_counter()])  # starting the epoch
        losses = np.zeros(len(train_loader))  # storing the losses
        # update the epoch counter for the train loader, so that shuffling works properly
        train_loader.batch_sampler.sampler.set_epoch(epoch)
        # adjust the learning rate (linear warmup and cosine schedule afterwards)
        if args.larc:
            adjust_learning_rate(optimizer.optim, args.init_lr, epoch, args, args.warmup)
        else:
            adjust_learning_rate(optimizer, args.init_lr, epoch, args, args.warmup)
        model.train()  # go to training mode
        for i, ((aug1, aug2), _) in enumerate(
                train_loader):  # iterate over dataloader, returns 2 differently augmented imgs.
            if args.nhwc:
                aug1 = aug1.to(memory_format=torch.channels_last)
                aug2 = aug2.to(memory_format=torch.channels_last)
            # move to GPU
            aug1 = aug1.cuda(device_id, non_blocking=True)
            aug2 = aug2.cuda(device_id, non_blocking=True)
            # if amp enabled, compute forward pass and loss in mixed precision, otherwise in normal precision
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                p1, p2, z1, z2 = model(aug1_batch=aug1, aug2_batch=aug2)  # forward pass
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5  # compute loss
            scaler.scale(loss).backward()  # Scales loss. Calls backward() on scaled loss to create scaled gradients
            scaler.step(optimizer)  # go one optimization step
            scaler.update()  # update the scaler
            optimizer.zero_grad()  # set to zero
            # store intermediate losses from rank 0
            if rank == 0:
                losses[i] = loss.cpu().detach().numpy()
                if i % 100 == 0:
                    time2 = np.array([time.perf_counter()])
                    print("Finished minibatch {} after {} seconds".format(i, time2 - time1))
        # wait for the other processes to stop training for this epoch as well --> important for timing
        dist.barrier()
        time2 = np.array([time.perf_counter()])
        if rank == 0:
            print("Finished epoch {}".format(epoch))
            # rank 0 creates a checkpoint every args-checkpoint_freq epochs
            if epoch % args.checkpoint_freq == 0:
                create_checkpoint(model, optimizer, scaler, epoch, args)
            results = pd.DataFrame({"Epoch": [epoch], "Training Time": time2 - time1, "Loss": np.mean(losses)})
            results.to_csv(args.logfile, header=not os.path.exists(args.logfile), mode="a")
        dist.barrier()  # the other processes wait for rank 0
    # wait until other
    dist.barrier()
    train_dataset.close()


def create_checkpoint(model: CustomSimSiam.SimSiam, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                      epoch: int, args: argparse.Namespace):
    """
    Creates a checkpoint file with the paameters of the model, the optimizer, the gradient scaler and the current
    epoch
    :param model: the model
    :param optimizer: the optimizer
    :param scaler: the gradient scaler
    :param epoch: the epoch
    :param args: the cmd arguments
    :return: nothing
    """
    # create state dict
    state_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  "scaler_state_dict": scaler.state_dict()}
    # construct checkpoint path using the epoch as filename
    model_path = os.path.join(args.modelpath, str(epoch) + ".pt")
    # store
    torch.save(state_dict, model_path)


def load_checkpoint(model: CustomSimSiam.SimSiam, optim: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                    args: argparse.Namespace, device_id: int):
    """
    Loads checkpointed parameters of the simsiam encoder, the optimizer and the gradientscaler into the respective
    objects and returns them
    :param device_id: device id to load the checkpoint to
    :param model: simsiam model
    :param optim: the optimizer
    :param scaler: the scaler
    :param args: cmd arguments
    :return: the updated objects and the epoch from which the checkpoint originates
    """

    # lets find the highest epoch with an available checkpoint first
    highest_epoch = -1  # temp variable
    for file in os.listdir(args.modelpath):  # iterate over all files
        if file.endswith(".pt"):  # it it is a checkpoint file
            # determine epoch from the name of the file
            file_base = file.strip(".pt")
            epoch = int(file_base)
            highest_epoch = max(epoch, highest_epoch)  # update
    # if the value as not changed, there were no checkpoint files
    if highest_epoch == -1:
        print("Didn't find any checkpoint file. Continue with new model.")
        return model, optim, scaler, 0
    # assemble path to model
    model_path = os.path.join(args.modelpath, str(highest_epoch) + ".pt")
    state_dict = torch.load(model_path, map_location="cuda:{}".format(device_id))  # load it
    model.load_state_dict(state_dict["model_state_dict"])  # load parameters into model
    optim.load_state_dict(state_dict["optimizer_state_dict"])  # loda parameters into optimizer
    # the if-statement is only to load some older checkpoints, checkpoints from recent evaluations will
    # always contain a scaler_state_dict key
    if "scaler_state_dict" in state_dict:
        scaler.load_state_dict(state_dict["scaler_state_dict"])  # load into scaler
    return model, optim, scaler, state_dict["epoch"]


def build_model(args: argparse.Namespace):
    """
    Build a new SimSiam model
    :param args: cmd arguments
    :return: the model
    """
    if args.num == 18:
        encoder = models.resnet18
    elif args.num == 34:
        encoder = models.resnet34
    elif args.num == 50:
        encoder = models.resnet50
    elif args.num == 101:
        encoder = models.resnet101
    elif args.num == 152:
        encoder = models.resnet152
    else:
        raise ValueError("Unknown architecture")
    # create model
    model = CustomSimSiam.SimSiam(encoder=encoder, projection_dim=args.dim, prediction_dim=args.pred_dim,
                                  projection_layers=args.pjl, prediction_layers=args.pdl,
                                  cifarresnet=False)
    return model


def build_optimizer(model: CustomSimSiam.SimSiam, args: argparse.Namespace):
    """
    Construct the optimizer
    :param model: the simsiam model
    :param args: the command-line arguments
    :return: sgd-optimizer
    """
    if args.fix_pred_lr:
        # construct parameter groups
        # the resnet-encoder and the projection head are both called encoder in the simsiam
        # paper. Therefore, they both have the variable learning rate
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.projection_mlp.parameters(), 'fix_lr': False},
                        {'params': model.module.prediction_mlp.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()
    # make optimizer
    optimizer = torch.optim.SGD(optim_params, args.init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # wrap LARC class around the optimizer
    if args.larc:
        optimizer = LARC.LARC(optimizer, trust_coefficient=args.trusteta)
    return optimizer


def main():
    # parse cmd arguments
    args = parser.parse_args()
    # increase learning rate proportional to batch size
    args.init_lr = args.lr * args.batch_size / 256
    # construct the model (random weights, any potential checkpoint will be loaded later on)
    model = build_model(args)
    if args.nhwc:
        model = model.to(memory_format=torch.channels_last)
    # call the main worker function
    worker_fct(model, args)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, init_lr: float, epoch: int, args: argparse.Namespace,
                         warm_up_k: int = 0):
    """
    Decay the learning with a cosine schedule after a linear warmup, as described in SIMSIAM publication
    """

    if epoch >= warm_up_k:
        # cosine schedule over epoch-warmup_k epochs
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warm_up_k) / (args.epochs - warm_up_k)))
    else:  # the first epochs (1 - warmup_k-1), we need to linearly increase the learning rate
        # at epoch warmup_k, we would have lr=init_lr, hich can also be obtained using the
        # cosine schedule above
        cur_lr = init_lr / warm_up_k * epoch
    # update learning rate for all parameter-groups with variable learning rates
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def get_norm_values(args: argparse.Namespace, image_dim: int, verbose: bool = False):
    """
    Approximate and return the mean and standard deviation per color channel within a few batches
    We'll only use the training set for this
    :param args: cmd arguments
    :param image_dim: size of the image (m x m)
    :param verbose: whether to print suppl. information
    :return: two lists, one wth the means and one with the standard deviations
    """
    # we need to make sure that the images have the specified size
    # and convert it to a tensor to work with
    transform = transforms.Compose([transforms.CenterCrop(image_dim), transforms.ToTensor()])
    if args.hf:
        train_dataset = SlowH5Set(root=args.data, transform=transform)
    else:
        train_dataset = ImageFolder(root=args.data, transform=transform)
    # construct train loader. Batches will be drawn at random
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, drop_last=True,
                                               num_workers=7, shuffle=True)
    # we'll take as many batches as specified, or (if that is not possible) the number if batches in the dataset
    max_batches = min(len(train_loader), args.normestim)
    # get mean
    if verbose:
        print("Computing mean for normalization...")
    cur_sum = None  # the sum
    batch_counter = 0  # the number of batches
    # iterate over the data loader
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.float()  # convert to float, because we want to take the mean later on
        # we permute the dimensions, so that the channels are in the last dim.
        images = torch.permute(images, dims=(0, 2, 3, 1))
        if cur_sum is None:  # first batch
            cur_sum = images.mean(dim=(0, 1, 2))  # reduce over batch size and all pixels
        else:  # later batches
            cur_sum += images.mean(dim=(0, 1, 2))  # reduce over batch size and all pixels
        batch_counter += 1
        # print some information, if required
        if batch_counter % (max_batches // 10) == 0 and verbose:
            print("Mean: Completed batch {}/{}".format(batch_counter, max_batches))
        if batch_counter == max_batches:  # break if the maximum number of batches is reached
            break
    # average over all the batches
    mean = cur_sum / batch_counter

    # get std
    if verbose:
        print("Computing std for normalization...")
    cur_sum = None  # for storing the quadratic deviations
    batch_counter = 0  # counting the batches
    # iterate over train loader
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.float()
        images = torch.permute(images, dims=(0, 2, 3, 1))
        if cur_sum is None:  # if first batch
            # compute quadratic deviation and average over batch size and pixels
            # subtaction will be broadcasted according to broadcasting rules
            # at https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
            cur_sum = torch.mean((mean - images) ** 2, dim=(0, 1, 2))
        else:  # otherwise
            # compute quadratic deviation and average over batch size and pixels
            cur_sum += torch.mean((mean - images) ** 2, dim=(0, 1, 2))
        batch_counter += 1  # increment counter

        if batch_counter % (max_batches // 10) == 0 and verbose:
            print("Std: Completed batch {}/{}".format(batch_counter, max_batches))
        if batch_counter == max_batches:  # break if max. unmber of batches is reached
            break

    cur_sum = cur_sum / batch_counter  # average quad. deviations over batches
    std = torch.sqrt(cur_sum)  # take square-root (want std, not variance)
    print(mean.tolist(), std.tolist())
    train_dataset.close()
    return mean.tolist(), std.tolist()  # return as lists


if __name__ == '__main__':
    main()  # call main function

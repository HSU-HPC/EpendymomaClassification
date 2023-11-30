import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import utils.loggingutils as lg
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
import random


def seed_worker(worker_id: int):
    """
    Set the seeds for worker processes. Function taken from
    https://pytorch.org/docs/stable/notes/randomness.html.
    This function should be unnecessary here - I just added it to be on the safe side.
    :param worker_id: the worker id --> ignored
    :return: nothing
    """
    worker_seed = torch.initial_seed() % 2 ** 32  # compute the seed for numpy/random-lib. that we want to use here
    np.random.seed(worker_seed)  # set numpy seed
    random.seed(worker_seed)  # set random seed


def store_clam(model: nn.Module, path: str, jobid: int, rep: int):
    """
    Store the current CLAM model. Counterpart of main.load_clam
    :param model: the model to save
    :param path: the directory where we store the model
    :param jobid: the SLURM-jobid
    :param rep: the current repetition
    :return: nothing
    """
    # construct the path
    path = os.path.join(path, "clam_model_{}_{}.pt".format(jobid, rep))
    # and store the model
    torch.save(model.state_dict(), path)


class EarlyStoppingCallBack:
    """
    Class to perform early stopping --> if the validation loss has not improved over the last N epochs, stop training
    """

    def __init__(self, patience: int = 10, minnumepochs: int = 50, path: str = "", rep: int = 0):
        """
        Construct new instance
        :param patience: the nuber of epochs to wait until resetting to the last stored state
        :param minnumepochs: the minimum number of epochs to train
        :param path: directory to store the current, best model
        :param rep: the current repetition
        """
        # store parameters as fields
        self.patience = patience
        self.rep = rep  # the repetition in this SLURM job
        self.path = path
        self.minnumepochs = minnumepochs
        self.jobid = os.environ["SLURM_JOB_ID"]  # jobid. Used for naming the checkpoint file
        self.counter = 0  # number of epochs that we already waited for the model to improve
        self.lowest = np.inf  # current best validation loss
        self.best_epoch = -1  # current best epoch

    def __call__(self, valloss: float, model: nn.Module, epoch: int):
        """
        Check, whether early stopping criterion is fullfilled
        :param valloss: the current validation loss
        :param model: the current clam model
        :param epoch: the current epoch
        :return: whether to stop
        """
        if valloss < self.lowest:  # validation loss improved
            self.best_epoch = epoch  # we have a new best epoch
            self.lowest = valloss  # and a new best loss
            self.counter = 0  # reset counter
            store_clam(model, self.path, self.jobid, self.rep)  # store current, best model
            return False  # don't stop now
        else:  # it didn't improve
            self.counter += 1  # we've waited another epooch
            # did we wait for long enough (or even longer?) and did the minimal number of epoch pass?
            if self.counter >= self.patience and epoch >= self.minnumepochs:
                # epoch index starts at 0 --> at epoch index 50, we will have trained for 51 epochs
                print("After training epoch {}, early stopping triggered. Best epoch is {}".format(epoch,
                                                                                                   self.best_epoch))
                return True  # stop early


def train(train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, model: nn.Module,
          learning_rate: float, weight_decay: float, max_epochs: int, device: str, bag_weight: float = 0.7,
          workers: int = 20, patience: int = 10, minnumepochs: int = 50, balance: bool = False, path: str = "",
          rep: int = 0):
    """
    Train the model.
    :param train_dataset: the training set
    :param val_dataset: the validation set
    :param model: the model to train
    :param learning_rate: the learning rate
    :param weight_decay: the weight_decay for the optimizer
    :param max_epochs: maximum number of epochs to train
    :param device: the device to use. Either "cpu" or "cuda"
    :param bag_weight: the weight to assign to the bag loss
    :param workers: the number of workers to use for data loading
    :param patience: patience for early stopping
    :param minnumepochs: minimum number of epochs to train, before early stopping may be triggered
    :param balance: whether to counter class imbalance by drawing bags with a corresponding probability (and classes
    with a lof frequency twice or even more often)
    :param path: directory to store the clam models
    :param rep: number of repetitions
    :return: nothing
    """
    # move the model to the device
    model = model.to(device)
    # retrieve the slurm id
    jobid = os.environ["SLURM_JOB_ID"]
    # construct optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # for reproducibility, we create two generators
    gen = torch.Generator()  # one for the random sampling (not used, if random sampling is disabled)
    gen.manual_seed(torch.initial_seed())
    worker_gen = torch.Generator()  # generator to create the worker seeds
    worker_gen.manual_seed(torch.initial_seed())

    if balance:  # if we want to use weighted random sampling to counter cöass imbalance
        print("Weighted random sampling of training set to counter class imbalance")
        # random sampler. Replacement=True, so that samples from the minority class can be sampled multiple times
        # should make sure, that the model sees on average the same number of samples per class
        sampler = WeightedRandomSampler(weights=train_dataset.get_sampling_probabilities(),
                                        num_samples=len(train_dataset), replacement=True,
                                        generator=gen)
        # the corresponding training loader
        train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, worker_init_fn=seed_worker,
                                  num_workers=workers, generator=worker_gen)
    else:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=workers,
                                  worker_init_fn=seed_worker, generator=worker_gen)
        # otherwise build a normal training loader

    # construct a validation loader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=workers, generator=worker_gen)
    # construct the early stopper
    stopper = EarlyStoppingCallBack(patience=patience, minnumepochs=minnumepochs, path=path, rep=rep)

    # initialize loggers for each considered metric (accuracy, loss, AUC, cross-entropy)
    # file names contain the SLURM jobid and the repetition, so that we can distinguish them afterwards
    accuracy_logger = lg.ProgressMeter("log_acc_{}_{}.csv".format(jobid, rep), name="Accuracy")
    loss_logger = lg.ProgressMeter("log_loss_{}_{}.csv".format(jobid, rep), name="Loss")
    auc_logger = lg.ProgressMeter("log_auc_{}_{}.csv".format(jobid, rep), name="AUC")
    ce_logger = lg.ProgressMeter("log_ce_{}_{}.csv".format(jobid, rep), name="CE")

    # construct bag-level loss and move it to the cpu
    bag_loss_fn = nn.CrossEntropyLoss()
    bag_loss_fn.to(device)

    # training loop. epoch starts at 0
    for epoch in range(max_epochs):
        # set model to training mode --> necessary, because the validate-function will set it to eval mode
        model.train()

        # auxilliary lists
        losses = []  # list for the combined loss per bag (bag-level + instance-level)
        predictions = []  # the predictions (hard thresholded --> larger score per class sets predicted class)
        targets = []  # the true class
        scores = []  # the respective scores that the model assigned

        for bag_idx, (filenames, bag, bag_label) in enumerate(train_loader):
            bag = bag.squeeze().to(device)  # 1xnxm --> nxm
            bag_label = bag_label.to(device)
            results = model(bag, bag_label)  # forward pass
            # cross-entropy for the scores
            bag_loss = bag_loss_fn(results["logits"], bag_label)
            # compute the total loss
            total_loss = bag_weight * bag_loss + (1 - bag_weight) * results["instance_loss"]

            optimizer.zero_grad()  # set gradient to zero
            total_loss.backward()  # backward pass
            optimizer.step()  # adapt parameters

            predictions.append(results["y_class"])  # the respective hard-thresholded predictions
            targets.append(bag_label)  # the true bag label
            scores.append(results["y_proba"].cpu().detach().numpy()[0])  # the corresponding, normalized score

            losses.append(total_loss.item())  # append the combined loss for this bag to the list with losses

        # format the lists --> trim unneccessary dimensions, make numpy array etc.
        predictions = torch.cat(predictions).squeeze()
        targets = torch.cat(targets).squeeze()
        scores = np.array(scores)
        # compute AUC score on the training dataset
        auc_train = roc_auc_score(targets.cpu().detach().numpy(), scores[:, 1])  # myxo corresponds to positive class
        # compute crossentropy on the training set
        crossentropyloss_train = log_loss(targets.cpu().detach().numpy(), scores)

        # compute all the metrics on the vlidation set
        val_loss, val_acc, val_auc, correct_str, val_crossentropyloss, val_df, val_at = validate(model, val_loader,
                                                                                                 device,
                                                                                                 bag_weight,
                                                                                                 bag_loss_fn)
        train_loss = np.mean(losses)  # mean of the combined losses per bag
        train_acc = (targets == predictions).sum().item() / predictions.shape[0]  # accuracy on the training set
        # store training and validation results
        accuracy_logger.pushback(epoch, train_acc, val_acc, display=True)
        loss_logger.pushback(epoch, train_loss, val_loss, display=True)
        auc_logger.pushback(epoch, auc_train, val_auc, display=True)
        ce_logger.pushback(epoch, crossentropyloss_train, val_crossentropyloss, display=True)
        # save the results
        accuracy_logger.save()
        loss_logger.save()
        auc_logger.save()
        ce_logger.save()

        if stopper(val_loss, model, epoch):  # check, whether early stopping criterion is fullfilled
            # remove the epochs that were trained, but did not yield any improvements anymore
            accuracy_logger.removeNLastEpochs(stopper.counter)
            loss_logger.removeNLastEpochs(stopper.counter)
            auc_logger.removeNLastEpochs(stopper.counter)
            ce_logger.removeNLastEpochs(stopper.counter)
            accuracy_logger.save()
            loss_logger.save()
            auc_logger.save()
            ce_logger.save()
            break

    return


def validate(model: nn.Module, val_loader: torch.utils.data.DataLoader, device: str, bag_weight: float,
             bag_loss_fn: callable):
    """
    Validate the current model
    :param model: the current model
    :param val_loader: the loader for the dataset to use
    :param device: the device to use. Either "cpu" or "cuda"
    :param bag_weight: the weight to assign to the bag-level loss
    :param bag_loss_fn: the loss function
    :return:
    """
    model.eval()  # set model to evaluation mode
    # lists for storing the results
    losses = []  # the losses
    targets = []  # the true classes
    files = []  # the file names of the predicted bags (potentially multiple slides per patient)
    predictions = []  # the hard-thresholded predictions
    scores = []  # the scores
    logits = []  # the raw scores prior to softmax
    attentions0 = []
    attentions1 = []

    # we don't need any gradient for evaluation. Should be unnecessary --> added to be on safe side
    with torch.no_grad():
        for bag_idx, (file, bag, bag_label) in enumerate(val_loader):
            files.append(file)  # store files
            bag = bag.squeeze().to(device)  # 1xmxn --> mxn
            bag_label = bag_label.to(device)  # target to device, if not already
            results = model(bag, bag_label)  # the evaluation pass

            bag_loss = bag_loss_fn(results["logits"], bag_label)  # bag loss
            total_loss = bag_weight * bag_loss + (1 - bag_weight) * results["instance_loss"]  # total loss
            predictions.append(results["y_class"])  # hard-thresholded predictions
            targets.append(bag_label)  # append true class
            losses.append(total_loss.item())  # append the loss for this bag
            scores.append(results["y_proba"].cpu().detach().numpy()[0])  # append the probability estimate
            logits.append(results["logits"].cpu().detach().numpy()[0])

            attentions0.append(list(results["raw_attention"].cpu().detach().numpy()[0]))
            if results["raw_attention"].cpu().detach().numpy().shape[0] == 2:
                attentions1.append(list(results["raw_attention"].cpu().detach().numpy()[1]))
            else:
                attentions1.append(list(results["raw_attention"].cpu().detach().numpy()[0]))

    # make all the attention scores have the same length, by filling with nans
    max_length = max([max([len(lst) for lst in attentions0]), max([len(lst) for lst in attentions1])])
    attentions0 = np.array([lst + [np.NaN] * (max_length - len(lst)) for lst in attentions0])
    attentions1 = np.array([lst + [np.NaN] * (max_length - len(lst)) for lst in attentions1])
    # make dataframe
    attentions0 = pd.DataFrame(attentions0)
    attentions0["Branch"] = 0
    attentions0["File"] = files
    attentions1 = pd.DataFrame(attentions1)
    attentions1["Branch"] = 1
    attentions1["File"] = files

    attentions = pd.concat([attentions0, attentions1], axis=0)
    # reformat lists (strip unnecessary dimensions, make numpy array, etc.)
    predictions = torch.cat(predictions).squeeze()
    targets = torch.cat(targets).squeeze()
    scores = np.array(scores)
    logits = np.array(logits)
    predictions_np = predictions.cpu().detach().numpy()
    targets_np = targets.cpu().detach().numpy()

    # number of correctly predicted samples from class 0 (spinal)
    class_0_correct = np.sum(predictions_np[np.nonzero(targets_np == 0)[0]] == 0)
    # number of correctly predicted samples from class 1 (myxo)
    class_1_correct = np.sum(predictions_np[np.nonzero(targets_np == 1)[0]] == 1)

    correct_str = "Class 0: {}/{}\t\tClass 1: {}/{}".format(class_0_correct, len(np.nonzero(targets_np == 0)[0]),
                                                            class_1_correct, len(np.nonzero(targets_np == 1)[0]))

    crossentropyloss = log_loss(targets.cpu().detach().numpy(), scores)
    auc = roc_auc_score(targets.cpu().detach().numpy(), scores[:, 1])  # myxo corresponds to positive class

    # construct a dataframe with the results
    df = pd.DataFrame({"File": files, "Predictions": predictions, "targets": targets,
                       "Probabilities Class 0": scores[:, 0], "Probabilities Class 1": scores[:, 1],
                       "Score Class 0": logits[:, 0], "Score Class 1": logits[:, 1]})

    return np.mean(losses), (targets == predictions).sum().item() / predictions.shape[
        0], auc, correct_str, crossentropyloss, df, attentions


def create_attention_plots(model: nn.Module, dataset: torch.utils.data.Dataset, directory: str, workers: int,
                           imsize: int, downsample: int = 64, tile_downsample: int = 4, device: str = "cpu",
                           n_highest: int = 50, filename_path: str = None):
    """
    Creates attention plots (or calls the respective method, to be more precise)
    :param model: the model to use
    :param dataset: the dataset to use
    :param directory: the directory to store the plots into
    :param workers: the number of workers to load the data
    :param imsize: the size of the extracted images
    :param downsample: at which downsample should we plot the images
    :param tile_downsample: at which downsample have the tiles been extracted
    :param device: the device to use
    :param n_highest: if the coordinates of the patches with the highest attention will be saved, then this number of
    patches will be considered (per attention branch)
    :param filename_path: if this is not None, then the n_highest patches with the highest attention will be stored to this file
    :return:
    """

    assert dataset.attention_plots  # needs to return the x,y tuples

    # the data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
    # set to evaluation mode
    model.eval()

    attention_dfs = []

    with torch.no_grad():
        # iterate over dataset
        for bag_idx, (files, bag, bag_label, x_values, y_values) in enumerate(data_loader):
            file = files[0]  # files is an array of length 1 --> retrieve the element

            bag = bag.squeeze().to(device)  # 1xmxn --> mxn
            bag_label = bag_label.to(device)  # the bag
            attention = model(bag, bag_label, attention_only=True)  # return the unnormalized attentions

            # make everything a numpy array
            attention = attention.detach().numpy()
            x_values = x_values.squeeze().numpy()
            y_values = y_values.squeeze().numpy()

            # construct the filename to store the plot into
            filename = file[0].split("/")[-1][:-11] + "_attention.pdf"
            # create the plot
            df = plot_attention(x_values, y_values, attention, directory, filename, imsize, downsample, tile_downsample)
            attention_dfs.append(df)
    total_df = pd.concat(attention_dfs, axis=0)
    if filename_path is not None:
        total_df.to_csv(filename_path + ".csv")


def plot_attention(xv: np.ndarray, yv: np.ndarray, av: np.ndarray, dir: str, name: str, imsize: int,
                   downsample: int = 64, tile_downsample: int = 4, n_highest=50):
    # determine area to plot
    x_min = np.min(xv)
    x_max = np.max(xv)
    y_min = np.min(yv)
    y_max = np.max(yv)
    # the matrix to plot. The last axis corresponds to different attention paths
    matrix = np.zeros(
        (int((x_max - x_min) // downsample) + imsize // 4, int((y_max - y_min) // downsample) + imsize // 4,
         av.shape[0]))
    # normalize each individual attention output to be between 0 and 1
    av = (av - np.min(av, axis=(1,), keepdims=True)) / (
            np.max(av, axis=(1,), keepdims=True) - np.min(av, axis=(1,), keepdims=True))
    # add the corresponding attention to the plot (the tiles may have overlap)
    for idx, (x, y) in enumerate(zip(xv, yv)):
        # x_pixels covered by this path
        x_pixels = np.arange(int((x - x_min) // downsample),
                             int(((x - x_min)) // downsample) + imsize * tile_downsample // downsample)
        # y_pixels covered by the patch
        y_pixels = np.arange(int((y - y_min) // downsample),
                             int(((y - y_min)) // downsample) + imsize * tile_downsample // downsample)
        # add the corresponding attention
        matrix[np.ix_(x_pixels, y_pixels)] += av[:, idx]

    # make the attention plots --> one heatmap per attention branch
    cols = av.shape[0]  # if attention has shape n_classes x n_instances, we have multibranching
    fig, axs = plt.subplots(ncols=cols)
    if cols > 1:
        for branch_idx in range(cols):
            axs[branch_idx].matshow(matrix[:, :, branch_idx])
            axs[branch_idx].set_xticks([])
            axs[branch_idx].set_yticks([])
            axs[branch_idx].set_title("Attention Branch {}".format(branch_idx))
    else:
        axs.matshow(matrix[:, :, 0])
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_title("Global Attention Map")
    save_path = os.path.join(dir, name)
    plt.savefig(save_path)
    plt.close()

    # now the n_highest most strongly attended patches per branch
    branch_list = []
    x_list = []
    y_list = []
    attention_list = []
    name_list = []
    for branch_idx in range(cols):
        best_indices = np.argpartition(av[branch_idx], -n_highest)[-n_highest:]
        x_list.extend(list(xv[best_indices]))
        y_list.extend(list(yv[best_indices]))
        name_list.extend([name for _ in best_indices])
        branch_list.extend([branch_idx for _ in best_indices])
        attention_list.extend(list(av[branch_idx][best_indices]))

    df = pd.DataFrame(
        {"File": name_list, "Branch": branch_list, "x-Val.": x_list, "y-Val": y_list, "Attention": attention_list})
    return df

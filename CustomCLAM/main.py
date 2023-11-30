import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import random

import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DistDatPar
import torch.nn as nn

import CLAM
import utils.LazyHistoDataset as LazyHistoDataset
from utils.clam_training import train, validate, create_attention_plots

parser = argparse.ArgumentParser()
# general config
parser.add_argument('--data', action="append", help='path to the csv description files of the tiles')
parser.add_argument('--descriptionfile', help='description file for the dataset')
parser.add_argument('--encodeddir', action="append", help='path to the pre-encoded dataset')
parser.add_argument('--numrep', default=5, type=int, help="number of training/evaluation repetitions")
parser.add_argument('--workers', default=8, type=int, help='no. of workers used for preprocessing and data loading')
parser.add_argument('--evaldescriptionfile', help='description file for the external dataset')
parser.add_argument('--evalencodeddir', action="append", help='path to the pre-encoded, external dataset')
parser.add_argument('--evaldata', action="append",
                    help='path to the csv description files of the tiles for the external validation cohort')

# multi resolution
parser.add_argument('--multires', action="store_true", help="whether to use multi-resolution input")
parser.add_argument('--concat', action="store_true", help="whether to use concatenate concentric multi-resolution "
                                                          "patches into a single instance")

# CLAM specific
parser.add_argument('--ninst', default=32, type=int, help="number of instances (patches) used for clustering")
parser.add_argument('--bagweight', default=0.7, type=float, help="weight of bag classification to loss")
parser.add_argument('--subtyping', default=False, type=bool,
                    help="whether to use subtyping (mutually exclusive classes)")
parser.add_argument("--clampath", default="./", type=str, help="path to folder for storing the CLAM model")

#
# learning specific
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--learningrate', default=0.0002, type=float, help='learning rate', dest='lr')
parser.add_argument('--weightdecay', default=0.00001, type=float, help='weight decay',
                    dest='weight_decay')
parser.add_argument('--patience', default=10, type=int,
                    help='patience for early stopping. Set  >epochs to disable early stopping')
parser.add_argument('--minnumepochs', default=50, type=int,
                    help='minimal number of epochs before early stopping is permitted')
parser.add_argument('--balanced', default=False, type=bool,
                    help='whether to account for class imbalance by multinomial sampling probabilities')
parser.add_argument('--classifypatients', default=False, type=bool, help='Whether to assign class labels to patients'
                                                                         '(True) or to slides (False)')
# misc.
parser.add_argument('--normalize', action="store_true", help="whether to normalize SimSiams output before CLAM")
parser.add_argument('--pathresults', default=None, type=str,
                    help="path to the results (attention plots for each slide and"
                         "classification results for each slide)")
parser.add_argument("--patchattention", default=None, type=str, help="if not None, will store the patches with highest"
                                                                     "attention to {filename}_{val/test}.csv")
parser.add_argument('--amdownsample', default=64, type=int, help="downsample factor for the attention maps")
parser.add_argument('--tiledownsample', default=4, type=int, help="downsample factor for the individual tile")
parser.add_argument('--imagedim', default=224, type=int, help="the size of the tiles")
parser.add_argument("--modelid", default=os.environ["SLURM_JOB_ID"], type=int, help="the slurm id to use for storing/"
                                                                                    "loading models")
parser.add_argument('--encoderdim', default=2048, type=int, help="number of features per encoded tile")
parser.add_argument('--multimb', action="store_true", help="whether to use the single-branch implementation of CLAM")
parser.add_argument("--instancedropout", action="store_true", help="whether to use instance dropout using training")
parser.add_argument("--cache", action="store_true", help="whether to cache the datasets during training.")
parser.add_argument("--eval", action="store_true", help="whether to evaluate with external dataset as well")

# for reproducibility
SEED = 2434162255
torch.manual_seed(SEED)  # pytorch seed
np.random.seed(SEED)  # numpy seed (just in case)
random.seed(SEED)  # python base seed (just in case)
torch.use_deterministic_algorithms(True, warn_only=True)  # make pytorch use deterministic algorithms only


def load_clam(model, args, rep):
    """
    Load CLAM-Model from checkpoint.
    :param model: the model to load the checkpoint into
    :param args: the cmd arguments
    :param rep: the repetition to load (if we wxecute the training multiple times)
    :return: the checkpointed model
    """
    # construct path to checkpoint
    path = os.path.join(args.clampath, "clam_model_{}_{}.pt".format(args.modelid, rep))
    # load the model --> is the strict counterpart of clam_training.store_clam
    model.load_state_dict(torch.load(path))
    return model


def worker_fct(args):
    """
    The worker function that executes all training and evaluation
    :param args: the cmd arguments
    :return: nothing
    """

    final_dict_patientlevel = {"test_acc": [], "validation_acc": [], "test_loss": [], "validation_loss": [],
                               "test_auc": [], "validation_auc": [], "test_ce": [], "validation_ce": []}
    final_dict_slidelevel = {"test_acc": [], "validation_acc": [], "test_loss": [], "validation_loss": [],
                             "test_auc": [], "validation_auc": [], "test_ce": [], "validation_ce": []}
    eval_dict_patientlevel = {"Accuracy": [], "Loss": [], "AUC": [], "CE": []}
    eval_dict_slidelevel = {"Accuracy": [], "Loss": [], "AUC": [], "CE": []}

    # train
    for rep in range(args.numrep):
        # construct the model
        clam_model = CLAM.CLAM(encoded_dim=args.encoderdim, input_dim_attention=args.encoderdim // 2,
                               projection_dim_attention=args.encoderdim // 4, dropout=True, n_classes=2,
                               n_inst=args.ninst, multi_branch=args.multimb, subtyping=args.subtyping)
        # train
        train_clam(args, clam_model, rep)
        # load model from this training iteration
        clam_model = load_clam(clam_model, args, rep)
        # set to evaluation mode
        clam_model.eval()

        # create the bag loss and move it to cpu
        bag_loss_fn = nn.CrossEntropyLoss()
        bag_loss_fn.to("cpu")
        # evaluate the model from the current repetition on validation and test set using patient-level results
        patientlevel_results = evaluate_clam(args, bag_loss_fn, clam_model, True)
        if args.eval:
            # eval_loss, eval_accuracy, eval_auc, eval_str, eval_ce, eval_df
            eval_dict_patientlevel["Accuracy"].append(patientlevel_results["Eval"][1])
            eval_dict_patientlevel["Loss"].append(patientlevel_results["Eval"][0])
            eval_dict_patientlevel["AUC"].append(patientlevel_results["Eval"][2])
            eval_dict_patientlevel["CE"].append(patientlevel_results["Eval"][4])

        # append to lists
        final_dict_patientlevel["test_acc"].append(patientlevel_results["Test"][1])
        final_dict_patientlevel["validation_acc"].append(patientlevel_results["Validation"][1])
        final_dict_patientlevel["test_loss"].append(patientlevel_results["Test"][0])
        final_dict_patientlevel["validation_loss"].append(patientlevel_results["Validation"][0])
        final_dict_patientlevel["test_auc"].append(patientlevel_results["Test"][2])
        final_dict_patientlevel["validation_auc"].append(patientlevel_results["Validation"][2])
        final_dict_patientlevel["test_ce"].append(patientlevel_results["Test"][4])
        final_dict_patientlevel["validation_ce"].append(patientlevel_results["Validation"][4])
        # construct paths for storing the results
        rep_path = os.path.join(args.pathresults, str(rep))
        Path(rep_path).mkdir(parents=True, exist_ok=True)

        # and store
        patientlevel_results["Validation"][5].to_csv(os.path.join(
            rep_path, "predictions_validation_patientlevel_{}_{}.csv".format(args.modelid, rep)))
        patientlevel_results["Validation"][6].to_csv(os.path.join(
            rep_path, "attentions_validation_patientlevel_{}_{}.csv".format(args.modelid, rep)))
        patientlevel_results["Test"][5].to_csv(os.path.join(
            rep_path, "predictions_test_patientlevel_{}_{}.csv".format(args.modelid, rep)))
        patientlevel_results["Test"][6].to_csv(os.path.join(
            rep_path, "attentions_test_patientlevel_{}_{}.csv".format(args.modelid, rep)))

        if args.eval:
            patientlevel_results["Eval"][5].to_csv(os.path.join(
                rep_path, "predictions_eval_patientlevel_{}_{}.csv".format(args.modelid, rep)))
            patientlevel_results["Eval"][6].to_csv(os.path.join(
                rep_path, "attentions_eval_patientlevel_{}_{}.csv".format(args.modelid, rep)))
        # eval_loss, eval_accuracy, eval_auc, eval_str, eval_ce, eval_df
        print("=================================================================================")
        print("                              Patient-Level Results                              ")
        print("=================================================================================")
        print("Validation loss:\t\t{}".format(patientlevel_results["Validation"][0]))
        print("Test loss:\t\t{}".format(patientlevel_results["Test"][0]))
        print("Validation Acc.:\t\t{}".format(patientlevel_results["Validation"][1]))
        print("Test Acc.:\t\t{}".format(patientlevel_results["Test"][1]))
        print("Validation AUC:\t\t{}".format(patientlevel_results["Validation"][2]))
        print("Test AUC:\t\t{}".format(patientlevel_results["Test"][2]))
        print("Validation Correct:\t\t" + patientlevel_results["Validation"][3])
        print("Test Correct:\t\t" + patientlevel_results["Test"][3])
        print("Validation CE:\t\t{}".format(patientlevel_results["Validation"][4]))
        print("Test CE:\t\t{}".format(patientlevel_results["Test"][4]))
        if args.eval:
            print("Eval Acc.:\t\t{}".format(patientlevel_results["Eval"][1]))
        print("=================================================================================")

        # evaluate on slide-level
        slidelevel_results = evaluate_clam(args, bag_loss_fn, clam_model, False)

        if args.eval:
            # eval_loss, eval_accuracy, eval_auc, eval_str, eval_ce, eval_df
            eval_dict_slidelevel["Accuracy"].append(slidelevel_results["Eval"][1])
            eval_dict_slidelevel["Loss"].append(slidelevel_results["Eval"][0])
            eval_dict_slidelevel["AUC"].append(slidelevel_results["Eval"][2])
            eval_dict_slidelevel["CE"].append(slidelevel_results["Eval"][4])

        # append to lists
        final_dict_slidelevel["test_acc"].append(slidelevel_results["Test"][1])
        final_dict_slidelevel["validation_acc"].append(slidelevel_results["Validation"][1])
        final_dict_slidelevel["test_loss"].append(slidelevel_results["Test"][0])
        final_dict_slidelevel["validation_loss"].append(slidelevel_results["Validation"][0])
        final_dict_slidelevel["test_auc"].append(slidelevel_results["Test"][2])
        final_dict_slidelevel["validation_auc"].append(slidelevel_results["Validation"][2])
        final_dict_slidelevel["test_ce"].append(slidelevel_results["Test"][4])
        final_dict_slidelevel["validation_ce"].append(slidelevel_results["Validation"][4])

        # and store
        slidelevel_results["Validation"][5].to_csv(os.path.join(
            rep_path, "predictions_validation_slidelevel_{}_{}.csv".format(args.modelid, rep)))
        slidelevel_results["Validation"][6].to_csv(os.path.join(
            rep_path, "attentions_validation_slidetlevel_{}_{}.csv".format(args.modelid, rep)))
        slidelevel_results["Test"][5].to_csv(os.path.join(
            rep_path, "predictions_test_slidetlevel_{}_{}.csv".format(args.modelid, rep)))
        slidelevel_results["Test"][6].to_csv(os.path.join(
            rep_path, "attentions_test_slidelevel_{}_{}.csv".format(args.modelid, rep)))

        if args.eval:
            slidelevel_results["Eval"][5].to_csv(os.path.join(
                rep_path, "predictions_eval_patientlevel_{}_{}.csv".format(args.modelid, rep)))
            slidelevel_results["Eval"][6].to_csv(os.path.join(
                rep_path, "attentions_eval_slidelevel_{}_{}.csv".format(args.modelid, rep)))

        # eval_loss, eval_accuracy, eval_auc, eval_str, eval_ce, eval_df
        print("=================================================================================")
        print("                              Slide-Level Results                              ")
        print("=================================================================================")
        print("Validation loss:\t\t{}".format(slidelevel_results["Validation"][0]))
        print("Test loss:\t\t{}".format(slidelevel_results["Test"][0]))
        print("Validation Acc.:\t\t{}".format(slidelevel_results["Validation"][1]))
        print("Test Acc.:\t\t{}".format(slidelevel_results["Test"][1]))
        print("Validation AUC:\t\t{}".format(slidelevel_results["Validation"][2]))
        print("Test AUC:\t\t{}".format(slidelevel_results["Test"][2]))
        print("Validation Correct:\t\t" + slidelevel_results["Validation"][3])
        print("Test Correct:\t\t" + slidelevel_results["Test"][3])
        print("Validation CE:\t\t{}".format(slidelevel_results["Validation"][4]))
        print("Test CE:\t\t{}".format(slidelevel_results["Test"][4]))
        if args.eval:
            print("Eval Acc.:\t\t{}".format(slidelevel_results["Eval"][1]))
        print("=================================================================================")

        # now the attention maps, if we're working on a single-resolution model
        if not args.multires:
            attentionmaps(args, clam_model, rep)

        patient_df = pd.DataFrame(final_dict_patientlevel)
        slide_df = pd.DataFrame(final_dict_slidelevel)
        eval_patient_df = pd.DataFrame(eval_dict_patientlevel)
        eval_slide_df = pd.DataFrame(eval_dict_slidelevel)

        patient_df.to_csv(os.path.join(args.pathresults, "patientlevel_{}.csv".format(args.modelid)))
        eval_patient_df.to_csv(os.path.join(args.pathresults, "eval_patientlevel_{}.csv".format(args.modelid)))
        slide_df.to_csv(os.path.join(args.pathresults, "slidelevel_{}.csv".format(args.modelid)))
        eval_slide_df.to_csv(os.path.join(args.pathresults, "eval_slidelevel_{}.csv".format(args.modelid)))

    print("Finishing computations.")


def attentionmaps(args: argparse.Namespace, clam_model: nn.Module, rep: int):
    """
    Constructs datasets and calls the method to draw the attention maps
    :param args: the cmd arguments
    :param clam_model: the (trained) clam model to use
    :param rep: the repetition
    :return: nothing
    """

    # create path
    rep_path = os.path.join(args.pathresults, str(rep))
    Path(rep_path).mkdir(parents=True, exist_ok=True)
    # validation dataset, containing one bag per slide
    val_dataset = LazyHistoDataset.H5SlideDataset(description_file=args.descriptionfile,
                                                  path_encoded_files=args.encodeddir,
                                                  patches_dir=args.data,
                                                  considered_set=["val"],
                                                  patient_bags=False,
                                                  normalize=args.normalize,
                                                  attention_plots=True)
    # make the corresponding plots
    create_attention_plots(clam_model, val_dataset, rep_path, args.workers, args.imagedim,
                           downsample=args.amdownsample, tile_downsample=args.tiledownsample, device="cpu",
                           filename_path=args.patchattention + "val" if args.patchattention is not None else None)
    # construct the test set
    test_dataset = LazyHistoDataset.H5SlideDataset(description_file=args.descriptionfile,
                                                   path_encoded_files=args.encodeddir,
                                                   patches_dir=args.data,
                                                   considered_set=["test"],
                                                   patient_bags=False,
                                                   normalize=args.normalize,
                                                   attention_plots=True)
    # make the corresponding plots
    create_attention_plots(clam_model, test_dataset, rep_path, args.workers, args.imagedim,
                           downsample=args.amdownsample, tile_downsample=args.tiledownsample, device="cpu",
                           filename_path=args.patchattention + "_test" if args.patchattention is not None else None)


def evaluate_clam(args: argparse.Namespace, bag_loss_fn: callable, clam_model: nn.Module, patientbags: bool):
    """
    Evaluate a trained clam model on the validation, and the test set
    :param args: the cmd arguments
    :param bag_loss_fn: the function for the bag loss
    :param clam_model: the model
    :param patientbags: whether to combine all slides per patient into one bag
    :return: Nothing
    """
    # construct the validation dataset
    if not args.multires:
        val_dataset = LazyHistoDataset.H5SlideDataset(description_file=args.descriptionfile,
                                                      path_encoded_files=args.encodeddir,
                                                      patches_dir=args.data,
                                                      considered_set=["val"],
                                                      patient_bags=patientbags,
                                                      normalize=args.normalize)
    else:
        val_dataset = LazyHistoDataset.MultiResolutionDataset(description_file=args.descriptionfile,
                                                              path_encoded_files=args.encodeddir,
                                                              patches_dir=args.data,
                                                              considered_set=["val"],
                                                              patient_bags=patientbags,
                                                              normalize=args.normalize,
                                                              concat=args.concat)
    # construct corresponding dataloader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    # the test dataset
    if not args.multires:
        test_dataset = LazyHistoDataset.H5SlideDataset(description_file=args.descriptionfile,
                                                       path_encoded_files=args.encodeddir,
                                                       patches_dir=args.data,
                                                       considered_set=["test"],
                                                       patient_bags=patientbags,
                                                       normalize=args.normalize)
    else:
        test_dataset = LazyHistoDataset.MultiResolutionDataset(description_file=args.descriptionfile,
                                                               path_encoded_files=args.encodeddir,
                                                               patches_dir=args.data,
                                                               considered_set=["test"],
                                                               patient_bags=patientbags,
                                                               normalize=args.normalize,
                                                               concat=args.concat)
    # if we're supposed to evaluate on the external validation cohort
    if args.eval:
        eval_dataset = LazyHistoDataset.H5SlideDataset(description_file=args.evaldescriptionfile,
                                                       path_encoded_files=args.evalencodeddir,
                                                       patches_dir=args.evaldata,
                                                       considered_set=["Eval"],
                                                       patient_bags=patientbags,
                                                       normalize=args.normalize)
        # corresponding eval loader
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    # corresponding data loader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    # validate on validation dataset
    validation_loss, validation_accuracy, validation_auc, validation_str, validation_ce, validation_df, validation_attention = validate(
        clam_model, val_loader, "cpu", args.bagweight, bag_loss_fn)
    validation_results = (
    validation_loss, validation_accuracy, validation_auc, validation_str, validation_ce, validation_df,
    validation_attention)
    # validate on test set
    test_loss, test_accuracy, test_auc, test_str, test_ce, test_df, test_attention = validate(clam_model, test_loader,
                                                                                              "cpu",
                                                                                              args.bagweight,
                                                                                              bag_loss_fn)
    test_results = (test_loss, test_accuracy, test_auc, test_str, test_ce, test_df, test_attention)

    results = {"Validation": validation_results, "Test": test_results}
    # validate on external cohort, if required
    if args.eval:
        eval_loss, eval_accuracy, eval_auc, eval_str, eval_ce, eval_df, eval_attention = validate(clam_model,
                                                                                                  eval_loader, "cpu",
                                                                                                  args.bagweight,
                                                                                                  bag_loss_fn)
        eval_results = (eval_loss, eval_accuracy, eval_auc, eval_str, eval_ce, eval_df, eval_attention)
        results["Eval"] = eval_results

    return results


def train_clam(args, clam_model, rep):
    """
    Train the clam model
    :param args: cmd arguments
    :param clam_model: the model to train
    :param rep: the current repetition
    :return: nothing
    """

    if not args.multires:
        # the training dataset
        train_dataset = LazyHistoDataset.H5SlideDataset(description_file=args.descriptionfile,
                                                        path_encoded_files=args.encodeddir,
                                                        patches_dir=args.data,
                                                        considered_set=["train"],
                                                        patient_bags=args.classifypatients,
                                                        normalize=args.normalize)
        # the validation dataset --> should be patient-level, since this is going to be the final evaluation metric
        val_dataset = LazyHistoDataset.H5SlideDataset(description_file=args.descriptionfile,
                                                      path_encoded_files=args.encodeddir,
                                                      patches_dir=args.data,
                                                      considered_set=["val"],
                                                      patient_bags=True,
                                                      normalize=args.normalize)
    else:
        train_dataset = LazyHistoDataset.MultiResolutionDataset(description_file=args.descriptionfile,
                                                                path_encoded_files=args.encodeddir,
                                                                patches_dir=args.data,
                                                                considered_set=["train"],
                                                                patient_bags=args.classifypatients,
                                                                normalize=args.normalize,
                                                                concat=args.concat,
                                                                instance_dropout=args.instancedropout,
                                                                cache=args.cache)

        # should also be patient level, since this is going to be the final evaluation metric
        val_dataset = LazyHistoDataset.MultiResolutionDataset(description_file=args.descriptionfile,
                                                              path_encoded_files=args.encodeddir,
                                                              patches_dir=args.data,
                                                              considered_set=["val"],
                                                              patient_bags=True,
                                                              normalize=args.normalize,
                                                              concat=args.concat)

    # perform the training
    train(train_dataset, val_dataset, clam_model, args.lr, args.weight_decay, args.epochs, "cpu", args.bagweight,
          workers=args.workers, patience=args.patience, minnumepochs=args.minnumepochs, balance=args.balanced,
          path=args.clampath, rep=rep)


def main():
    # parse the cmd arguments and call the worker function
    args = parser.parse_args()
    print(args)
    # if we work with a single resolution, we don't need the directories as lists
    if not args.multires:
        args.data = args.data[0]
        args.encodeddir = args.encodeddir[0]
        if args.eval:
            args.evaldata = args.evaldata[0]
            args.evalencodeddir = args.evalencodeddir[0]
    worker_fct(args)


if __name__ == '__main__':
    main()

import os
from sys import platform

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import torch.nn as nn


class H5SlideDataset(Dataset):
    """
    Dataset class to load the hdf5 slides. Either loads individual slides or all slides from a patient.
    LazyHistoDataset.H5SlideDataset(description_file=args.descriptionfile,
                                                        path_encoded_files=args.encodeddir,
                                                        considered_set=["train"],
                                                        patient_bags=args.classifypatients, normalize=args.normalize)
    """

    def __init__(self, description_file: str, path_encoded_files: str, patches_dir: str,
                 considered_set: list = ["train", "train_dump"], patient_bags: bool = False, normalize: bool = False,
                 attention_plots: bool = False, cache: bool = False):
        """
        Create new instance
        :param description_file: path to .csv-file with the annotations (train/test/val)
        :param path_encoded_files: path to the directory with the encoded files
        :param considered_set: set to consider. May be any combination of the values in description_file
        :param patient_bags: boolean. Whether to combine multiple slides into one bag, if they come from the same patient.
        :param normalize: Whether to normalize the encoded tiles.
        :param attention_plots: Whether to create attention plots. Then, we'll need to return x,y-values for each tile as well.
        :param cache: Whether to cache the tensors instead of loading from file. Should accelerate training,
        but num_workers might need to be adjusted
        """
        super(H5SlideDataset, self).__init__()

        if normalize:
            print("Normalizing tiles.")
        if patient_bags:
            print("Working with patients")
            if attention_plots:
                raise ValueError("Working with patients may not be combined with attention_plots")
        if attention_plots:
            print("Will return positions and slidenames")
        if cache:
            print("Will cache tensors.")

        # store the parameters
        self.attention_plots = attention_plots
        self.patient_bags = patient_bags
        self.normalize = normalize
        self.path_encoded_files = path_encoded_files
        self.patches_dir = patches_dir
        self.enable_caching = cache
        self.cache = {}

        self.decription_file = pd.read_csv(description_file)  # read description file
        # drop everything except the considered set
        self.decription_file = self.decription_file[self.decription_file["Set"].isin(considered_set)]
        # replace ordinal classes numeric values
        self.decription_file["Class"] = self.decription_file["Class"].replace({"myxo": 1, "spinal": 0})

        self.h5_names = []  # path names to the hdf5 files for the encoded slides
        self.tiling_files = []  # the corresponding dataframes describing the individual, cropped tiles
        self.diagnoses = []  # patient/slide-wise diagnoses

        # iterate over the slides in the sub-dataset
        for slide_idx, slide_name in enumerate(self.decription_file["Path"].values):
            slide_base = ".".join(slide_name.split(".")[:-1])  # trim file ending
            # corresponding name of the hdf5-file with the encoded tiles
            encoded_filename = slide_base + "_patches_encoded.h5"
            # corresponding name of the hdf5-file with the description of cropped tiles
            tiling_file_name = slide_base + "_patches.csv"
            # both files should exist
            assert os.path.exists(os.path.join(self.path_encoded_files, encoded_filename))
            assert os.path.exists(os.path.join(self.patches_dir, tiling_file_name))
            # append hdf5-file-path with the encoded slides
            self.h5_names.append(os.path.join(self.path_encoded_files, encoded_filename))
            # corresponding dataframe (not path!!) with the descriptions of the cropped tiles
            self.tiling_files.append(pd.read_csv(os.path.join(self.patches_dir, tiling_file_name), index_col=0))
            # corresponding diagnoses
            self.diagnoses.append(self.decription_file.iloc[slide_idx]["Class"])

    def __len__(self):
        """
        :return: Length of the dataset (respective number of slides/patients)
        """
        if self.patient_bags:
            # return number of unique cases
            return len(self.decription_file["Case ID"].unique())
        else:
            # return number of slides in dataset
            return len(self.h5_names)

    def get_sampling_probabilities(self):
        """
        :return: Multinomial sampling probabilities, such that all classes are sampled comparably often
        """
        if self.patient_bags:
            # unique patients
            unique_patients = self.decription_file["Case ID"].unique()
            patients = self.decription_file["Case ID"].values
            # corresponding diagnoses
            classes = []
            for patient in unique_patients:
                patient_idx = np.nonzero(patients == patient)[0][0]  # first occurence
                classes.append(self.decription_file.iloc[patient_idx]["Class"])
            classes = np.array(classes)
        else:
            classes = np.array(self.diagnoses)
        # empty array for the corresponding probabilities
        probas = np.zeros(classes.shape, dtype=np.float64)
        # all unique classes
        vals, counts = np.unique(classes, return_counts=True)
        for v, c in zip(vals, counts):
            # for each diagnoses, the relative probability should be the inverse of the corresponding frequency
            probas[np.nonzero(classes == v)] = 1 / c
        return probas

    def __getitem__(self, idx):
        """
        Return the corresponding bag
        :param idx: index of bag to return
        :return: bag as torch.tensor
        """
        if not self.enable_caching or idx not in self.cache.keys():
            # select all slide-indices corresponding to the patient with index idx
            if self.patient_bags:
                # all unique patients
                unique_patients = self.decription_file["Case ID"].unique()
                # all available patients (including duplicates)
                patients = self.decription_file["Case ID"].values
                # corresponding unique patient
                patient = unique_patients[idx]
                # all slide indices with this unique patient
                slide_indices = np.nonzero(patients == patient)[0]
            else:
                # make array
                slide_indices = [idx]

            # array for the images
            imgs = []
            # arrays for the xy-positions
            x_values = []
            y_values = []
            # corresponding array of filenames with slides from the respective patients
            filenames = []

            # iterate over the slides
            for idx in slide_indices:
                # corresponding description dataframe
                tiling_df = self.tiling_files[idx]
                # open the h5 file
                h5_file = h5py.File(self.h5_names[idx], "r")
                # load corresponding diagnoses
                diagnosis = self.diagnoses[idx]
                # append corresponding filename for this slide
                filenames.append(self.h5_names[idx])

                # iterate over each x,y position in the description file
                for x, y in zip(tiling_df["X"].values, tiling_df["Y"].values):
                    # load the encoded tile
                    imgs.append(torch.from_numpy(h5_file[str(x)][str(y)]['default'][()]))
                    # if we want to create attention plots, we need to have the respective coordinates of the slides
                    if self.attention_plots:
                        x_values.append(x)
                        y_values.append(y)
                h5_file.close()  # need to close, otherwise memory-leakage occurs

            if self.attention_plots:
                x_values = np.array(x_values)
                y_values = np.array(y_values)

            # append all the encoded images to one big bag
            imgs = torch.stack(imgs)

            # if we're supposed to normalize the encoded slides, do so
            if self.normalize:
                imgs = nn.functional.normalize(imgs)

            if self.enable_caching:
                self.cache[idx] = (filenames, imgs, diagnosis, x_values, y_values)
        else:
            filenames, imgs, diagnosis, x_values, y_values = self.cache[idx]

        # return positions, if required
        if self.attention_plots:
            return filenames, imgs, diagnosis, x_values, y_values
        else:  # or not
            return filenames, imgs, diagnosis


class MultiResolutionDataset(Dataset):
    """
    Dataset class to load the hdf5 slides at multiple resolutions. Expects the images to be concentric, with 
    the csv files specifying their center. Either loads individual slides or all slides from a patient.
    """

    def __init__(self, description_file: str, path_encoded_files: list, patches_dir: list,
                 considered_set: list = ["train", "train_dump"], patient_bags: bool = False, normalize: bool = False,
                 concat: bool = False, instance_dropout: bool = False, cache: bool = False):
        """
        Create new instance
        :param patches_dir: list of patches directories
        :param description_file: path to .csv-files with the annotations (train/test/val)
        :param path_encoded_files: list of paths to the directories with the encoded files
        :param considered_set: set to consider. May be any combination of the values in description_file
        :param patient_bags: boolean. Whether to combine multiple slides into one bag, if they come from the same patient.
        :param normalize: Whether to normalize the encoded tiles.
        :param concat : Whether to concatenate encoded images with the same
        :param instance_dropout : whether to use instance dropout (only during training!!)
        :param cache : whether to use caching of the tensors. Num_workers will need to be adjusted accordingly!
        """

        super(MultiResolutionDataset, self).__init__()

        if normalize:
            print("Normalizing tiles.")
        if patient_bags:
            print("Working with patients")
        if instance_dropout:
            print("10% instance droput.")
        if cache:
            print("Will cache tensors.")

        # store the parameters
        self.instance_dropout = instance_dropout
        self.concat = concat
        self.patient_bags = patient_bags
        self.normalize = normalize
        self.path_encoded_files = path_encoded_files
        self.patches_dir = patches_dir
        self.enable_caching = cache
        self.cache = {}

        self.decription_file = pd.read_csv(description_file)  # read description file
        # drop everything except the considered set
        self.decription_file = self.decription_file[self.decription_file["Set"].isin(considered_set)]
        # replace ordinal classes with numeric values
        self.decription_file["Class"] = self.decription_file["Class"].replace({"myxo": 1, "spinal": 0})

        # path names to the hdf5 files at each resolution for the encoded slides
        self.h5_names = [[] for _ in range(len(self.path_encoded_files))]
        # the corresponding dataframes describing the individual, cropped tiles
        self.tiling_files = [[] for _ in range(len(self.patches_dir))]
        self.diagnoses = []  # patient/slide-wise diagnoses

        # iterate over the slides in the sub-dataset
        for slide_idx, slide_name in enumerate(self.decription_file["Path"].values):
            for level_idx, (p_e_f, p_d) in enumerate(zip(self.path_encoded_files, self.patches_dir)):
                slide_base = ".".join(slide_name.split(".")[:-1])  # trim file ending
                # corresponding name of the hdf5-file with the encoded tiles
                encoded_filename = slide_base + "_patches_encoded.h5"
                # corresponding name of the hdf5-file with the description of cropped tiles
                tiling_file_name = slide_base + "_patches.csv"
                # both files should exist
                assert os.path.exists(os.path.join(p_e_f, encoded_filename))
                assert os.path.exists(os.path.join(p_d, tiling_file_name))
                # append hdf5-file-path with the encoded slides
                self.h5_names[level_idx].append(os.path.join(p_e_f, encoded_filename))
                # corresponding dataframe (not path!!) with the descriptions of the cropped tiles
                self.tiling_files[level_idx].append(pd.read_csv(os.path.join(p_d, tiling_file_name), index_col=0))
            # corresponding diagnoses
            self.diagnoses.append(self.decription_file.iloc[slide_idx]["Class"])

    def __len__(self):
        """
        :return: Length of the dataset (respective number of slides/patients)
        """
        if self.patient_bags:
            # return number of unique cases
            return len(self.decription_file["Case ID"].unique())
        else:
            # return number of slides in dataset
            return len(self.h5_names[0])

    def get_sampling_probabilities(self):
        """
        :return: Multinomial sampling probabilities, such that all classes are sampled comparably often
        """
        if self.patient_bags:
            # unique patients
            unique_patients = self.decription_file["Case ID"].unique()
            patients = self.decription_file["Case ID"].values
            # corresponding diagnoses
            classes = []
            for patient in unique_patients:
                patient_idx = np.nonzero(patients == patient)[0][0]  # first occurence
                classes.append(self.decription_file.iloc[patient_idx]["Class"])
            classes = np.array(classes)
        else:
            classes = np.array(self.diagnoses)
        # empty array for the corresponding probabilities
        probas = np.zeros(classes.shape, dtype=np.float64)
        # all unique classes
        vals, counts = np.unique(classes, return_counts=True)
        for v, c in zip(vals, counts):
            # for each diagnoses, the relative probability should be the inverse of the corresponding frequency
            probas[np.nonzero(classes == v)] = 1 / c
        return probas

    def __getitem__(self, idx):
        """
        Return the corresponding bag
        :param idx: index of bag to return
        :return: bag as torch.tensor
        """
        if not self.enable_caching or idx not in self.cache.keys():
            # select all slide-indices corresponding to the patient with index idx
            if self.patient_bags:
                # all unique patients
                unique_patients = self.decription_file["Case ID"].unique()
                # all available patients (including duplicates)
                patients = self.decription_file["Case ID"].values
                # corresponding unique patient
                patient = unique_patients[idx]
                # all slide indices with this unique patient
                slide_indices = np.nonzero(patients == patient)[0]
            else:
                # make array
                slide_indices = [idx]

            # array for the images
            imgs = []
            # corresponding array of filenames with slides from the respective patients
            filenames = []

            # iterate over the slides
            for idx in slide_indices:
                # corresponding description dataframe (one resolution is sufficient, because the datasets should
                # only contain concentric values)
                tiling_df = self.tiling_files[0][idx]
                # open the h5 files
                h5_files = [h5py.File(h5level[idx], "r") for h5level in self.h5_names]
                # load corresponding diagnoses
                diagnosis = self.diagnoses[idx]
                # append corresponding filename for this slide
                filenames.append(self.h5_names[0][idx])

                # iterate over each x,y position in the description file
                for x, y in zip(tiling_df["X"].values, tiling_df["Y"].values):
                    all_res_imgs = []
                    for h5f in h5_files:
                        # load the encoded tile
                        all_res_imgs.append(torch.from_numpy(h5f[str(x)][str(y)]['default'][()]))
                    if self.concat:
                        if self.normalize:
                            all_res_imgs = [a / torch.sqrt(torch.sum(torch.pow(a, 2))) for a in all_res_imgs]
                        all_res_imgs = torch.hstack(all_res_imgs)
                        imgs.append(all_res_imgs)
                    else:
                        imgs.extend(all_res_imgs)
                for h5f in h5_files:
                    h5f.close()  # need to close, otherwise memory-leakage occurs

            # append all the encoded images to one big bag
            imgs = torch.stack(imgs)

            # if we're supposed to normalize the encoded slides, do so
            if self.normalize and not self.concat:
                imgs = nn.functional.normalize(imgs)

            if self.enable_caching:
                self.cache[idx] = (filenames, imgs, diagnosis)
        else:
            filenames, imgs, diagnosis = self.cache[idx]

        if self.instance_dropout:
            num_tiles = imgs.shape[0]
            keep = torch.randperm(num_tiles)[:int(0.9 * num_tiles)]
            imgs = imgs[keep]

        return filenames, imgs, diagnosis

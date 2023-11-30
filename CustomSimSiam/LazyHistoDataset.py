import os
import time
from sys import platform

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# legacy code
if platform == "linux" or platform == "linux2" or platform == "darwin":
    import openslide
elif platform == "win32":
    print("Error")

def compare_performance(num=100):
    """
    Legacy Code
    :param num: number of iterations
    :return: nothing
    """
    rng = np.random.default_rng()
    # setup
    ds_h5 = H5HistoDataset(r'C:\Users\Yannis Schumann\Documents\PhD\HistoClassification\Datasets\HistoExamples\patches')
    ds_openslide = LazyHistoDataset(
        r'C:\Users\Yannis Schumann\Documents\PhD\HistoClassification\Datasets\HistoExamples\patches')
    len_df = len(ds_h5)
    t1 = time.time()
    for _ in range(num):
        img = ds_h5[rng.integers(low=0, high=len_df - 1)]
    t2 = time.time()
    print("On average, H5Dataset took {} ms to load one tile".format(round(1000 * (t2 - t1) / num, 3)))

    t3 = time.time()
    for _ in range(num):
        img = ds_openslide[rng.integers(low=0, high=len_df - 1)]
    t4 = time.time()
    print("On average, LazyHistoDataset took {} ms to load one tile".format(round(1000 * (t4 - t3) / num, 3)))


class SlowH5Set(Dataset):

    def __init__(self, root: str, transform: callable):
        self.root = root
        self.transform = transform

        self.csv_filenames = [x for x in os.listdir(self.root) if x.endswith(".csv")]
        self.csv_descr = pd.concat([pd.read_csv(os.path.join(self.root, x)) for x in self.csv_filenames], axis=0)

        self.h5_filenames = [x[:-4] + ".h5" for x in self.csv_filenames]

        self.pair_dict = {x: h5py.File(os.path.join(self.root, y), "r") for x, y in
                          zip(self.csv_descr["File"].unique(), self.h5_filenames)}

        self.counter = 0

    def open(self):
        self.pair_dict = {x: h5py.File(os.path.join(self.root, y), "r") for x, y in
                          zip(self.csv_descr["File"].unique(), self.h5_filenames)}

    def __len__(self):
        return self.csv_descr.shape[0]

    def __getitem__(self, item):
        self.counter += 1
        if self.counter >= 100000:
            self.close()
            self.open()
            self.counter = 0
        item = self.csv_descr.iloc[item]
        file, x, y = self.pair_dict[item["File"]], item["X"], item["Y"]
        item = Image.fromarray(file[str(x)][str(y)]['default'][()])
        if self.transform:
            item = self.transform(item)
        return item, (x, y)

    def close(self):
        for f in self.pair_dict.values():
            f.close()


class H5HistoDataset(Dataset):
    """
    Dataclass to provide data loading from patches stored in h5-files
    """

    def __init__(self, patches_dir: str, description_file: str, considered_set: list = ["train", "train_dump"],
                 transform: callable = None):
        """
        Create new H5HistoDataset
        :param patches_dir: directory, where the extracted patches are stored in .h5 files
        :param description_file: path to the file that describes the dataset (train/val/test splits)
        :param considered_set: the set (or the sets) that we use here
        :param transform: transformations to apply when loading a sample
        """
        super(H5HistoDataset, self)

        self.descriptor = pd.read_csv(description_file)  # read the file describing the dataset
        self.descriptor = self.descriptor[
            self.descriptor["Set"].isin(considered_set)]  # reduce to the considered subset
        self.found_files = []  # list for h5-file we may use here
        self.h5_filenames = []  # store the paths to the h5 files
        # keep track of the number of retreived images to periodically close and re-open all h5 files
        # every 10 epochs
        self.call_tracker_idx = 0

        dataframes = []  # temporary list for the description dataframes
        h5_files = []  # list with paths to the h5 files
        for file in os.listdir(patches_dir):  # iterate over the directory
            if file.endswith(".csv"):  # this is a description csv-file
                if np.any(self.descriptor["Path"] == (file[:-12] + ".ndpi")):  # if this sample is in the considered set
                    self.found_files.append((file[:-12] + ".ndpi"))  # append the name of the original file to the list
                    file_path = os.path.join(patches_dir,
                                             file)  # full path to description file, required for reading it
                    dataframes.append(pd.read_csv(file_path, index_col=0))  # read the file to dataframe
                    h5_path = file_path[:-4] + ".h5"  # corresponding h5-file path
                    self.h5_filenames.append(h5_path)
                    h5_files.append(h5py.File(h5_path, mode="r"))  # open the h5-file
        self.h5_files = h5_files  # store as field
        self.df = pd.concat(dataframes, axis=0)  # concatenate dataframes, to a long df
        self.tilesize = self.df.iloc[0]["tilesize"]  # read the corresponding size of the tile in pixels per side
        self.level = self.df.iloc[0]["level"]  # the level on which the files were created
        self.transform = transform  # store the transformations
        self.paths = self.df["File"].unique()  # store the number of unique slides

        self.length = self.__len__()  # store length

    def __len__(self):
        return self.df.shape[0]  # number of total tiles

    def __reopen__(self):
        # close all the files
        for file in self.h5_files:
            file.close()
        # store new, open files
        self.h5_files = [h5py.File(path, mode="r") for path in self.h5_filenames]
        # reset counter
        self.call_tracker_idx = 0

    def __getitem__(self, idx):
        # increase call counter
        self.call_tracker_idx += 1
        if self.call_tracker_idx % (10 * self.length) == 0:
            self.__reopen__()

        tile = self.df.iloc[idx]  # the row corresponding to this index
        x = tile["X"]  # the corresponding x-position in the slide image
        y = tile["Y"]  # the corresponding y-position in the slide image
        path = tile["File"]  # the path to the original slide-image

        slide = self.h5_files[np.nonzero(self.paths == path)[0][0]]  # the h5-file storing the raw tiles
        img = Image.fromarray(slide[str(x)][str(y)]['default'][()])  # read PIL image from the stored numpy array
        if self.transform:
            img = self.transform(img)  # apply transformations, if applicable
        return img


class LazyHistoDataset(Dataset):
    """
    Legacy Code.
    """

    def __init__(self, patches_dir, description_file, considered_set=("train", "train_dump"), transform=None):
        super(LazyHistoDataset, self)

        self.descriptor = pd.read_csv(description_file)
        self.descriptor = self.descriptor[self.descriptor["Set"].isin(considered_set)]

        dataframes = []
        for file in os.listdir(patches_dir):
            if file.endswith(".csv"):
                if np.any(self.descriptor["Path"] == file[:-12] + ".ndpi"):
                    file_path = os.path.join(patches_dir, file)
                    dataframes.append(pd.read_csv(file_path, index_col=0))
        self.df = pd.concat(dataframes, axis=0)
        self.tilesize = self.df.iloc[0]["tilesize"]
        self.level = self.df.iloc[0]["level"]
        self.transform = transform
        self.slides = [openslide.open_slide(path) for path in self.df["File"].unique()]
        self.paths = self.df["File"].unique()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        tile = self.df.iloc[idx]
        x = tile["X"]
        y = tile["Y"]
        path = tile["File"]
        slide = self.slides[np.nonzero(self.paths == path)[0][0]]
        img = slide.read_region((x, y), self.level, (self.tilesize, self.tilesize)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

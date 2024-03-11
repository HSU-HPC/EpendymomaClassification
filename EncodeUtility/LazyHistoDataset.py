import os

import h5py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class H5SlideDataset(Dataset):
    """
    Dataset class to load the hdf5 slides individually for encoding
    """

    def __init__(self, csv_file, filebase, raw_directory, directory_encoded, transform=None):
        super(H5SlideDataset, self).__init__()

        self.description = pd.read_csv(csv_file, index_col=0)
        self.raw_directory = raw_directory

        self.directory_encoded = directory_encoded
        self.transform = transform
        self.filebase = filebase

        # read into memory once
        self.x_values = []
        self.y_values = []
        self.imgs = []

        for x, y in zip(self.description["X"].values, self.description["Y"].values):
            image_path = os.path.join(self.raw_directory, "{}_{}.png".format(x, y))
            # load PIL Image
            img = Image.open(image_path)
            # apply transformations
            if self.transform is not None:
                img = self.transform(img)
            # store
            self.imgs.append(img)
            self.x_values.append(x)
            self.y_values.append(y)

        # some length-checking
        assert len(self.imgs) == len(self.x_values)
        assert len(self.imgs) == len(self.y_values)
        assert len(self.imgs) == self.description.shape[0]

    def __len__(self):
        """
        :return: Length of the dataset (respective number of tiles)
        """
        return self.description.shape[0]

    def __getitem__(self, idx):
        return self.x_values[idx], self.y_values[idx], self.imgs[idx]

    def store(self, imgs):
        enc_h5_file = h5py.File(os.path.join(self.directory_encoded, self.filebase + "_patches_encoded.h5"), "w")

        for index, (x, y) in enumerate(zip(self.x_values, self.y_values)):
            img_encoded = imgs[index]
            grp_x = enc_h5_file.require_group(str(x))
            grp_y = grp_x.create_group(str(y))
            grp_y.create_dataset("default", shape=img_encoded.shape, dtype=img_encoded.dtype, data=img_encoded)
        enc_h5_file.close()

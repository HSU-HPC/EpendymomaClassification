import PIL.Image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import LazyHistoDataset
import pandas as pd


class HistogramMatcher:

    def __init__(self, x1: np.ndarray, x2: np.ndarray):
        """
        Computes the transformation from x2 to x1
        """
        # expect x1, x2 to have the same number of color channels
        assert x1.shape[-1] == x2.shape[-1]
        # expect nxhxwxc format and care only about the color channels
        x1 = x1.reshape((-1, 3))
        x2 = x2.reshape((-1, 3))

        # u values for dictionary
        u_vals = np.arange(0, 256, dtype=np.int64)
        # cdf's
        cdf_x1 = np.zeros((256, 3), dtype=np.float64)
        cdf_x2 = np.zeros((256, 3), dtype=np.float64)

        for channel in range(3):
            for u_idx, u in enumerate(u_vals):
                cdf_x1[u_idx, channel] = np.sum(x1[:, channel] <= u) / x1.shape[0]
                cdf_x2[u_idx, channel] = np.sum(x2[:, channel] <= u) / x2.shape[0]

        lut = np.zeros((256, 3), dtype="uint8")
        for channel in range(3):
            for u_idx, u in enumerate(u_vals):
                lut[u_idx, channel] = u_vals[(cdf_x1[:, channel] >= cdf_x2[u_idx, channel]).argmax()]
        self.lut = lut

    def transform(self, x: PIL.Image.Image):
        x = np.asarray(x).copy()
        for channel in range(x.shape[-1]):
            x[..., channel] = self.lut[:, channel][x[..., channel]]
        return Image.fromarray(x)

    def __call__(self, x):
        return self.transform(x)


def get_examples(N, description_file, patchingdescrdir, rawdata):

    description_file = pd.read_csv(description_file)

    to_select = int(N/description_file.shape[0])
    rng = np.random.default_rng()

    imgs = []

    for idx in description_file.index.values:
        slideset = description_file.loc[idx]["Set"]
        slidename = description_file.loc[idx]["Path"]
        slidebase = ".".join(slidename.split(".")[:-1])
        csv_description_name = slidebase + "_patches.csv"
        csv_path = os.path.join(patchingdescrdir, csv_description_name)
        description = pd.read_csv(csv_path, index_col=0)
        raw_directory = os.path.join(rawdata, slideset, slidebase)

        indices = rng.choice(np.arange(description.shape[0]), size=to_select)
        small_description = description.iloc[indices]

        for x, y in zip(small_description["X"].values, small_description["Y"].values):
            image_path = os.path.join(raw_directory, "{}_{}.png".format(x, y))
            # load PIL Image
            img = Image.open(image_path)
            imgs.append(np.asarray(img).copy())
            img.close()

    imgs = np.array(imgs)

    print("Obtained {} images from dataset {}".format(imgs.shape[0], rawdata))

    return imgs

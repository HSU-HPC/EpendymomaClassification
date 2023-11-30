import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import h5py
import shutil
from pathlib import Path


def __isBlack__(patch, threshold):
    return np.all(np.mean(patch, axis=(0, 1)) < threshold)


def __isWhitePatch__(patch, threshold):
    hsv_space = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return np.mean(hsv_space[:, :, 1]) < threshold


class Patcher:

    def __init__(self, slidescan, tile_size, level, stride, black_threshold=80, white_threshold=30,
                 use_black_padding=False, save_path=None, save_style="png", h5_path=None, png_path=None):
        self.slidescan = slidescan
        self.slidename = Path(self.slidescan.path).name
        self.tile_size = tile_size
        self.level = level
        self.stride = stride
        self.black_threshold = black_threshold
        self.white_threshold = white_threshold
        self.save_path = save_path
        self.valid_corner_x_list = []
        self.valid_corner_y_list = []
        self.h5_path = h5_path
        self.h5_file = None
        self.counter = 0
        self.use_black_padding = use_black_padding
        self.store = save_path is not None
        self.save_style = save_style
        self.png_path = png_path

        if self.store:
            self.clear_file()
        if self.h5_path is not None:
            self.h5_file = h5py.File(self.h5_path, mode="a")

    def get_patches(self, center_file=None, ret=True):
        downsample = self.slidescan.slide.level_downsamples[self.level]
        if center_file is None:
            self.__find_patches__(downsample)
            self.__save_patches__()
            print("Found {} valid patches in file {}".format(self.counter, self.slidescan.path))
        else:
            self.__save_patches_center__(center_file, downsample)
        if ret:
            patches = self.__load_patches__()
            return patches
        return self.counter

    def __save_patches_center__(self, center_file, downsample):
        patch = np.zeros(shape=(2,), dtype=np.int32)
        minxy = (self.tile_size * downsample) // 2  # minimum distance of a tile to upper and left boundary
        for x_value, y_value in zip(center_file["X"].values, center_file["Y"].values):
            self.valid_corner_x_list.append(x_value)
            self.valid_corner_y_list.append(y_value)
            self.counter += 1
            patch[0] = max(x_value - (self.tile_size * downsample) // 2, minxy)
            patch[1] = max(y_value - (self.tile_size * downsample) // 2, minxy)
            patch_img = np.array(
                self.slidescan.slide.read_region(patch, self.level, (self.tile_size, self.tile_size)).convert('RGB'))
            self.__save_png__(tile_x=x_value, tile_y=y_value, array=patch_img)
        self.__save_patches__()

    def __save_patches__(self):
        patches = pd.DataFrame({"X": self.valid_corner_x_list, "Y": self.valid_corner_y_list})
        patches["File"] = self.slidescan.path
        patches["tilesize"] = self.tile_size
        patches["level"] = self.level
        if self.store:
            patches.to_csv(self.save_path, mode="a", header=not os.path.exists(self.save_path))
        self.valid_corner_x_list = []
        self.valid_corner_y_list = []

    def __load_patches__(self):
        patches = pd.read_csv(self.save_path, index_col=0)
        return patches

    def __find_patches__(self, downsample):
        patch = np.zeros(shape=(2,), dtype=np.int32)
        for contour_idx, contour in enumerate(self.slidescan.foreground_contours):
            start_x, start_y, w, h = cv2.boundingRect(contour)
            for tile_x in np.arange(start_x, start_x + w, step=self.stride * downsample, dtype=np.int64):
                for tile_y in np.arange(start_y, start_y + h, step=self.stride * downsample, dtype=np.int64):
                    patch[0], patch[1] = tile_x, tile_y
                    if self.__is_valid_patch__(patch, contour, self.slidescan.hole_contours[contour_idx], tile_x,
                                               tile_y):
                        self.valid_corner_x_list.append(tile_x)
                        self.valid_corner_y_list.append(tile_y)
                        self.counter += 1
                    self.__save_patches__()

    def __is_valid_patch__(self, patch, contour, holes, tile_x, tile_y):
        if not self.use_black_padding:
            if not self.__is_in_image__(patch):
                return False
        if not self.__isInContour__(patch, contour):
            return False
        if len(holes) > 0:
            if self.__isInHoles__(patch, holes):
                return False
        patch_img = np.array(
            self.slidescan.slide.read_region(patch, self.level, (self.tile_size, self.tile_size)).convert('RGB'))
        if self.use_black_padding:
            if __isBlack__(patch_img, self.black_threshold):
                return False
        if __isWhitePatch__(patch_img, self.white_threshold):
            return False
        if self.save_style == "h5":
            if self.h5_path is not None:  # don't load patch twice
                self.__save_hf5__(tile_x=tile_x, tile_y=tile_y, array=patch_img)
        elif self.save_style == "png":
            self.__save_png__(tile_x=tile_x, tile_y=tile_y, array=patch_img)
        return True

    def __save_hf5__(self, tile_x, tile_y, array):
        grp_x = self.h5_file.require_group(str(tile_x))
        grp_y = grp_x.create_group(str(tile_y))  # combination of xy should be unique
        grp_y.create_dataset("default", shape=array.shape, dtype=np.uint8, data=array)

    def __save_png__(self, tile_x, tile_y, array):
        image = Image.fromarray(array)
        filename = "{}_{}.png".format(tile_x, tile_y)
        Path(self.png_path).mkdir(parents=True, exist_ok=True)
        image.save(os.path.join(self.png_path, filename), "png")

    def __is_in_image__(self, patch):
        image_dim_x, image_dim_y = self.slidescan.slide.level_dimensions[self.level]
        if (image_dim_x < patch[0] + self.tile_size) or (image_dim_y < patch[1] + self.tile_size):
            return False
        return True

    def __isInContour__(self, patch, contour):
        points = [(patch[0], patch[1]), (patch[0] + self.tile_size, patch[1]), (patch[0], patch[1] + self.tile_size),
                  (patch[0] + self.tile_size, patch[1] + self.tile_size)]
        return np.all(np.array(
            [cv2.pointPolygonTest(contour, tuple(np.array(point).astype(float)), False) for point in points]) >= 0)

    def __isInHoles__(self, patch, holes):
        points = [(patch[0], patch[1]), (patch[0] + self.tile_size, patch[1]), (patch[0], patch[1] + self.tile_size),
                  (patch[0] + self.tile_size, patch[1] + self.tile_size)]
        for hole in holes:
            if np.any(np.array(
                    [cv2.pointPolygonTest(hole, tuple(np.array(point).astype(float)), False) for point in points]) > 0):
                return True
        return False

    def visualize(self, path_to_save, loadlevel):
        patches = self.__load_patches__()
        img = np.array(self.slidescan.slide.read_region((0, 0), loadlevel,
                                                        self.slidescan.slide.level_dimensions[loadlevel]).convert(
            'RGB'))
        patches_len = patches.shape[0]
        load_downsample = self.slidescan.slide.level_downsamples[loadlevel]
        downsample_tiling = self.slidescan.slide.level_downsamples[self.level]
        for p_idx in range(patches_len):
            img[int(patches.iloc[p_idx]["Y"] / load_downsample):int(patches.iloc[p_idx]["Y"] / load_downsample) + int(
                224 * downsample_tiling / load_downsample),
            int(patches.iloc[p_idx]["X"] / load_downsample):int(patches.iloc[p_idx]["X"] / load_downsample) + int(
                224 * downsample_tiling / load_downsample), :] = 0
        img = Image.fromarray(img)
        img.save(path_to_save)

    def clear_file(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        if self.save_style == "h5":
            if os.path.exists(self.h5_path):
                os.remove(self.h5_path)
        else:
            try:
                shutil.rmtree(self.png_path)  # try to delete the old png directory
            except:
                pass

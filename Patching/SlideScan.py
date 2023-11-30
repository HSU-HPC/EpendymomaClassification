import os
from sys import platform

import cv2
import numpy as np
from PIL import Image
import os

if platform == "linux" or platform == "linux2" or platform == "darwin":
    import openslide
elif platform == "win32":
    print("Error")


def __scaleContour__(contours, scale):
    return [np.array(c * scale, dtype=np.int32) for c in contours]


class SlideScan:

    def __init__(self, path):
        self.path = path
        self.slide = openslide.open_slide(self.path)
        self.mask = None
        self.image = None
        self.hole_contours = None
        self.foreground_contours = None

        self.vendor = self.slide.properties[openslide.PROPERTY_NAME_VENDOR]

    def computeMask(self, level=-1, maxval=255, kernel_size=4, min_area_tissue=100, max_area_hole=16,
                    max_n_holes=8, save_path=None, ref_patch_size=512, viz_level=5, blocksize=11, c=2, erosionsize=15):
        """
        Computes the tissue mask
        :param viz_level: level for plotting
        :param ref_patch_size: size of the tiles
        :param max_n_holes: only this number of holes are considered per contour
        :param max_area_hole: maximum size of hole area
        :param min_area_tissue: minimal size of tissue area
        :param level: level to read
        :param k_size: aperture size for median blurring
        :param maxval: maximum value for thresholding
        :param kernel_size: size of (circular) kernel for morphological closing
        :param save_path: if not None, where to save the mask
        :param blocksize: block size for adaptive thresholding
        :param c: constant c for adaptive thresholding
        :param erosionsize: size of kernel for erosion
        :return: None
        """
        if self.image is None:
            self.read(level=level)
        self.__to_hsv__()
        # self.__median_blur__(k_size=k_size)
        # self.__thresholding__(maxval=maxval)
        # self.__morph_closing__(kernel_size=kernel_size)
        self.__adaptive_thresholding__(maxval=maxval, blocksize=blocksize, c=c)
        self.__morphological_opening_circular_kernel__(kernel_size)
        self.__erode__(erosionsize)
        self.__apply_area_threshold__(level=level, min_area_tissue=min_area_tissue, max_area_hole=max_area_hole,
                                      max_n_holes=max_n_holes, ref_patch_size=ref_patch_size)
        if save_path is not None:
            self.plot_segmentation(viz_level, save_path, line_thickness=250)


    def __erode__(self, erosion_size):
        kernel = np.ones((erosion_size, erosion_size), np.uint8)
        self.mask = cv2.erode(self.mask, kernel=kernel)


    def read(self, level):
        """
        Reads the slide scan to numpy array
        :param level: level number to read
        :return: None
        """
        self.image = np.array(self.slide.read_region((0, 0), level, self.slide.level_dimensions[level]))

    def __to_hsv__(self):
        """
        Converts self.image to HSV colorspace
        :return: None
        """
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)

    # def __median_blur__(self, k_size):
    #     """
    #     Applies median blurring to image
    #     :param k_size: int, aperture size
    #     :return: None
    #     """
    #     self.image = cv2.medianBlur(self.image[:, :, 1], k_size)
    #
    # def __thresholding__(self, maxval):
    #     """
    #     Applies Otsu's thresholding operation
    #     :param maxval: maximum value to use
    #     :return:
    #     """
    #     _, self.mask = cv2.threshold(self.image, 0, maxval, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    def __adaptive_thresholding__(self, maxval=255, blocksize=11, c=2):
        self.mask = cv2.adaptiveThreshold(self.image[:, :, 1], maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY,
                                          blocksize, c)

    def __morphological_opening_circular_kernel__(self, kernel_size=10):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.mask = cv2.bitwise_not(self.mask)

    def __morph_closing__(self, kernel_size):
        """
        Performs morphological closing of the mask
        :param kernel_size: size of the kernel
        :return: None
        """
        kernel = np.ones(kernel_size, dtype=np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

    def __apply_area_threshold__(self, level, min_area_tissue, max_area_hole, max_n_holes=8, ref_patch_size=512):
        min_area_tissue = min_area_tissue * (int(ref_patch_size / self.slide.level_downsamples[level]) ** 2)
        max_area_hole = max_area_hole * (int(ref_patch_size / self.slide.level_downsamples[level]) ** 2)

        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

        hierarchy_foreground = np.flatnonzero(hierarchy[:, 1] == -1)

        holes_all = []
        foreground_contours = []

        for contour_idx in hierarchy_foreground:
            contour = contours[contour_idx]
            contained_holes = np.flatnonzero(hierarchy[:, 1] == contour_idx)
            total_area = cv2.contourArea(contour)
            area_holes = np.sum([cv2.contourArea(contours[h_idx]) for h_idx in contained_holes])
            total_area_tissue = total_area - area_holes
            if total_area_tissue == 0:
                continue
            elif min_area_tissue < total_area_tissue:
                holes_all.append(contained_holes)
                foreground_contours.append(contour)

        hole_contours = []

        for hole_idxs in holes_all:
            unfiltered_holes = [contours[id] for id in hole_idxs]
            unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
            unfiltered_holes = unfiltered_holes[:max_n_holes]
            hole_contours.append([hole for hole in unfiltered_holes if cv2.contourArea(hole) > max_area_hole])

        self.hole_contours = [__scaleContour__(holes, self.slide.level_downsamples[level]) for holes in hole_contours]
        self.foreground_contours = __scaleContour__(foreground_contours, self.slide.level_downsamples[level])

    def plot_segmentation(self, level, save_path, line_thickness=250):

        img = np.array(self.slide.read_region((0, 0), level, self.slide.level_dimensions[level]).convert("RGB"))

        color_hole = (0, 0, 255)
        color_tissue = (0, 255, 0)
        scale = 1 / self.slide.level_downsamples[level]
        line_thickness = int(line_thickness * scale)
        cv2.drawContours(img, __scaleContour__(self.foreground_contours, scale), -1, color_tissue,
                         line_thickness,
                         lineType=cv2.LINE_8)
        for holes in self.hole_contours:
            cv2.drawContours(img, __scaleContour__(holes, scale), -1, color_hole, line_thickness, lineType=cv2.LINE_8)
        img = Image.fromarray(img)
        img.save(save_path)

    def __del__(self):
        self.slide.close()

from PIL import ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

rng = np.random.default_rng()


class RandomGaussianBlur:
    """
    Class to perform gaussian blurring with random sigma
    """

    def __init__(self, min_sigma, max_sigma):
        # define interval to choose from
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, x):
        # choose the sigma for this particular augmentation
        sigma = rng.uniform(low=self.min_sigma, high=self.max_sigma)
        # apply the gaussian filter
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class RandomDiscreteRotation:
    """
    Applies a rotation by an angle that is randomly chosen from a set of prescribed angles
    """

    def __init__(self, allowed_angles):
        # store the allowed angles
        self.angles = allowed_angles

    def __call__(self, x):
        # choose the angle
        angle = rng.choice(self.angles)
        # apply the rotation
        return TF.rotate(x, angle)


class TwoPathTransformer:
    """
    Creates two differently augmented versions of the same image
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline  # store the pipeline

    def __call__(self, x):
        x1 = self.pipeline(x)  # version 1
        x2 = self.pipeline(x)  # version 2
        return x1, x2


def get_twoway_transforms(blur: bool, verticalflip: bool, rotate: bool, mean: list, std: list, image_dim: int = 224):
    """
    Method to create a TwoPathTransformer with the correct augmentation pipeline
    :param blur: whether to use blurring augmentation
    :param verticalflip: whether to use vertical flips
    :param rotate: whether to rotate the image
    :param mean: means to use for normalization
    :param std: std to use for normalization
    :param image_dim: the size of the (quadratic) image
    :return: the corresponding TwoPathTransformer
    """

    augmentations = [
        transforms.RandomResizedCrop(image_dim, scale=(0.2, 1.)),  # resized crop, as in simsiam
        # here, I decided to use SimSiam's default values, not from Dehaene et al.
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),  # same as in simsiam
    ]
    if blur:  # we might not want any blurring for the cifar10 dataset
        augmentations.append(transforms.RandomApply([RandomGaussianBlur(0.1, 2.0)], p=0.5), )
    # in dehaene et al, they only used rotations and did not use any flipping
    # there are however images, where flpping can create augmentations that are impossible to achieve by
    # rotation alone. We therefore use it here, but want to be able to deactivate it for simsiam.
    if verticalflip:
        augmentations.append(transforms.RandomVerticalFlip())
    if rotate:
        augmentations.append(RandomDiscreteRotation([0.0, 90.0, 180.0, 270.0]))
    # now again as in simsiam
    augmentations.extend([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    augmentations = transforms.Compose(augmentations)
    return TwoPathTransformer(augmentations)


def get_regular_transforms(image_dim, mean, std):
    """Used only for KNN validation on the cifar10 dataset"""
    augmentations = [
        transforms.CenterCrop(image_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    augmentations = transforms.Compose(augmentations)
    return augmentations

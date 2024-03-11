import numpy as np
from PIL import Image


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    # Function from FDA Paper
    # taken from https://github.com/YanchaoYang/FDA/blob/master/utils/__init__.py
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def FDA_source_to_target_np(src_img, trg_img, L=0.1):
    # Function from FDA Paper
    # taken from https://github.com/YanchaoYang/FDA/blob/master/utils/__init__.py

    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img  # .cpu().numpy()
    trg_img_np = trg_img  # .cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(trg_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


class FDAMatcher:
    def __init__(self, x1: np.ndarray, beta):
        self.rng = np.random.default_rng()
        self.x1 = x1
        self.beta = beta

        # all ffts for analyses
        fft_x = np.fft.fft2(self.x1, axes=(1,2))
        name = "_".join([str(x) for x in fft_x.shape]) + ".npy"
        np.save(name, fft_x)

    def transform(self, x: Image):
        ref_idx = self.rng.integers(low=0, high=self.x1.shape[0])
        x = np.asarray(x, np.float32)
        x = x.transpose((2, 0, 1))
        x_ref = self.x1[ref_idx].transpose((2, 0, 1))
        np_new = FDA_source_to_target_np(x, x_ref, L=self.beta)
        # scale range
        np_new = (np_new - np.min(np_new)) * 255 / (np.max(np_new) - np.min(np_new))
        np_new = np_new.transpose((1, 2, 0))

        return Image.fromarray(np_new.astype(np.uint8))

    def __call__(self, x):
        return self.transform(x)

# img1 = Image.open("C:\\Users\\Yannis Schumann\\Downloads\\demo1.png").convert('RGB')
# img2 = Image.open("C:\\Users\\Yannis Schumann\\Downloads\\demo2.png").convert('RGB')
# im_src = img1.resize((1024, 512), Image.BICUBIC)
# im_trg = img2.resize((1024, 512), Image.BICUBIC)
##
#
# fda = FDAMatcher(np.array([np.asarray(im_trg), np.asarray(im_trg)], np.float32), 0.01)
# fda(im_src).show()

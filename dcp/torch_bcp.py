import torch
import torch.nn.functional as F
import math
import sys
import numpy as np
sys.path.append("/media/harry/ExtDrive/PycharmProjects/diffusion_opthalmology")

from dcp.guided_filter import GuidedFilter2d
from utils import clear_color, clear

from kornia.morphology import erosion, dilation
from torch import nn
from torchvision.transforms.functional import gaussian_blur

"""
All functions defined for extracting, and reconstruction using the dark channel prior (DCP) assumes batched 4D input.
Modifies https://github.com/He-Zhang/image_dehaze
"""


def BrightChannel(img, sz, verbose=False):
    bc, _ = torch.max(img, dim=1, keepdim=True)
    kernel = torch.ones(sz, sz, device=img.device)
    bright = dilation(bc, kernel)
    if verbose:
        return bright, bc
    else:
        return bright


def AtmLight_bright(img, reg_weight=1.5):
    """
    Estimate Atmospheric Light (Inhomogeneous) for low-light condition used in BCP
    (weight): weight of the convolutional kernel to be applied
    """
    b, _, h, w = img.shape
    img = img.mean(dim=1, keepdim=True)
    E = gaussian_blur(img, kernel_size=65, sigma=5.0)
    E[:, 0, ...] *= reg_weight
    Eprime = E / 2.
    return E, Eprime


def TransmissionEstimate_bright(Eprime, bright, omega=0.80):
    Eprime = Eprime[:, 0]
    transmission = 1. - omega*(1. - bright)/(1. - Eprime)
    return transmission


def TransmissionRefine(img, te, GF):
    """
    Coarse --> Fine transmission using guided filtering
    """
    gray_img = img.mean(dim=1, keepdim=True)
    t = GF(te, gray_img)
    return t


def Recover(img, t, Eprime, tx=0.1):
    """
    Finally, recover the dehazed result
    """
    t[t < tx] = tx
    Eprime = Eprime.repeat(1, 3, 1, 1)
    img_out = (img - Eprime)/t + Eprime
    return img_out


def BCP_recon(img, patch_size=15, radius=60, reg_weight=1.5, omega=0.80, eps=1e-4, device='cuda:0', verbose=False):
    GF = GuidedFilter2d(radius, eps).to(device)
    bright = BrightChannel(img, patch_size)
    E, Eprime = AtmLight_bright(img, reg_weight=reg_weight)
    te = TransmissionEstimate_bright(Eprime, bright, omega=omega)
    t = TransmissionRefine(img, te, GF)
    recon = Recover(img, t, Eprime, 0.1)
    if verbose:
        return recon, t, te, bright
    else:
        return recon


if __name__ == "__main__":
    from pathlib import Path
    from time import time
    import matplotlib.pyplot as plt

    load_root = Path(f"/media/harry/tomo/opthalmology/221115_bad")
    fname = "vk030546"

    device = 'cuda:0'
    verbose = False

    save_root = Path(f"./results/bcp/{fname}")
    save_root.mkdir(exist_ok=True, parents=True)

    # loading, preprocessing
    tmp = plt.imread(str(load_root / f"{fname}.jpg"))
    img = torch.tensor(tmp).type(torch.float32).to(device)
    img /= 255.0
    h, w, _ = img.shape
    img = img.permute(2, 0, 1).view(1, 3, h, w)

    # Define guided filtering class
    radius = 60
    eps = 1e-4
    GF = GuidedFilter2d(radius, eps).to(device)

    tic = time()
    recon = BCP_recon(img, device=device, verbose=verbose)
    toc = time() - tic
    print(f"Time took for dcp recon: {toc} sec.")

    recon = np.clip(clear_color(recon), 0.0, 1.0)
    plt.imsave(str(save_root / f"recon.png"), recon)
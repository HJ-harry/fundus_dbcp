import torch
import math
import sys
import numpy as np
sys.path.append("/media/harry/ExtDrive/PycharmProjects/diffusion_opthalmology")

from dcp.guided_filter import GuidedFilter2d
from utils import clear_color, clear

from kornia.morphology import erosion, dilation

"""
All functions defined for extracting, and reconstruction using the dark channel prior (DCP) assumes batched 4D input.
Modifies https://github.com/He-Zhang/image_dehaze
"""

def DarkChannel(img, sz, verbose=False):
    """
    Compute the dark channel of given img.
    """
    dc, _ = torch.min(img, dim=1, keepdim=True)
    kernel = torch.ones(sz, sz, device=img.device)
    dark = erosion(dc, kernel)
    if verbose:
        return dark, dc
    else:
        return dark


def BrightChannel(img, sz, verbose=False):
    bc, _ = torch.max(img, dim=1, keepdim=True)

    kernel = torch.ones(sz, sz, device=img.device)
    bright = erosion(bc, kernel)
    if verbose:
        return bright, dc
    else:
        return bright


def AtmLight(img, dark):
    """
    Estimate Atmospheric Light
    """
    b, _, h, w = img.shape
    img_sz = h * w

    # 1. Vectorize
    darkvec = torch.reshape(dark, (b, img_sz))
    img_vec = torch.reshape(img, (b, 3, img_sz)).permute(0, 2, 1)

    # 2. Pick the top 0.1% brightest pixels in the dark channel
    numpx = int(max(math.floor(img_sz / 1000), 1))
    indices = torch.argsort(darkvec)
    indices = indices[:, img_sz-numpx::]

    indices = indices.unsqueeze(2).repeat(b, 1, 3)
    A_all = torch.gather(img_vec, 1, indices)
    A = A_all.mean(dim=1)
    return A


def AtmLight_red_reg(img, dark, reg_weight=1.0):
    """
    Estimate Atmospheric Light, but regularize the red channel to have small value
    Explicitly down-weight the atmospheric light of the red channel.
    """
    b, _, h, w = img.shape
    img_sz = h * w

    # 1. Vectorize
    darkvec = torch.reshape(dark, (b, img_sz))
    img_vec = torch.reshape(img, (b, 3, img_sz)).permute(0, 2, 1)

    # 2. Pick the top 0.1% brightest pixels in the dark channel
    numpx = int(max(math.floor(img_sz / 1000), 1))
    indices = torch.argsort(darkvec)
    indices = indices[:, img_sz-numpx::]

    indices = indices.unsqueeze(2).repeat(b, 1, 3)
    A_all = torch.gather(img_vec, 1, indices)
    A = A_all.mean(dim=1)

    # 3. Red channel regularization
    A[:, 0] *= reg_weight
    return A


def TransmissionEstimate(img, A, sz, omega=0.95):
    scaled_im = img / A.view(1, 3, 1, 1).expand_as(img)
    transmission = 1 - omega*DarkChannel(scaled_im, sz)
    return transmission


def TransmissionRefine(img, te, GF):
    """
    Coarse --> Fine transmission using guided filtering
    """
    gray_img = img.mean(dim=1, keepdim=True)
    t = GF(te, gray_img)
    return t


def Recover(img, t, A, tx=0.1):
    """
    Finally, recover the dehazed result
    """
    t[t < tx] = tx
    A = A.view(1, 3, 1, 1).expand_as(img)
    img_out = (img - A)/t + A
    return img_out


def DCP_recon(img, patch_size=15, radius=60, omega=0.95, reg_weight=1.0,
              eps=1e-4, device='cuda:0', verbose=False):
    GF = GuidedFilter2d(radius, eps).to(device)
    dark = DarkChannel(img, patch_size)
    A = AtmLight(img, dark)
    te = TransmissionEstimate(img, A, patch_size, omega=omega)
    t = TransmissionRefine(img, te, GF)
    recon = Recover(img, t, A, 0.1)
    if verbose:
        return recon, t, te, dark, A
    else:
        return recon


if __name__ == "__main__":
    from pathlib import Path
    from time import time
    import matplotlib.pyplot as plt

    load_root = Path(f"/media/harry/tomo/opthalmology/221115_bad")
    fname = "vk030544"

    device = 'cuda:0'

    save_root = Path(f"./results/dcp/{fname}")
    save_root.mkdir(exist_ok=True, parents=True)

    # preprocessing
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
    dark, dc = DarkChannel(img, 15, verbose=True)
    A = AtmLight(img, dark)
    te = TransmissionEstimate(img, A, 15)
    t = TransmissionRefine(img, te, GF)
    recon = Recover(img, t, A, 0.1)
    toc = time() - tic
    print(f"Time took for dcp recon: {toc} sec.")

    # Crucial part. recon will often be in range [0.0, 4.0]
    # normalizing will make the image heavily dark. What we should do is CLIP.
    recon = np.clip(clear_color(recon), 0.0, 1.0)
    te = np.clip(clear(te), 0.0, 1.0)
    t = np.clip(clear(t), 0.0, 1.0)
    dark = np.clip(clear(dark), 0.0, 1.0)
    dc = np.clip(clear(dc), 0.0, 1.0)

    plt.imsave(str(save_root / f"recon.png"), recon)
    plt.imsave(str(save_root / f"transmission.png"), te, cmap='gray')
    plt.imsave(str(save_root / f"transmission_refine.png"), t, cmap='gray')
    plt.imsave(str(save_root / f"dark.png"), dark, cmap='gray')
    plt.imsave(str(save_root / f"dark_coarse.png"), dc, cmap='gray')
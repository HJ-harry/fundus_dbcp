import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm
from time import time

from dcp.torch_bcp import BCP_recon
from utils import clear, clear_color


verbose = True
if not verbose:
    save_root = Path(f"./results/bcp")
    save_root.mkdir(exist_ok=True, parents=True)

load_root = Path(f"/media/harry/tomo/opthalmology/221115_bad")
fname_list = sorted(list(load_root.glob("*.jpg")))

device = 'cuda:0'

for fname in tqdm(fname_list):
    filename = str(fname).split("/")[-1][:-4]
    if verbose:
        save_root = Path(f"./results/bcp_verbose/{filename}")
        save_root.mkdir(exist_ok=True, parents=True)

    # preprocessing
    img = torch.tensor(plt.imread(str(load_root / f"{filename}.jpg"))).type(torch.float32).to(device)
    img /= 255.0
    h, w, _ = img.shape
    img = img.permute(2, 0, 1).view(1, 3, h, w)

    recon_all = BCP_recon(img, device=device, verbose=verbose)
    if verbose:
        recon, te, t, bright = recon_all
        plt.imsave(str(save_root / f"transmission.png"), clear(te), cmap='gray')
        plt.imsave(str(save_root / f"transmission_refine.png"), clear(t), cmap='gray')
        plt.imsave(str(save_root / f"bright.png"), clear(bright), cmap='gray')
        plt.imsave(str(save_root / f"recon.png"), clear_color(recon))
    else:
        recon = recon_all
        plt.imsave(str(save_root / f"{filename}.png"), clear_color(recon))




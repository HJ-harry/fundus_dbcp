import matplotlib.pyplot as plt
import numpy as np
import torch
import shutup
shutup.please()

from pathlib import Path
from tqdm import tqdm
from time import time

from dcp.torch_dcp import DCP_recon
from utils import clear, clear_color


verbose = True

reg_weight = 1.0
omega = 0.95

folder_type = "cataract"
# folder_type = "additional"
# folder_type = "small_pupil"

load_root = Path(f"/media/harry/tomo/opthalmology/221115_bad")
# load_root = Path(f"/media/harry/tomo/opthalmology/230110/{folder_type}")
fname_list = sorted(list(load_root.glob("*.jpg")))

device = 'cuda:0'

if not verbose:
    save_root = Path(f"./results/dcp_verbose/omega{omega}")
    save_root.mkdir(exist_ok=True, parents=True)
for fname in tqdm(fname_list):
    filename = str(fname).split("/")[-1][:-4]
    if verbose:
        save_root = Path(f"./results/dcp_verbose/{filename}")
        save_root.mkdir(exist_ok=True, parents=True)

    # preprocessing
    img = torch.tensor(plt.imread(str(load_root / f"{filename}.jpg"))).type(torch.float32).to(device)
    img /= 255.0
    h, w, _ = img.shape
    img = img.permute(2, 0, 1).view(1, 3, h, w)

    recon_all = DCP_recon(img, reg_weight=reg_weight, device=device, verbose=verbose, omega=omega)
    if verbose:
        recon, te, t, dark, A = recon_all
        plt.imsave(str(save_root / f"transmission.png"), clear(te), cmap='gray')
        plt.imsave(str(save_root / f"transmission_refine.png"), clear(t), cmap='gray')
        plt.imsave(str(save_root / f"dark.png"), clear(dark), cmap='gray')
        plt.imsave(str(save_root / f"recon.png"), clear_color(recon))
    else:
        recon = recon_all
        plt.imsave(str(save_root / f"{filename}.png"), clear_color(recon))




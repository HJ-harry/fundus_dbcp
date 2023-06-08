import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

from dcp.torch_bcp import BCP_recon
from dcp.torch_dcp import DCP_recon
from utils import clear, clear_color

verbose = False
# dcp first usually works better
first = 'dcp'

# folder_type = "cataract"
# folder_type = "small_pupil"
folder_type = "additional"

dcp_omega = 0.95
bcp_omega_list = [0.8]

# red channel regularization weight
dcp_reg_weight = 1.0
bcp_reg_weight = 1.0

for bcp_omega in bcp_omega_list:
    if verbose:
        # save_root = Path(f"./results/221220/proposed/{folder_type}/bcp_omega{bcp_omega}")
        save_root = Path(f"./results/230110/proposed/{folder_type}/bcp_omega{bcp_omega}")
        # save_root = Path(f"./results/iterative/{first}_first_verbose/dcp_om{dcp_omega}_bcp_om{bcp_omega}")
    else:
        # save_root = Path(f"./results/221220/proposed/{folder_type}/bcp_omega{bcp_omega}")
        save_root = Path(f"./results/230110/proposed/{folder_type}/bcp_omega{bcp_omega}")
        # save_root = Path(f"./results/iterative/{first}_first_red_reg_dcp{dcp_reg_weight}_bcp{bcp_reg_weight}")

    save_root.mkdir(exist_ok=True, parents=True)

    # load_root = Path(f"/media/harry/tomo/opthalmology/221115_bad")
    # load_root = Path(f"/media/harry/tomo/opthalmology/221220/{folder_type}")
    load_root = Path(f"/media/harry/tomo/opthalmology/230110/{folder_type}")
    fname_list = sorted(list(load_root.glob("*.jpg")))

    device = 'cuda:0'

    for fname in tqdm(fname_list):
        filename = str(fname).split("/")[-1][:-4]

        # preprocessing
        img = torch.tensor(plt.imread(str(load_root / f"{filename}.jpg"))).type(torch.float32).to(device)
        img /= 255.0
        h, w, _ = img.shape
        img = img.permute(2, 0, 1).view(1, 3, h, w)

        # iterative recon
        if first == 'bcp':
            bcp_recon = BCP_recon(img, reg_weight=bcp_reg_weight, omega=bcp_omega, device=device, verbose=False)
            recon = DCP_recon(bcp_recon, reg_weight=dcp_reg_weight, omega=dcp_omega, device=device, verbose=False)

            if verbose:
                plt.imsave(str(save_root / f"{filename}_bcp.png"), clear_color(bcp_recon))
            plt.imsave(str(save_root / f"{filename}.png"), clear_color(recon))
        elif first == 'dcp':
            dcp_recon = DCP_recon(img, reg_weight=dcp_reg_weight, omega=dcp_omega, device=device, verbose=False)
            recon = BCP_recon(dcp_recon, reg_weight=bcp_reg_weight, omega=bcp_omega, device=device, verbose=False)

            if verbose:
                plt.imsave(str(save_root / f"{filename}_dcp.png"), clear_color(dcp_recon))
            plt.imsave(str(save_root / f"{filename}.png"), clear_color(recon))






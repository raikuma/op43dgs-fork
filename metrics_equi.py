#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = renders_dir / fname
        renders.append(render)
        image_names.append(fname)
    for fname in os.listdir(gt_dir):
        gt = gt_dir / fname
        gts.append(gt)
    # filter png
    gts = [g for g in gts if g.suffix == ".png"]
    gts = sorted(gts)[1::2] # Test views
    return renders, gts, image_names

def evaluate(model_paths, source_path):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test_equi"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            # gt_dir = method_dir/ "gt"
            gt_dir = Path(source_path)
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []
            alexs = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                render = tf.to_tensor(Image.open(renders[idx])).unsqueeze(0)[:, :3, :, :].cuda()
                gt = tf.to_tensor(Image.open(gts[idx])).unsqueeze(0)[:, :3, :, :].cuda()
                ssims.append(ssim(render, gt))
                psnrs.append(psnr(render, gt))
                lpipss.append(lpips(render, gt, net_type='vgg'))
                alexs.append(lpips(render, gt, net_type='alex'))

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("  Alex : {:>12.7f}".format(torch.tensor(alexs).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item(),
                                                    "Alex": torch.tensor(alexs).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                        "Alex": {name: lp for lp, name in zip(torch.tensor(alexs).tolist(), image_names)}})

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--source_path', '-s', required=True, type=str, default="")
    args = parser.parse_args()
    evaluate(args.model_paths, args.source_path)

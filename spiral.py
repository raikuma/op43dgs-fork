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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import numpy as np

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    #     rendering = render(view, gaussians, pipeline, background)["render"]
    #     gt = view.original_image[0:3, :, :]
    #     torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    #     torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    view = views[args.index]
    _T = np.array(view.T)
    _R = np.array(view.R)

    vid = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(view.image_width), int(view.image_height)))

    for i in range(args.frame):
        t = i / args.frame

        x = np.sin(t * 2 * np.pi) * args.t_scale
        y = np.sin(2*t * 2 * np.pi) * args.t_scale
        z = 0

        rx = -(y/args.t_scale) * np.pi * args.r_scale
        ry = (x/args.t_scale) * np.pi * args.r_scale
        rz = 0

        R = np.array([[np.cos(ry) * np.cos(rz), np.cos(ry) * np.sin(rz), -np.sin(ry)],
                        [np.sin(rx) * np.sin(ry) * np.cos(rz) - np.cos(rx) * np.sin(rz),
                        np.sin(rx) * np.sin(ry) * np.sin(rz) + np.cos(rx) * np.cos(rz),
                        np.sin(rx) * np.cos(ry)],
                        [np.cos(rx) * np.sin(ry) * np.cos(rz) + np.sin(rx) * np.sin(rz),
                        np.cos(rx) * np.sin(ry) * np.sin(rz) - np.sin(rx) * np.cos(rz),
                        np.cos(rx) * np.cos(ry)]])

        w2c = np.eye(4)
        w2c[:3, :3] = _R
        w2c[:3, 3] = _T
        c2w = np.linalg.inv(w2c)

        M = np.eye(4)
        M[:3, :3] = R

        c2w = M @ c2w

        w2c = np.linalg.inv(c2w)
        w2c[:3, 3] = w2c[:3, 3] + np.array([x, y, z])

        view.T = w2c[:3, 3]
        view.R = w2c[:3, :3]
        view.recalculate()

        rendering = render(view.cuda(), gaussians, pipeline, background)["render"]
        rendering = rendering.cpu().numpy().transpose(1, 2, 0)
        rendering = (rendering * 255).astype(np.uint8)
        vid.write(rendering)

    vid.release()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, fov_ratio : float, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, fov_ratio=fov_ratio)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #     if fov_ratio != 1:
        #         render_set(dataset.model_path, f"train_{fov_ratio}", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)
        #     else:
        #         render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
            if fov_ratio != 1:
                render_set(dataset.model_path, f"test_{fov_ratio}", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)
            else:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fov_ratio", default=1, type=float)
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--t_scale", default=0.1, type=float)
    parser.add_argument("--r_scale", default=0.0, type=float)
    parser.add_argument("--frame", default=90, type=int)
    parser.add_argument("--output", default="spiral.mp4", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.fov_ratio, args)
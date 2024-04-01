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

def reshape_and_expand(uncertainty):
    L = torch.zeros((uncertainty.shape[0], 3,3), dtype=torch.float, device="cuda")

    L[:, 0,0] = uncertainty[:, 0]
    L[:, 0,1] = uncertainty[:, 1]
    L[:, 0,2] = uncertainty[:, 2]
    L[:, 1,0] = L[:,0,1]
    L[:, 1,1] = uncertainty[:, 3]
    L[:, 1,2] = uncertainty[:, 4]
    L[:, 2,0]=L[:,0,2]
    L[:, 2,1]=L[:,1,2]
    L[:, 2,2] = uncertainty[:, 5]
    return L

def compute_normals(cov_matrices, scales):
    # 获取缩放向量中最小的缩放值对应的索引
    min_scale_index = torch.argmin(scales, dim=1)
    
    # 获取每个高斯分布的法线方向
    normals = []
    _, eigenvectors = torch.linalg.eig(cov_matrices)
    eigenvectors = torch.real(eigenvectors)
    # min_scale_axis = eigenvectors[:,:, min_scale_index].squeeze()
    
    # print(cov_matrices.shape)
    batch=100000
    for i in range(0,len(cov_matrices),batch):
        # 计算协方差矩阵的特征值和特征向量
        # eigenvalues = torch.linalg.eigvals(cov_matrices[i])
        # print(cov_matrices[i].shape)
        _, eigenvectors = torch.linalg.eig(cov_matrices[i:i+batch])
        eigenvectors = torch.real(eigenvectors)
        # print(eigenvectors)
        # 获取最小缩放值对应的协方差矩阵轴（特征向量）
        batch_indices = torch.arange(0, eigenvectors.size(0))
        min_scale_axis = eigenvectors[batch_indices, :,min_scale_index[i:i+batch]].squeeze()
        
        # 确保法线方向的z坐标为负数
        # if min_scale_axis[2] > 0:
        #     min_scale_axis *= -1
        
        # 将法线方向添加到法线列表中
        normals.append(min_scale_axis)
        print(eigenvectors.shape,min_scale_axis.shape)
    
    # 将法线列表转换为张量并返回
    return torch.cat(normals,dim=0)




def render_set(model_path, name, iteration, views, gaussians, pipeline, background,scale):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        override_color=None
        # override_color = compute_normals(reshape_and_expand(gaussians.get_covariance()), gaussians.get_scaling)
        
        # override_color = gaussians.get_xyz - view.camera_center
        
        # l = 0.001
        # ids1 = override_color[:,2]<l
        # ids2 = override_color[:,2]>l
        # ## gaussians._opacity[ids1] = -100000
        # override_color[ids1,2]=0
        # override_color[ids2,2] = 1/override_color[ids2,2] #/torch.max(override_color[:,2])
        # override_color[:,1]=override_color[:,2]
        # override_color[:,0]=override_color[:,2]

        rendering = render(view, gaussians, pipeline, background,override_color=override_color)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,scale):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians._scaling *= scale
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,scale)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,scale)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--scale", default=1, type=float)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.scale)
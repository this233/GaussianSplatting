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
import numpy as np
import open3d as o3d
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH,SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.feature_linear = nn.Linear(W, W)
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        
        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        
        self.scale_linear = nn.Linear(W//2, 3)
        self.rotate_linear = nn.Linear(W//2, 3)
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs  

# 创建 GaussianModel 模型，给点云中的每个点去创建一个3D gaussian，
# 每个3D gaussian都包含中心点的位置、颜色、不透明度、尺度、旋转参数。
# 给定一个新视角，可以根据这些参数把每个3D gaussian都投影到2D，
# 得到对应的2D gaussian。通过将这些2D gaussian按照深度的顺序进行混合，
# 就可以得到新视角2D图像上每个pixel的颜色。
class GaussianModel:

    def setup_functions(self):
        # 从尺度和旋转参数中去构建3Dgaussian的协方差矩阵
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            # print(scaling.shape,rotation.shape,L.shape,symm.shape)
            return symm
        
        self.scaling_activation = torch.exp# 将尺度限制为非负数
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid# 将不透明度限制在0-1的范围内
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,data_device="cuda"):
        self.data_device = data_device
        self.active_sh_degree = 0#使用的球谐阶数
        self.max_sh_degree = sh_degree  #最大球谐阶数
        self._xyz = torch.empty(0)# 中心点位置, 也即3Dgaussian的均值
        self._features_dc = torch.empty(0)# 第一个球谐系数, 球谐系数用来表示RGB颜色
        self._features_rest = torch.empty(0) # 其余球谐系数
        self._scaling = torch.empty(0)# 尺度
        self._rotation = torch.empty(0)# 旋转参数, 四元组
        self._opacity = torch.empty(0)# 不透明度
        self.max_radii2D = torch.empty(0)# 投影到2D时, 每个2D gaussian最大的半径
        self.xyz_gradient_accum = torch.empty(0)# 3Dgaussian的均值的累积梯度
        self.denom = torch.empty(0)
        self.optimizer = None# 上述各参数的优化器
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 重置不透明度
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def downsample(self):
        points_np = self._xyz.detach().cpu().numpy().astype(np.float64) # (P,3)
        colors_np = SH2RGB(self._features_dc).detach().cpu().numpy().reshape(-1, 3).astype(np.float64)  # 调整颜色维度以便与点云合并P,1,3 => P,3

        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors_np)

        # 进行体素网格过滤
        voxel_size = np.linalg.norm(np.max(points_np, axis=0) - np.min(points_np, axis=0)) / 25000
        down_pcd = pcd.voxel_down_sample(voxel_size)    

        # 将下采样后的点云转换回torch.Tensor
        downsampled_points = torch.tensor(np.asarray(down_pcd.points)).reshape(-1, 3).float().cuda()
        downsampled_colors = RGB2SH(torch.tensor(np.asarray(down_pcd.colors).flatten()).reshape(-1,1,3).float()).cuda()

        optimizable_tensors= self.replace_tensor_to_optimizer(downsampled_points, "xyz")
        self._xyz = optimizable_tensors["xyz"]
        
        optimizable_tensors = self.replace_tensor_to_optimizer(downsampled_colors, "f_dc")
        self._features_dc = optimizable_tensors["f_dc"]

        return 
        num_points =self._xyz.shape[0]
        retain_ratio = 0.9
        indices = torch.randperm(num_points)

        # 计算要保留的点数
        num_retain = int(num_points * retain_ratio)

        # 选择要保留的索引
        
        selected_indices = indices[:num_retain]
        
        selection_mask = torch.full((num_points,), True)

        # 将选中的索引位置设为True
        selection_mask[selected_indices] = False

        prune_filter = selection_mask
        # 使用 `prune_points` 方法进行点的修剪。
        self.prune_points(prune_filter)
        
    
    def reset_all(self):
        print("pre ",self._xyz.shape[0])
        self.downsample()
        print("after ",self._xyz.shape[0])    
        
        features_rest_new = torch.zeros((self.get_xyz.shape[0], 15, 3), device=self.data_device)
        optimizable_tensors = self.replace_tensor_to_optimizer(features_rest_new, "f_rest")
        self._features_rest = optimizable_tensors["f_rest"]
        
        opacity_new = inverse_sigmoid(0.1 * torch.ones((self.get_xyz.shape[0], 1), dtype=torch.float, device=self.data_device)) # (P, 1), 每个点的不透明度, 初始化为0.1
        optimizable_tensors = self.replace_tensor_to_optimizer(opacity_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        
        rotation_new = torch.zeros((self.get_xyz.shape[0], 4), device=self.data_device)# (P, 4), 每个点的旋转参数, 四元组
        rotation_new[:, 0] = 1
        optimizable_tensors = self.replace_tensor_to_optimizer(rotation_new, "rotation")
        self._rotation = optimizable_tensors["rotation"]
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(self.get_xyz.detach().cpu())).float().cuda()), 0.0000001)# (P,)
        scaling_new = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) # (P, 3), 每个点在X, Y, Z方向上的尺度
        optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
        self._scaling = optimizable_tensors["scaling"]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0],1), device=self.data_device)# (P,)

        self.denom = torch.zeros((self.get_xyz.shape[0],1), device=self.data_device)# (P,)
        
        # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
        # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
        # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.data_device)# (P,)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        print(pcd.points.shape,type(pcd.points))
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()# (P, 3)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())# (P, 3), 将RGB转换成球谐系数, C0项的系数
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()# (P, 3, 16), 每个颜色通道有16个球谐系数
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # distCUDA2 计算点云中的每个点到与其最近的K个点的平均距离的平方
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)# (P,)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) # (P, 3), 每个点在X, Y, Z方向上的尺度
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.data_device)# (P, 4), 每个点的旋转参数, 四元组
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.data_device)) # (P, 1), 每个点的不透明度, 初始化为0.1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True)) # (P, 3)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))# (P, 1, 3)
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))# (P, 15, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))# (P, 3)
        self._rotation = nn.Parameter(rots.requires_grad_(True)) # (P, 4)
        self._opacity = nn.Parameter(opacities.requires_grad_(True))# (P, 1)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.data_device)# (P,)
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense # 0.01
        # 存储每个3D gaussian的均值xyz的梯度, 用于判断是否对该3D gaussian进行克隆或者
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.data_device)# (P, 1)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.data_device)# (P, 1)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 创建optimizer 
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # 创建对xyz参数进行学习率调整的scheduler
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    # 对xyz的学习率进行调整
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.data_device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.data_device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.data_device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.data_device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.data_device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.data_device).requires_grad_(True))
        
        self.active_sh_degree = self.max_sh_degree

    # 重置不透明度
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 删除不符合要求的3D gaussian在self.optimizer中对应的参数(均值、球谐系数、不透明度、尺度、旋转参数)
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # 将挑选出来的3D gaussian的参数拼接到原有的参数之后
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.data_device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.data_device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.data_device)

    # 对于那些均值的梯度超过一定阈值且尺度大于一定阈值的3D gaussian进行分割操作
    # 这个函数的主要目的是在满足梯度条件的情况下对点进行稠密化和分割操作，以增加点的数量并改善点云的表示。
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):    
        # 获取初始点数 `n_init_points`。
        # with open('/data/liweihong/code/gaussian-splatting/gpu_memory_usage.log', 'a') as file:
        #     file.write(f"{grads.shape},{self.get_xyz.shape}\n")
        n_init_points = self.get_xyz.shape[0]
        
        # Extract points that satisfy the gradient condition
        # 创建布尔掩码 `selected_pts_mask`，将满足梯度条件且缩放值大于 `percent_dense * scene_extent` 的点选中。
        padded_grad = torch.zeros((n_init_points), device=self.data_device)  # 包括本次克隆过的点
        padded_grad[:grads.shape[0]] = grads.squeeze()  # 不包括本次克隆过的点
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 为满足条件的点创建分割坐标和属性，并将它们重复 `N` 次，形成 `N` 个分割点集。
        # 计算分割点的缩放值、旋转值和其他属性。
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=self.data_device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        
        # 在以原来3Dgaussian的均值xyz为中心, stds为形状, rots为方向的椭球内随机采样新的3Dgaussian
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)# (2 * P, 3)
        
        # 由于原来的3D gaussian的尺度过大, 现在将3D gaussian的尺度缩小为原来的1/1.6
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))# (2 * P, 3)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)# (2 * P, 4)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)# (2 * P, 1, 3)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)# (2 * P, 15, 3)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1) # (2 * P, 1)

        # 调用 `densification_postfix` 方法，对新的点和属性进行后处理。
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # 将原来的那些均值的梯度超过一定阈值且尺度大于一定阈值的3D gaussian进行删除 (因为已经将它们分割成了两个新的3D gaussian，原先的不再需要了)
        # 创建布尔掩码 `prune_filter`，其中包括选中的点和新生成的分割点。
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.data_device, dtype=bool)))
        # 使用 `prune_points` 方法进行点的修剪。
        self.prune_points(prune_filter)

    # 对于那些均值的梯度超过一定阈值且尺度小于一定阈值的3D gaussian进行克隆操作
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom # 3Dgaussian的均值的累积梯度
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)# 如果某些3Dgaussian的均值的梯度过大且尺度小于一定阈值，说明是欠重建，则对它们进行克隆
        self.densify_and_split(grads, max_grad, extent)# 如果某些3Dgaussian的均值的梯度过大且尺度超过一定阈值，说明是过重建，则对它们进行切分

        prune_mask = (self.get_opacity < min_opacity).squeeze() # 删除不透明度小于一定阈值的3Dgaussian
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size # 删除2D半径超过2D尺寸阈值的高斯
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent # 删除尺度超过一定阈值的高斯
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask) # 对不符合要求的高斯进行删除

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # print(viewspace_point_tensor.shape,update_filter.shape)
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
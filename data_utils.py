""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), 
Deep Global Registration (https://github.com/chrischoy/DeepGlobalRegistration),
SECOND (https://github.com/traveller59/second.pytorch), modified. """

import os
import glob
import numpy as np
import torch
import torch.utils.data
import six
import copy
import csv
import open3d as o3d

import utils


def load_3dmatch_batch_data(p0_fi, p1_fi, voxel_ratio):
    p0 = np.load(p0_fi)['pcd']
    p1 = np.load(p1_fi)['pcd']
    
    # voxelization
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(p0)
    p0_downsampled_pcd = pcd0.voxel_down_sample(voxel_size=voxel_ratio)   # open3d 0.8.0.0+
    p0_downsampled = np.array(p0_downsampled_pcd.points)
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(p1)
    p1_downsampled_pcd = pcd1.voxel_down_sample(voxel_size=voxel_ratio)   # open3d 0.8.0.0+
    p1_downsampled = np.array(p1_downsampled_pcd.points)
    
    return p0_downsampled, p1_downsampled
    

def find_voxel_overlaps(p0, p1, voxel):
    xmin, ymin, zmin = np.max(np.stack([np.min(p0, 0), np.min(p1, 0)]), 0)
    xmax, ymax, zmax = np.min(np.stack([np.max(p0, 0), np.max(p1, 0)]), 0)
    
    # truncate the point cloud
    eps = 1e-6
    p0_ = p0[np.all(p0>[xmin+eps,ymin+eps,zmin+eps], axis=1) & np.all(p0<[xmax-eps,ymax-eps,zmax-eps], axis=1)]
    p1_ = p1[np.all(p1>[xmin+eps,ymin+eps,zmin+eps], axis=1) & np.all(p1<[xmax-eps,ymax-eps,zmax-eps], axis=1)]
    
    # recalculate the constraints
    xmin, ymin, zmin = np.max(np.stack([np.min(p0, 0), np.min(p1, 0)]), 0)
    xmax, ymax, zmax = np.min(np.stack([np.max(p0, 0), np.max(p1, 0)]), 0)
    vx = (xmax - xmin) / voxel
    vy = (ymax - ymin) / voxel
    vz = (zmax - zmin) / voxel
    
    return p0_, p1_, xmin, ymin, zmin, xmax, ymax, zmax, vx, vy, vz


class ThreeDMatch_Testing(torch.utils.data.Dataset):
    def __init__(self, dataset_path, category, overlap_ratio, voxel_ratio, voxel, max_voxel_points, num_voxels, rigid_transform, vis, voxel_after_transf):
        self.dataset_path = dataset_path
        self.pairs = []
        with open(category, 'r') as fi:
            cinfo_fi = fi.read().split()   # category names
            for i in range(len(cinfo_fi)):
                cat_name = cinfo_fi[i]
                cinfo_name = cat_name + '*%.2f.txt' % overlap_ratio
                cinfo = glob.glob(os.path.join(self.dataset_path, cinfo_name))
                for fi_name in cinfo:
                    with open(fi_name) as fi:
                        fi_list = [x.strip().split() for x in fi.readlines()]
                    for fi in fi_list:
                        self.pairs.append([fi[0], fi[1]])
                        
        self.voxel_ratio = voxel_ratio
        self.voxel = int(voxel)
        self.max_voxel_points = max_voxel_points
        self.num_voxels = num_voxels
        self.perturbation = load_pose(rigid_transform, len(self.pairs))
        self.vis = vis
        self.voxel_after_transf = voxel_after_transf
        
    def __len__(self):
        return len(self.pairs)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6], twist-params
        g = utils.exp(x).to(p0) # [1, 4, 4]
        p1 = utils.transform(g, p0)
        igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt
    
    def __getitem__(self, index):
        p0_pre, p1_pre = load_3dmatch_batch_data(os.path.join(self.dataset_path, self.pairs[index][0]), os.path.join(self.dataset_path, self.pairs[index][1]), self.voxel_ratio)
        
        if self.voxel_after_transf:
            x = torch.from_numpy(self.perturbation[index][np.newaxis,...])
            p1_pre, igt = self.do_transform(torch.from_numpy(p1_pre).double(), x)
        
            p0_pre_mean = np.mean(p0_pre,0)
            p1_pre_mean = np.mean(p1_pre.numpy(),0)
            p0_pre_ = p0_pre - p0_pre_mean
            p1_pre_ = p1_pre.numpy() - p1_pre_mean
            
            # voxelization
            p0, p1, xmin, ymin, zmin, xmax, ymax, zmax, vx, vy, vz = find_voxel_overlaps(p0_pre_, p1_pre_, self.voxel)   # constraints of P1 ^ P2, where contains roughly overlapped area
            
            p0 = p0 + p0_pre_mean
            p1 = p1 + p1_pre_mean
            xmin0 = xmin + p0_pre_mean[0]
            ymin0 = ymin + p0_pre_mean[1]
            zmin0 = zmin + p0_pre_mean[2]
            xmax0 = xmax + p0_pre_mean[0]
            ymax0 = ymax + p0_pre_mean[1]
            zmax0 = zmax + p0_pre_mean[2]

            xmin1 = xmin + p1_pre_mean[0]
            ymin1 = ymin + p1_pre_mean[1]
            zmin1 = zmin + p1_pre_mean[2]
            xmax1 = xmax + p1_pre_mean[0]
            ymax1 = ymax + p1_pre_mean[1]
            zmax1 = zmax + p1_pre_mean[2]
            
            voxels_p0, coords_p0, num_points_per_voxel_p0 = points_to_voxel_second(p0, (xmin0, ymin0, zmin0, xmax0, ymax0, zmax0), 
                            (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
            voxels_p1, coords_p1, num_points_per_voxel_p1 = points_to_voxel_second(p1, (xmin1, ymin1, zmin1, xmax1, ymax1, zmax1), 
                            (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
        else:
            # voxelization
            p0, p1, xmin, ymin, zmin, xmax, ymax, zmax, vx, vy, vz = find_voxel_overlaps(p0_pre, p1_pre, self.voxel)   # constraints of P1 ^ P2, where contains roughly overlapped area
            voxels_p0, coords_p0, num_points_per_voxel_p0 = points_to_voxel_second(p0, (xmin, ymin, zmin, xmax, ymax, zmax), 
                            (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
            voxels_p1, coords_p1, num_points_per_voxel_p1 = points_to_voxel_second(p1, (xmin, ymin, zmin, xmax, ymax, zmax), 
                            (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
        
        coords_p0_idx = coords_p0[:,1]*(int(self.voxel**2)) + coords_p0[:,0]*(int(self.voxel)) + coords_p0[:,2]
        coords_p1_idx = coords_p1[:,1]*(int(self.voxel**2)) + coords_p1[:,0]*(int(self.voxel)) + coords_p1[:,2]
        
        if self.voxel_after_transf:
            # calculate for the voxel medium
            xm_x0 = np.linspace(xmin0+vx/2, xmax0-vx/2, int(self.voxel))
            xm_y0 = np.linspace(ymin0+vy/2, ymax0-vy/2, int(self.voxel))
            xm_z0 = np.linspace(zmin0+vz/2, zmax0-vz/2, int(self.voxel))
            mesh3d0 = np.vstack(np.meshgrid(xm_x0,xm_y0,xm_z0)).reshape(3,-1).T
            xm_x1 = np.linspace(xmin1+vx/2, xmax1-vx/2, int(self.voxel))
            xm_y1 = np.linspace(ymin1+vy/2, ymax1-vy/2, int(self.voxel))
            xm_z1 = np.linspace(zmin1+vz/2, zmax1-vz/2, int(self.voxel))
            mesh3d1 = np.vstack(np.meshgrid(xm_x1,xm_y1,xm_z1)).reshape(3,-1).T
            
            voxel_coords_p0 = mesh3d0[coords_p0_idx]
            voxel_coords_p1 = mesh3d1[coords_p1_idx]
        else:
            # calculate for the voxel medium
            xm_x = np.linspace(xmin+vx/2, xmax-vx/2, int(self.voxel))
            xm_y = np.linspace(ymin+vy/2, ymax-vy/2, int(self.voxel))
            xm_z = np.linspace(zmin+vz/2, zmax-vz/2, int(self.voxel))
            mesh3d = np.vstack(np.meshgrid(xm_x,xm_y,xm_z)).reshape(3,-1).T
            voxel_coords_p0 = mesh3d[coords_p0_idx]
            voxel_coords_p1 = mesh3d[coords_p1_idx]
            
        # find voxels where number of points >= 80% of the maximum number of points
        idx_conditioned_p0 = coords_p0_idx[np.where(num_points_per_voxel_p0>=0.1*self.max_voxel_points)]
        idx_conditioned_p1 = coords_p1_idx[np.where(num_points_per_voxel_p1>=0.1*self.max_voxel_points)]
        idx_conditioned, _, _ = np.intersect1d(idx_conditioned_p0, idx_conditioned_p1, assume_unique=True, return_indices=True)
        _, _, idx_p0 = np.intersect1d(idx_conditioned, coords_p0_idx, assume_unique=True, return_indices=True)
        _, _, idx_p1 = np.intersect1d(idx_conditioned, coords_p1_idx, assume_unique=True, return_indices=True)
        voxel_coords_p0 = voxel_coords_p0[idx_p0]
        voxel_coords_p1 = voxel_coords_p1[idx_p1]
        voxels_p0 = voxels_p0[idx_p0]
        voxels_p1 = voxels_p1[idx_p1]
        
        if not self.voxel_after_transf:
            x = torch.from_numpy(self.perturbation[index][np.newaxis,...])
            voxels_p1_, igt = self.do_transform(torch.from_numpy(voxels_p1.reshape(-1,3)), x)
            voxels_p1 = voxels_p1_.reshape(voxels_p1.shape)
            voxel_coords_p1, _ = self.do_transform(torch.from_numpy(voxel_coords_p1).double(), x)
            p1, _ = self.do_transform(torch.from_numpy(p1), x)
        
        if self.vis:
            return voxels_p0, voxel_coords_p0, voxels_p1, voxel_coords_p1, igt, p0, p1
        else:    
            return voxels_p0, voxel_coords_p0, voxels_p1, voxel_coords_p1, igt


class ToyExampleData(torch.utils.data.Dataset):
    def __init__(self, p0, p1, voxel_ratio, voxel, max_voxel_points, num_voxels, rigid_transform, vis):
        self.voxel_ratio = voxel_ratio
        self.voxel = int(voxel)
        self.max_voxel_points = max_voxel_points
        self.num_voxels = num_voxels
        self.perturbation = rigid_transform
        self.p0 = p0
        self.p1 = p1
        self.vis = vis

    def __len__(self):
        return len(self.p0)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6], twist-params
        g = utils.exp(x).to(p0) # [1, 4, 4]
        p1 = utils.transform(g, p0)
        igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt

    def __getitem__(self, index):
        p0_pre = self.p0[index]
        p1_pre = self.p1[index]
        
        # voxelization
        p0, p1, xmin, ymin, zmin, xmax, ymax, zmax, vx, vy, vz = find_voxel_overlaps(p0_pre, p1_pre, self.voxel)   # constraints of P1 ^ P2, where contains roughly overlapped area
        voxels_p0, coords_p0, num_points_per_voxel_p0 = points_to_voxel_second(p0, (xmin, ymin, zmin, xmax, ymax, zmax), 
                        (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
        voxels_p1, coords_p1, num_points_per_voxel_p1 = points_to_voxel_second(p1, (xmin, ymin, zmin, xmax, ymax, zmax), 
                        (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
        
        coords_p0_idx = coords_p0[:,1]*(int(self.voxel**2)) + coords_p0[:,0]*(int(self.voxel)) + coords_p0[:,2]
        coords_p1_idx = coords_p1[:,1]*(int(self.voxel**2)) + coords_p1[:,0]*(int(self.voxel)) + coords_p1[:,2]
        
        # calculate for the voxel medium
        xm_x = np.linspace(xmin+vx/2, xmax-vx/2, int(self.voxel))
        xm_y = np.linspace(ymin+vy/2, ymax-vy/2, int(self.voxel))
        xm_z = np.linspace(zmin+vz/2, zmax-vz/2, int(self.voxel))
        mesh3d = np.vstack(np.meshgrid(xm_x,xm_y,xm_z)).reshape(3,-1).T
        voxel_coords_p0 = mesh3d[coords_p0_idx]
        voxel_coords_p1 = mesh3d[coords_p1_idx]
        
        # find voxels where number of points >= 80% of the maximum number of points
        idx_conditioned_p0 = coords_p0_idx[np.where(num_points_per_voxel_p0>=0.1*self.max_voxel_points)]
        idx_conditioned_p1 = coords_p1_idx[np.where(num_points_per_voxel_p1>=0.1*self.max_voxel_points)]
        idx_conditioned, _, _ = np.intersect1d(idx_conditioned_p0, idx_conditioned_p1, assume_unique=True, return_indices=True)
        _, _, idx_p0 = np.intersect1d(idx_conditioned, coords_p0_idx, assume_unique=True, return_indices=True)
        _, _, idx_p1 = np.intersect1d(idx_conditioned, coords_p1_idx, assume_unique=True, return_indices=True)
        voxel_coords_p0 = voxel_coords_p0[idx_p0]
        voxel_coords_p1 = voxel_coords_p1[idx_p1]
        voxels_p0 = voxels_p0[idx_p0]
        voxels_p1 = voxels_p1[idx_p1]
        
        x = torch.from_numpy(self.perturbation[index][np.newaxis,...])
        voxels_p1_, igt = self.do_transform(torch.from_numpy(voxels_p1.reshape(-1,3)), x)
        voxels_p1 = voxels_p1_.reshape(voxels_p1.shape)
        voxel_coords_p1, _ = self.do_transform(torch.from_numpy(voxel_coords_p1).double(), x)
        p1, _ = self.do_transform(torch.from_numpy(p1), x)
        
        if self.vis:
            return voxels_p0, voxel_coords_p0, voxels_p1, voxel_coords_p1, igt, p0, p1
        else:
            return voxels_p0, voxel_coords_p0, voxels_p1, voxel_coords_p1, igt
    
    
class RandomTransformSE3:
    """ randomly generate rigid transformations """

    def __init__(self, mag=1, mag_randomly=True):
        self.mag = mag
        self.randomly = mag_randomly
        self.gt = None
        self.igt = None

    def generate_transform(self):
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        return x

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6], twist params
        g = utils.exp(x).to(p0)   # [1, 4, 4]
        gt = utils.exp(-x).to(p0)  # [1, 4, 4]
        p1 = utils.transform(g, p0)
        self.gt = gt   # p1 --> p0
        self.igt = g   # p0 --> p1
        
        return p1, g.squeeze(0)  # 返回变换后的点云和变换矩阵

    def transform(self, tensor):
        x = self.generate_transform()
        p1, igt = self.apply_transform(tensor, x)
        return p1

    def __call__(self, tensor):
        return self.transform(tensor)


def add_noise(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += torch.clamp(sigma * torch.randn(N, C), -1 * clip, clip)

    return pointcloud


class PointRegistration(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, sigma=0.00, clip=0.00):
        self.dataset = dataset
        self.transf = rigid_transform
        self.sigma = sigma
        self.clip = clip

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pm, _ = self.dataset[index]   # one point cloud
        p_ = add_noise(pm, sigma=self.sigma, clip=self.clip)
        p1 = self.transf(p_)
        igt = self.transf.igt.squeeze(0)
        p0 = pm
        
        # p0: template, p1: source, igt:transform matrix from p0 to p1
        return p0, p1, igt
        

class PointRegistration_fixed_perturbation(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, sigma=0.00, clip=0.00):
        torch.manual_seed(713)
        self.dataset = dataset
        self.transf_ = load_pose(rigid_transform, len(self.dataset))
        list_order = torch.randperm(len(self.dataset))
        self.transf = self.transf_[list_order]
        self.sigma = sigma
        self.clip = clip

    def __len__(self):
        return len(self.dataset)

    def transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6], twist-vector (rotation and translation)
        g = utils.exp(x).to(p0)   # [1, 4, 4]
        p1 = utils.transform(g, p0)
        igt = g.squeeze(0)
        
        return p1, igt
    
    def __getitem__(self, index):
        pm, _ = self.dataset[index]   # one point cloud
        p_ = add_noise(pm, sigma=self.sigma, clip=self.clip)
        p0 = pm
        x = torch.from_numpy(self.transf[index][np.newaxis, ...]).to(p0)
        p1, igt = self.transform(p_, x)
        
        # p0: template, p1: source, igt:transform matrix from p0 to p1
        return p0, p1, igt
        
        
# adapted from SECOND: https://github.com/nutonomy/second.pytorch/blob/master/second/core/point_cloud/point_cloud_ops.py
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coords_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    ndim = 3
    grid_size = (coords_range[3:] - coords_range[:3]) / voxel_size
    grid_size = np.around(grid_size, 0, grid_size).astype(np.int32)

    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coords_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            # print(voxel_num)
            if voxel_num > max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1

    return voxel_num


# adapted from SECOND: https://github.com/nutonomy/second.pytorch/blob/master/second/core/point_cloud/point_cloud_ops.py
def points_to_voxel_second(points,
                     coords_range,
                     voxel_size,
                     max_points=100,
                     reverse_index=False,
                     max_voxels=20000):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.
    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coords_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.
    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coords_range, np.ndarray):
        coords_range = np.array(coords_range, dtype=points.dtype)
    voxelmap_shape = (coords_range[3:] - coords_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.around(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.ones(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype) * np.mean(points, 0)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    voxel_num = _points_to_voxel_kernel(
        points, voxel_size, coords_range, num_points_per_voxel,
        coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]

    return voxels, coors, num_points_per_voxel


def load_pose(trans_pth, num_pose):
    with open(trans_pth, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        poses = []
        for row in csvreader:
            row = [float(i) for i in row]
            poses.append(row)
        init_gt = np.array(poses)[:num_pose]
    print('init_trans shape is {}'.format(init_gt.shape))
    
    return init_gt


def find_classes(root):
    """ find ${root}/${class}/* """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """
    root = os.path.expanduser(root)
    samples = []

    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue

        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue

        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
                item = (path, target_idx)
                samples.append(item)
                
    return samples


class Globset(torch.utils.data.Dataset):
    """ glob ${rootdir}/${classes}/${pattern}
    """
    def __init__(self, rootdir, pattern, fileloader, transform=None, classinfo=None):
        super().__init__()

        if isinstance(pattern, six.string_types):
            pattern = [pattern]

        if classinfo is not None:
            classes, class_to_idx = classinfo
        else:
            classes, class_to_idx = find_classes(rootdir)

        samples = glob_dataset(rootdir, class_to_idx, pattern)
        if not samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))

        self.rootdir = rootdir
        self.pattern = pattern
        self.fileloader = fileloader
        self.transform = transform

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

    def __repr__(self):
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.rootdir)
        fmt_str += '    File Patterns: {}\n'.format(self.pattern)
        fmt_str += '    File Loader: {}\n'.format(self.fileloader)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.fileloader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def num_classes(self):
        return len(self.classes)

    def class_name(self, cidx):
        return self.classes[cidx]

    def indices_in_class(self, cidx):
        targets = np.array(list(map(lambda s: s[1], self.samples)))
        return np.where(targets == cidx).tolist()

    def select_classes(self, cidxs):
        indices = []
        for i in cidxs:
            idxs = self.indices_in_class(i)
            indices.extend(idxs)
        return indices

    def split(self, rate):
        """ dateset -> dataset1, dataset2. s.t.
            len(dataset1) = rate * len(dataset),
            len(dataset2) = (1-rate) * len(dataset)
        """
        orig_size = len(self)
        select = np.zeros(orig_size, dtype=int)
        csize = np.zeros(len(self.classes), dtype=int)
        dsize = np.zeros(len(self.classes), dtype=int)

        for i in range(orig_size):
            _, target = self.samples[i]
            csize[target] += 1
        dsize = (csize * rate).astype(int)
        for i in range(orig_size):
            _, target = self.samples[i]
            if dsize[target] > 0:
                select[i] = 1
                dsize[target] -= 1

        dataset1 = copy.deepcopy(self)
        dataset2 = copy.deepcopy(self)

        samples1 = list(map(lambda i: dataset1.samples[i], np.where(select == 1)[0]))
        samples2 = list(map(lambda i: dataset2.samples[i], np.where(select == 0)[0]))

        dataset1.samples = samples1
        dataset2.samples = samples2
        return dataset1, dataset2


class Mesh:
    def __init__(self):
        self._vertices = [] # array-like (N, D)
        self._faces = [] # array-like (M, K)
        self._edges = [] # array-like (L, 2)

    def clone(self):
        other = copy.deepcopy(self)
        return other

    def clear(self):
        for key in self.__dict__:
            self.__dict__[key] = []

    def add_attr(self, name):
        self.__dict__[name] = []

    @property
    def vertex_array(self):
        return np.array(self._vertices)

    @property
    def vertex_list(self):
        return list(map(tuple, self._vertices))

    @staticmethod
    def faces2polygons(faces, vertices):
        p = list(map(lambda face: \
                        list(map(lambda vidx: vertices[vidx], face)), faces))
        return p

    @property
    def polygon_list(self):
        p = Mesh.faces2polygons(self._faces, self._vertices)
        return p

    def on_unit_sphere(self, zero_mean=False):
        # radius == 1
        v = self.vertex_array # (N, D)
        if zero_mean:
            a = np.mean(v[:, 0:3], axis=0, keepdims=True) # (1, 3)
            v[:, 0:3] = v[:, 0:3] - a
        n = np.linalg.norm(v[:, 0:3], axis=1) # (N,)
        m = np.max(n) # scalar
        v[:, 0:3] = v[:, 0:3] / m
        self._vertices = v
        return self

    def on_unit_cube(self, zero_mean=False):
        # volume == 1
        v = self.vertex_array # (N, D)
        if zero_mean:
            a = np.mean(v[:, 0:3], axis=0, keepdims=True) # (1, 3)
            v[:, 0:3] = v[:, 0:3] - a
        m = np.max(np.abs(v)) # scalar
        v[:, 0:3] = v[:, 0:3] / (m * 2)
        self._vertices = v
        return self

    def rot_x(self):
        # camera local (up: +Y, front: -Z) -> model local (up: +Z, front: +Y).
        v = self.vertex_array
        t = np.copy(v[:, 1])
        v[:, 1] = -np.copy(v[:, 2])
        v[:, 2] = t
        self._vertices = list(map(tuple, v))
        return self

    def rot_zc(self):
        # R = [0, -1;
        #      1,  0]
        v = self.vertex_array
        x = np.copy(v[:, 0])
        y = np.copy(v[:, 1])
        v[:, 0] = -y
        v[:, 1] = x
        self._vertices = list(map(tuple, v))
        return self

def offread(filepath, points_only=True):
    """ read Geomview OFF file. """
    with open(filepath, 'r') as fin:
        mesh, fixme = _load_off(fin, points_only)
    if fixme:
        _fix_modelnet_broken_off(filepath)
    return mesh

def _load_off(fin, points_only):
    """ read Geomview OFF file. """
    mesh = Mesh()

    fixme = False
    sig = fin.readline().strip()
    if sig == 'OFF':
        line = fin.readline().strip()
        num_verts, num_faces, num_edges = tuple([int(s) for s in line.split(' ')])
    elif sig[0:3] == 'OFF': # ...broken data in ModelNet (missing '\n')...
        line = sig[3:]
        num_verts, num_faces, num_edges = tuple([int(s) for s in line.split(' ')])
        fixme = True
    else:
        raise RuntimeError('unknown format')

    for v in range(num_verts):
        vp = tuple(float(s) for s in fin.readline().strip().split(' '))
        mesh._vertices.append(vp)

    if points_only:
        return mesh, fixme

    for f in range(num_faces):
        fc = tuple([int(s) for s in fin.readline().strip().split(' ')][1:])
        mesh._faces.append(fc)

    return mesh, fixme


def _fix_modelnet_broken_off(filepath):
    oldfile = '{}.orig'.format(filepath)
    os.rename(filepath, oldfile)
    with open(oldfile, 'r') as fin:
        with open(filepath, 'w') as fout:
            sig = fin.readline().strip()
            line = sig[3:]
            print('OFF', file=fout)
            print(line, file=fout)
            for line in fin:
                print(line.strip(), file=fout)


def objread(filepath, points_only=True):
    """Loads a Wavefront OBJ file. """
    _vertices = []
    _normals = []
    _texcoords = []
    _faces = []
    _mtl_name = None

    material = None
    for line in open(filepath, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            v = tuple(map(float, values[1:4]))
            _vertices.append(v)
        elif values[0] == 'vn':
            v = tuple(map(float, values[1:4]))
            _normals.append(v)
        elif values[0] == 'vt':
            _texcoords.append(tuple(map(float, values[1:3])))
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'mtllib':
            _mtl_name = values[1]
        elif values[0] == 'f':
            face_ = []
            texcoords_ = []
            norms_ = []
            for v in values[1:]:
                w = v.split('/')
                face_.append(int(w[0]) - 1)
                if len(w) >= 2 and len(w[1]) > 0:
                    texcoords_.append(int(w[1]) - 1)
                else:
                    texcoords_.append(-1)
                if len(w) >= 3 and len(w[2]) > 0:
                    norms_.append(int(w[2]) - 1)
                else:
                    norms_.append(-1)
            #_faces.append((face_, norms_, texcoords_, material))
            _faces.append(face_)

    mesh = Mesh()
    mesh._vertices = _vertices
    if points_only:
        return mesh

    mesh._faces = _faces

    return mesh


class Mesh2Points:
    def __init__(self):
        pass

    def __call__(self, mesh):
        mesh = mesh.clone()
        v = mesh.vertex_array
        return torch.from_numpy(v).type(dtype=torch.float)


class OnUnitCube:
    def __init__(self):
        pass

    def method1(self, tensor):
        m = tensor.mean(dim=0, keepdim=True) # [N, D] -> [1, D]
        v = tensor - m
        s = torch.max(v.abs())
        v = v / s * 0.5
        return v

    def method2(self, tensor):
        c = torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0] # [N, D] -> [D]
        s = torch.max(c) # -> scalar
        v = tensor / s
        return v - v.mean(dim=0, keepdim=True)

    def __call__(self, tensor):
        #return self.method1(tensor)
        return self.method2(tensor)


class ModelNet(Globset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """
    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        loader = offread
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class ShapeNet2(Globset):
    """ [ShapeNet](https://www.shapenet.org/) v2 """
    def __init__(self, dataset_path, transform=None, classinfo=None):
        loader = objread
        pattern = '*/models/model_normalized.obj'
        super().__init__(dataset_path, pattern, loader, transform, classinfo)
        

class Resampler:
    """ [N, D] -> [M, D] """
    def __init__(self, num):
        self.num = num

    def __call__(self, tensor):
        num_points, dim_p = tensor.size()
        out = torch.zeros(self.num, dim_p).to(tensor)

        selected = 0
        while selected < self.num:
            remainder = self.num - selected
            idx = torch.randperm(num_points)
            sel = min(remainder, num_points)
            val = tensor[idx[:sel]]
            out[selected:(selected + sel)] = val
            selected += sel
        return out


class C3VDDataset(torch.utils.data.Dataset):
    """
    C3VD医学点云数据集，支持体素化和智能采样
    基于addc3vd.mdc文档的完整实现
    """
    
    def __init__(self, source_root, target_root=None, pairing_strategy='one_to_one',
                 voxel_config=None, sampling_config=None, transform=None, 
                 train=True, vis=False):
        """
        初始化C3VD数据集
        
        Args:
            source_root: 源点云根目录路径
            target_root: 目标点云根目录路径（可选，默认与source_root相同）
            pairing_strategy: 配对策略 ('one_to_one', 'scene_reference', 'source_to_source', 'target_to_target', 'all')
            voxel_config: 体素化配置参数
            sampling_config: 智能采样配置参数
            transform: Ground Truth变换（训练时使用）
            train: 是否为训练模式
            vis: 是否返回可视化数据
        """
        super().__init__()
        self.source_root = source_root
        self.target_root = target_root or source_root
        self.pairing_strategy = pairing_strategy
        self.transform = transform
        self.train = train
        self.vis = vis
        
        # 体素化配置（C3VD推荐配置）
        self.voxel_config = voxel_config or {
            'voxel_size': 0.05,
            'voxel_grid_size': 32,
            'max_voxel_points': 100,
            'max_voxels': 20000,
            'min_voxel_points_ratio': 0.1
        }
        
        # 智能采样配置
        self.sampling_config = sampling_config or {
            'target_points': 1024,
            'intersection_priority': True,
            'min_intersection_ratio': 0.3,
            'max_intersection_ratio': 0.7
        }
        
        # 解剖区域映射
        self.anatomy_classes = {
            'cecum': 1,
            'desc': 2, 
            'sigmoid': 3,
            'trans': 4
        }
        
        # 加载数据对列表
        self.pairs = self._load_pairs()
        print(f"✅ C3VD数据集加载完成: {len(self.pairs)}个点云对")
    
    def _load_pairs(self):
        """根据配对策略加载点云对列表"""
        pairs = []
        
        # 扫描源目录获取所有场景
        source_scenes = self._scan_scenes(self.source_root)
        target_scenes = self._scan_scenes(self.target_root)
        
        if self.pairing_strategy == 'one_to_one':
            pairs = self._create_one_to_one_pairs(source_scenes, target_scenes)
        elif self.pairing_strategy == 'scene_reference':
            pairs = self._create_scene_reference_pairs(source_scenes, target_scenes)
        elif self.pairing_strategy == 'source_to_source':
            pairs = self._create_source_to_source_pairs(source_scenes)
        elif self.pairing_strategy == 'target_to_target':
            pairs = self._create_target_to_target_pairs(target_scenes)
        elif self.pairing_strategy == 'all':
            pairs.extend(self._create_one_to_one_pairs(source_scenes, target_scenes))
            pairs.extend(self._create_source_to_source_pairs(source_scenes))
            pairs.extend(self._create_target_to_target_pairs(target_scenes))
        else:
            raise ValueError(f"不支持的配对策略: {self.pairing_strategy}")
        
        return pairs
    
    def _scan_scenes(self, root_path):
        """扫描目录获取所有场景和对应的PLY文件"""
        scenes = {}
        
        if not os.path.exists(root_path):
            print(f"警告: 路径不存在 {root_path}")
            return scenes
            
        for scene_dir in os.listdir(root_path):
            scene_path = os.path.join(root_path, scene_dir)
            if not os.path.isdir(scene_path):
                continue
                
            # 解析场景名称 (anatomy_trial_sequence)
            parts = scene_dir.split('_')
            if len(parts) >= 3:
                anatomy = parts[0]
                trial = parts[1] 
                sequence = parts[2]
                
                # 查找PLY文件
                ply_files = []
                for file_name in os.listdir(scene_path):
                    if file_name.endswith('.ply'):
                        ply_files.append(os.path.join(scene_path, file_name))
                
                if ply_files:
                    scenes[scene_dir] = {
                        'anatomy': anatomy,
                        'trial': trial,
                        'sequence': sequence,
                        'files': sorted(ply_files),
                        'class_id': self.anatomy_classes.get(anatomy, 0)
                    }
        
        return scenes
    
    def _create_one_to_one_pairs(self, source_scenes, target_scenes):
        """创建一对一配对"""
        pairs = []
        
        for scene_name in source_scenes:
            if scene_name in target_scenes:
                source_files = source_scenes[scene_name]['files']
                target_files = target_scenes[scene_name]['files']
                
                # 配对同一场景的文件
                min_len = min(len(source_files), len(target_files))
                for i in range(min_len):
                    pairs.append({
                        'source': source_files[i],
                        'target': target_files[i],
                        'scene': scene_name,
                        'anatomy': source_scenes[scene_name]['anatomy'],
                        'class_id': source_scenes[scene_name]['class_id']
                    })
        
        return pairs
    
    def _create_scene_reference_pairs(self, source_scenes, target_scenes):
        """创建场景参考配对（每个场景使用一个共享目标）"""
        pairs = []
        
        for scene_name in source_scenes:
            if scene_name in target_scenes:
                source_files = source_scenes[scene_name]['files']
                target_files = target_scenes[scene_name]['files']
                
                if target_files:
                    # 使用第一个目标文件作为参考
                    reference_target = target_files[0]
                    
                    for source_file in source_files:
                        pairs.append({
                            'source': source_file,
                            'target': reference_target,
                            'scene': scene_name,
                            'anatomy': source_scenes[scene_name]['anatomy'],
                            'class_id': source_scenes[scene_name]['class_id']
                        })
        
        return pairs
    
    def _create_source_to_source_pairs(self, source_scenes):
        """创建源到源配对（数据增强）"""
        pairs = []
        
        for scene_name, scene_info in source_scenes.items():
            files = scene_info['files']
            
            # 创建同一场景内的文件配对
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    pairs.append({
                        'source': files[i],
                        'target': files[j],
                        'scene': scene_name,
                        'anatomy': scene_info['anatomy'],
                        'class_id': scene_info['class_id']
                    })
        
        return pairs
    
    def _create_target_to_target_pairs(self, target_scenes):
        """创建目标到目标配对（数据增强）"""
        return self._create_source_to_source_pairs(target_scenes)
    
    def __len__(self):
        return len(self.pairs)
    
    def _load_ply_file(self, file_path):
        """加载PLY文件并返回点云数据"""
        try:
            # 使用Open3D加载PLY文件
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                print(f"警告: PLY文件为空 {file_path}")
                return np.array([]).reshape(0, 3)
            
            return points.astype(np.float32)
            
        except Exception as e:
            print(f"错误: 无法加载PLY文件 {file_path}: {e}")
            return np.array([]).reshape(0, 3)
    
    def _apply_ground_truth_transform(self, target_points):
        """应用Ground Truth变换"""
        if self.transform is None:
            return target_points, np.eye(4)
        
        # 转换为torch tensor
        target_tensor = torch.from_numpy(target_points).float()
        
        # 应用变换
        if hasattr(self.transform, 'apply_transform'):
            # 使用RandomTransformSE3的apply_transform方法
            x = self.transform.generate_transform()
            transformed_tensor, igt = self.transform.apply_transform(target_tensor, x)
            return transformed_tensor.numpy(), igt.numpy()
        else:
            # 直接调用transform
            transformed_tensor = self.transform(target_tensor)
            return transformed_tensor.numpy(), np.eye(4)
    
    def _voxelize_point_clouds(self, source_points, target_points):
        """
        对点云对进行体素化处理
        实现addc3vd.mdc中描述的体素化流程
        """
        try:
            # 1. 寻找重叠区域并计算体素网格参数
            source_overlap, target_overlap, xmin, ymin, zmin, xmax, ymax, zmax, vx, vy, vz = \
                find_voxel_overlaps(source_points, target_points, self.voxel_config['voxel_grid_size'])
            
            if len(source_overlap) == 0 or len(target_overlap) == 0:
                print("警告: 点云无重叠区域，回退到原始点云")
                return source_points, target_points, np.array([])
            
            # 2. 体素化转换
            coords_range = (xmin, ymin, zmin, xmax, ymax, zmax)
            voxel_size = (vx, vy, vz)
            
            # 对源点云体素化
            source_voxels, source_coords, source_num_points = points_to_voxel_second(
                source_overlap, coords_range, voxel_size,
                max_points=self.voxel_config['max_voxel_points'],
                max_voxels=self.voxel_config['max_voxels']
            )
            
            # 对目标点云体素化
            target_voxels, target_coords, target_num_points = points_to_voxel_second(
                target_overlap, coords_range, voxel_size,
                max_points=self.voxel_config['max_voxel_points'],
                max_voxels=self.voxel_config['max_voxels']
            )
            
            # 3. 体素筛选和交集计算
            intersection_indices = self._compute_voxel_intersection(
                source_coords, target_coords, source_num_points, target_num_points
            )
            
            # 4. 智能采样策略
            final_source, final_target = self._smart_sampling_strategy(
                source_voxels, target_voxels, source_coords, target_coords,
                intersection_indices, source_num_points, target_num_points
            )
            
            return final_source, final_target, intersection_indices
            
        except Exception as e:
            print(f"体素化处理失败: {e}")
            print("回退到随机采样策略")
            return self._fallback_random_sampling(source_points, target_points)
    
    def _compute_voxel_intersection(self, source_coords, target_coords, 
                                   source_num_points, target_num_points):
        """计算体素交集"""
        # 基于点密度筛选有效体素
        min_points_threshold = int(self.voxel_config['min_voxel_points_ratio'] * 
                                 self.voxel_config['max_voxel_points'])
        
        # 筛选有效体素
        valid_source_mask = source_num_points >= min_points_threshold
        valid_target_mask = target_num_points >= min_points_threshold
        
        # 计算体素索引
        grid_size = self.voxel_config['voxel_grid_size']
        source_indices = (source_coords[:, 1] * (grid_size**2) + 
                         source_coords[:, 0] * grid_size + 
                         source_coords[:, 2])
        target_indices = (target_coords[:, 1] * (grid_size**2) + 
                         target_coords[:, 0] * grid_size + 
                         target_coords[:, 2])
        
        # 获取有效体素的索引
        valid_source_indices = source_indices[valid_source_mask]
        valid_target_indices = target_indices[valid_target_mask]
        
        # 计算交集
        intersection_indices, source_intersect_idx, target_intersect_idx = np.intersect1d(
            valid_source_indices, valid_target_indices, 
            assume_unique=True, return_indices=True
        )
        
        return intersection_indices
    
    def _smart_sampling_strategy(self, source_voxels, target_voxels, 
                               source_coords, target_coords, intersection_indices,
                               source_num_points, target_num_points):
        """
        智能采样策略：优先保留交集体素，其余随机采样
        实现addc3vd.mdc中描述的智能采样算法
        """
        target_points = self.sampling_config['target_points']
        
        # 1. 提取交集体素的点
        intersection_source_points = []
        intersection_target_points = []
        
        if len(intersection_indices) > 0:
            # 找到交集体素在原始体素数组中的位置
            grid_size = self.voxel_config['voxel_grid_size']
            source_indices = (source_coords[:, 1] * (grid_size**2) + 
                             source_coords[:, 0] * grid_size + 
                             source_coords[:, 2])
            target_indices = (target_coords[:, 1] * (grid_size**2) + 
                             target_coords[:, 0] * grid_size + 
                             target_coords[:, 2])
            
            for intersection_idx in intersection_indices:
                # 找到对应的体素
                source_voxel_mask = source_indices == intersection_idx
                target_voxel_mask = target_indices == intersection_idx
                
                if np.any(source_voxel_mask):
                    source_voxel_idx = np.where(source_voxel_mask)[0][0]
                    voxel_points = source_voxels[source_voxel_idx]
                    num_points = source_num_points[source_voxel_idx]
                    intersection_source_points.append(voxel_points[:num_points])
                
                if np.any(target_voxel_mask):
                    target_voxel_idx = np.where(target_voxel_mask)[0][0]
                    voxel_points = target_voxels[target_voxel_idx]
                    num_points = target_num_points[target_voxel_idx]
                    intersection_target_points.append(voxel_points[:num_points])
        
        # 合并交集点
        if intersection_source_points:
            intersection_source = np.concatenate(intersection_source_points, axis=0)
        else:
            intersection_source = np.array([]).reshape(0, 3)
            
        if intersection_target_points:
            intersection_target = np.concatenate(intersection_target_points, axis=0)
        else:
            intersection_target = np.array([]).reshape(0, 3)
        
        # 2. 应用智能采样策略
        final_source = self._apply_sampling_to_pointcloud(
            source_voxels, source_num_points, intersection_source, 
            intersection_indices, source_coords, target_points
        )
        
        final_target = self._apply_sampling_to_pointcloud(
            target_voxels, target_num_points, intersection_target,
            intersection_indices, target_coords, target_points
        )
        
        return final_source, final_target
    
    def _apply_sampling_to_pointcloud(self, voxels, num_points_per_voxel, 
                                    intersection_points, intersection_indices,
                                    coords, target_points):
        """对单个点云应用采样策略"""
        num_intersection = len(intersection_points)
        
        if num_intersection >= target_points:
            # 情况1：交集点已经足够，直接从交集中采样
            indices = np.random.choice(num_intersection, target_points, replace=False)
            return intersection_points[indices]
        else:
            # 情况2：交集点不够，保留所有交集点 + 随机采样其他点
            remaining_points = target_points - num_intersection
            
            # 获取非交集体素的点
            non_intersection_points = self._get_non_intersection_points(
                voxels, num_points_per_voxel, intersection_indices, coords
            )
            
            if len(non_intersection_points) >= remaining_points:
                # 从非交集点中随机采样
                indices = np.random.choice(len(non_intersection_points), 
                                         remaining_points, replace=False)
                sampled_non_intersection = non_intersection_points[indices]
                return np.concatenate([intersection_points, sampled_non_intersection], axis=0)
            else:
                # 非交集点也不够，使用重复采样
                all_points = np.concatenate([intersection_points, non_intersection_points], axis=0)
                shortage = target_points - len(all_points)
                
                if shortage > 0:
                    # 重复采样补足
                    indices = np.random.choice(len(all_points), shortage, replace=True)
                    repeated_points = all_points[indices]
                    return np.concatenate([all_points, repeated_points], axis=0)
                else:
                    return all_points
    
    def _get_non_intersection_points(self, voxels, num_points_per_voxel, 
                                   intersection_indices, coords):
        """获取非交集体素中的所有点"""
        grid_size = self.voxel_config['voxel_grid_size']
        voxel_indices = (coords[:, 1] * (grid_size**2) + 
                        coords[:, 0] * grid_size + 
                        coords[:, 2])
        
        non_intersection_points = []
        
        for i, voxel_idx in enumerate(voxel_indices):
            if voxel_idx not in intersection_indices:
                num_points = num_points_per_voxel[i]
                if num_points > 0:
                    non_intersection_points.append(voxels[i][:num_points])
        
        if non_intersection_points:
            return np.concatenate(non_intersection_points, axis=0)
        else:
            return np.array([]).reshape(0, 3)
    
    def _fallback_random_sampling(self, source_points, target_points):
        """回退到随机采样策略"""
        target_size = self.sampling_config['target_points']
        
        # 对源点云采样
        if len(source_points) > target_size:
            indices = np.random.choice(len(source_points), target_size, replace=False)
            source_sampled = source_points[indices]
        else:
            source_sampled = source_points
            
        # 对目标点云采样
        if len(target_points) > target_size:
            indices = np.random.choice(len(target_points), target_size, replace=False)
            target_sampled = target_points[indices]
        else:
            target_sampled = target_points
            
        return source_sampled, target_sampled, np.array([])
    
    def __getitem__(self, index):
        """获取数据项"""
        pair_info = self.pairs[index]
        
        # 1. 加载原始点云
        source_points = self._load_ply_file(pair_info['source'])
        target_points = self._load_ply_file(pair_info['target'])
        
        if len(source_points) == 0 or len(target_points) == 0:
            # 返回空数据
            empty_points = np.zeros((self.sampling_config['target_points'], 3), dtype=np.float32)
            return {
                'source': torch.from_numpy(empty_points),
                'target': torch.from_numpy(empty_points),
                'igt': torch.eye(4),
                'scene': pair_info['scene'],
                'anatomy': pair_info['anatomy'],
                'class_id': pair_info['class_id'],
                'intersection_ratio': 0.0
            }
        
        # 2. 应用Ground Truth变换（训练时）
        igt = np.eye(4)
        if self.train and self.transform is not None:
            target_points, igt = self._apply_ground_truth_transform(target_points)
        
        # 3. 体素化处理和智能采样
        final_source, final_target, intersection_indices = self._voxelize_point_clouds(
            source_points, target_points
        )
        
        # 4. 计算交集比例
        total_voxels = max(len(final_source) // self.sampling_config['target_points'], 1)
        intersection_ratio = len(intersection_indices) / total_voxels if total_voxels > 0 else 0.0
        
        # 5. 转换为torch tensor
        result = {
            'source': torch.from_numpy(final_source).float(),
            'target': torch.from_numpy(final_target).float(),
            'igt': torch.from_numpy(igt).float(),
            'scene': pair_info['scene'],
            'anatomy': pair_info['anatomy'],
            'class_id': pair_info['class_id'],
            'intersection_ratio': intersection_ratio
        }
        
        # 6. 可视化数据（可选）
        if self.vis:
            result.update({
                'source_original': torch.from_numpy(source_points).float(),
                'target_original': torch.from_numpy(target_points).float()
            })
        
        return result
    
    def get_sample_info(self, index):
        """获取样本信息（用于分析）"""
        if index >= len(self.pairs):
            raise IndexError(f"索引 {index} 超出范围 (数据集大小: {len(self.pairs)})")
        
        pair_info = self.pairs[index]
        return {
            'sequence': pair_info['scene'],
            'frame1': os.path.basename(pair_info['source']),
            'frame2': os.path.basename(pair_info['target']),
            'anatomy': pair_info['anatomy'],
            'class_id': pair_info['class_id']
        }
    
    def analyze_processing_pipeline(self, index):
        """分析处理流程中点云数量的变化"""
        if index >= len(self.pairs):
            raise IndexError(f"索引 {index} 超出范围 (数据集大小: {len(self.pairs)})")
        
        pair_info = self.pairs[index]
        
        # 1. 加载原始点云
        source_points = self._load_ply_file(pair_info['source'])
        target_points = self._load_ply_file(pair_info['target'])
        
        original_points1 = len(source_points)
        original_points2 = len(target_points)
        
        if original_points1 == 0 or original_points2 == 0:
            return {
                'original_points1': original_points1,
                'original_points2': original_points2,
                'voxelized_points1': 0,
                'voxelized_points2': 0,
                'sampled_points1': 0,
                'sampled_points2': 0,
                'final_points1': 0,
                'final_points2': 0
            }
        
        # 2. 应用Ground Truth变换（如果需要）
        igt = np.eye(4)
        if self.train and self.transform is not None:
            target_points, igt = self._apply_ground_truth_transform(target_points)
        
        # 3. 体素化处理
        try:
            # 使用Open3D进行体素化
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_points)
            source_voxelized = source_pcd.voxel_down_sample(voxel_size=self.voxel_config['voxel_size'])
            source_voxelized_points = np.asarray(source_voxelized.points)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_points)
            target_voxelized = target_pcd.voxel_down_sample(voxel_size=self.voxel_config['voxel_size'])
            target_voxelized_points = np.asarray(target_voxelized.points)
            
            voxelized_points1 = len(source_voxelized_points)
            voxelized_points2 = len(target_voxelized_points)
            
            # 4. 采样处理
            target_size = self.sampling_config['target_points']
            
            # 对源点云采样
            if voxelized_points1 > target_size:
                indices = np.random.choice(voxelized_points1, target_size, replace=False)
                sampled_source = source_voxelized_points[indices]
            else:
                sampled_source = source_voxelized_points
                
            # 对目标点云采样
            if voxelized_points2 > target_size:
                indices = np.random.choice(voxelized_points2, target_size, replace=False)
                sampled_target = target_voxelized_points[indices]
            else:
                sampled_target = target_voxelized_points
            
            sampled_points1 = len(sampled_source)
            sampled_points2 = len(sampled_target)
            
            # 5. 最终输出（确保达到目标点数）
            if sampled_points1 < target_size:
                # 重复采样补足
                shortage = target_size - sampled_points1
                indices = np.random.choice(sampled_points1, shortage, replace=True)
                repeated_points = sampled_source[indices]
                final_source = np.concatenate([sampled_source, repeated_points], axis=0)
            else:
                final_source = sampled_source
                
            if sampled_points2 < target_size:
                # 重复采样补足
                shortage = target_size - sampled_points2
                indices = np.random.choice(sampled_points2, shortage, replace=True)
                repeated_points = sampled_target[indices]
                final_target = np.concatenate([sampled_target, repeated_points], axis=0)
            else:
                final_target = sampled_target
            
            final_points1 = len(final_source)
            final_points2 = len(final_target)
            
        except Exception as e:
            print(f"处理过程中出错: {e}")
            # 返回基本统计信息
            voxelized_points1 = original_points1
            voxelized_points2 = original_points2
            sampled_points1 = min(original_points1, self.sampling_config['target_points'])
            sampled_points2 = min(original_points2, self.sampling_config['target_points'])
            final_points1 = self.sampling_config['target_points']
            final_points2 = self.sampling_config['target_points']
        
        return {
            'original_points1': original_points1,
            'original_points2': original_points2,
            'voxelized_points1': voxelized_points1,
            'voxelized_points2': voxelized_points2,
            'sampled_points1': sampled_points1,
            'sampled_points2': sampled_points2,
            'final_points1': final_points1,
            'final_points2': final_points2
        }


def create_c3vd_dataset(source_root, target_root=None, pairing_strategy='one_to_one',
                       mag=0.8, train=True, vis=False, **kwargs):
    """
    创建C3VD数据集的便捷函数
    
    Args:
        source_root: 源点云根目录
        target_root: 目标点云根目录
        pairing_strategy: 配对策略
        mag: Ground Truth变换幅度
        train: 是否为训练模式
        vis: 是否返回可视化数据
        **kwargs: 其他配置参数
    
    Returns:
        C3VDDataset实例
    """
    # C3VD推荐的体素化配置
    voxel_config = kwargs.get('voxel_config', {
        'voxel_size': 0.05,
        'voxel_grid_size': 32,
        'max_voxel_points': 100,
        'max_voxels': 20000,
        'min_voxel_points_ratio': 0.1
    })
    
    # 智能采样配置
    sampling_config = kwargs.get('sampling_config', {
        'target_points': 1024,
        'intersection_priority': True,
        'min_intersection_ratio': 0.3,
        'max_intersection_ratio': 0.7
    })
    
    # Ground Truth变换（训练时）
    transform = None
    if train:
        transform = RandomTransformSE3(mag=mag, mag_randomly=True)
    
    return C3VDDataset(
        source_root=source_root,
        target_root=target_root,
        pairing_strategy=pairing_strategy,
        voxel_config=voxel_config,
        sampling_config=sampling_config,
        transform=transform,
        train=train,
        vis=vis
    )
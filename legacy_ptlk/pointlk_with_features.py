""" PointLK ver. 2018.07.06 - 支持可替换特征提取器版本
    using approximated Jacobian by backward-difference.
"""

import numpy
import torch
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from . import se3, so3, invmat
from feature_extractors import create_feature_extractor, FeatureExtractorFactory


class PointLKWithFeatures(torch.nn.Module):
    """支持可替换特征提取器的PointLK类
    
    保持与原版PointLK的完全兼容性，同时支持多种特征提取器
    """
    
    def __init__(self, feature_extractor_name='pointnet', feature_config=None, 
                 delta=1.0e-2, learn_delta=False):
        """
        Args:
            feature_extractor_name: 特征提取器名称 ('pointnet', 'attention', 'cformer', 'fast_attention', 'mamba3d')
            feature_config: 特征提取器配置字典
            delta: 数值微分步长
            learn_delta: 是否学习delta参数
        """
        super().__init__()
        
        # 创建特征提取器
        self.feature_extractor_name = feature_extractor_name
        self.ptnet = create_feature_extractor(feature_extractor_name, feature_config)
        
        # PointLK核心组件
        self.inverse = invmat.InvMatrix.apply
        self.exp = se3.Exp # [B, 6] -> [B, 4, 4]
        self.transform = se3.transform # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

        w1 = delta
        w2 = delta
        w3 = delta
        v1 = delta
        v2 = delta
        v3 = delta
        twist = torch.Tensor([w1, w2, w3, v1, v2, v3])
        self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=learn_delta)

        # 结果存储
        self.last_err = None
        self.g_series = None # for debug purpose
        self.prev_r = None
        self.g = None # estimation result
        self.itr = 0

    @staticmethod
    def rsq(r):
        """残差平方和"""
        z = torch.zeros_like(r)
        return torch.nn.functional.mse_loss(r, z, size_average=False)

    @staticmethod
    def comp(g, igt):
        """ |g*igt - I| (should be 0) """
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        return torch.nn.functional.mse_loss(A, I, size_average=True) * 16

    @staticmethod
    def do_forward(net, p0, p1, maxiter=10, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True):
        """静态前向传播方法"""
        a0 = torch.eye(4).view(1, 4, 4).expand(p0.size(0), 4, 4).to(p0) # [B, 4, 4]
        a1 = torch.eye(4).view(1, 4, 4).expand(p1.size(0), 4, 4).to(p1) # [B, 4, 4]
        
        if p0_zero_mean:
            p0_m = p0.mean(dim=1) # [B, N, 3] -> [B, 3]
            a0[:, 0:3, 3] = p0_m
            q0 = p0 - p0_m.unsqueeze(1)
        else:
            q0 = p0

        if p1_zero_mean:
            p1_m = p1.mean(dim=1) # [B, N, 3] -> [B, 3]
            a1[:, 0:3, 3] = -p1_m
            q1 = p1 - p1_m.unsqueeze(1)
        else:
            q1 = p1

        r = net(q0, q1, maxiter=maxiter, xtol=xtol)

        if p0_zero_mean or p1_zero_mean:
            est_g = net.g
            if p0_zero_mean:
                est_g = a0.to(est_g).bmm(est_g)
            if p1_zero_mean:
                est_g = est_g.bmm(a1.to(est_g))
            net.g = est_g

            est_gs = net.g_series # [M, B, 4, 4]
            if p0_zero_mean:
                est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
            if p1_zero_mean:
                est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
            net.g_series = est_gs

        return r

    def forward(self, p0, p1, maxiter=10, xtol=1.0e-7):
        """前向传播"""
        g0 = torch.eye(4).to(p0).view(1, 4, 4).expand(p0.size(0), 4, 4).contiguous()
        r, g, itr = self.iclk(g0, p0, p1, maxiter, xtol)

        self.g = g
        self.itr = itr
        return r, g  # 返回r和g两个值

    def update(self, g, dx):
        """更新变换矩阵"""
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = self.exp(dx)
        return dg.matmul(g)

    def approx_Jic(self, p0, f0, dt):
        """近似雅可比矩阵计算"""
        # p0: [B, N, 3], Variable
        # f0: [B, K], corresponding feature vector
        # dt: [B, 6], Variable
        # Jk = (ptnet(p(-delta[k], p0)) - f0) / delta[k]

        batch_size = p0.size(0)
        num_points = p0.size(1)

        # 计算变换
        transf = torch.zeros(batch_size, 6, 4, 4).to(p0)
        for b in range(p0.size(0)):
            d = torch.diag(dt[b, :]) # [6, 6]
            D = self.exp(-d) # [6, 4, 4]
            transf[b, :, :, :] = D[:, :, :]
        transf = transf.unsqueeze(2).contiguous()  #   [B, 6, 1, 4, 4]
        p = self.transform(transf, p0.unsqueeze(1)) # x [B, 1, N, 3] -> [B, 6, N, 3]

        f0 = f0.unsqueeze(-1) # [B, K, 1]
        f = self.ptnet(p.view(-1, num_points, 3)).view(batch_size, 6, -1).transpose(1, 2) # [B, K, 6]

        df = f0 - f # [B, K, 6]
        J = df / dt.unsqueeze(1)

        return J

    def iclk(self, g0, p0, p1, maxiter, xtol):
        """Iterative Closest Point - 类似Lucas-Kanade算法"""
        training = self.ptnet.training
        batch_size = p0.size(0)

        g = g0
        self.g_series = torch.zeros(maxiter+1, *g0.size(), dtype=g0.dtype)
        self.g_series[0] = g0.clone()

        if training:
            # 首先更新BatchNorm模块
            f0 = self.ptnet(p0)
            f1 = self.ptnet(p1)
        self.ptnet.eval() # 然后固定它们

        # 用当前模块重新计算
        f0 = self.ptnet(p0) # [B, N, 3] -> [B, K]

        # 用有限差分近似雅可比矩阵
        dt = self.dt.to(p0).expand(batch_size, 6)
        J = self.approx_Jic(p0, f0, dt)

        self.last_err = None
        itr = -1
        
        # 计算pinv(J)来解J*x = -r
        try:
            Jt = J.transpose(1, 2) # [B, 6, K]
            H = Jt.bmm(J) # [B, 6, 6]
            B = self.inverse(H)
            pinv = B.bmm(Jt) # [B, 6, K]
        except RuntimeError as err:
            # 奇异矩阵...?
            self.last_err = err
            f1 = self.ptnet(p1) # [B, N, 3] -> [B, K]
            r = f1 - f0
            self.ptnet.train(training)
            return r, g, itr

        itr = 0
        r = None
        for itr in range(maxiter):
            self.prev_r = r
            p = self.transform(g.unsqueeze(1), p1) # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            f = self.ptnet(p) # [B, N, 3] -> [B, K]
            r = f - f0

            dx = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

            check = dx.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0 # no update.
                break

            g = self.update(g, dx)
            self.g_series[itr+1] = g.clone()

        rep = len(range(itr, maxiter))
        self.g_series[(itr+1):] = g.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

        self.ptnet.train(training)
        return r, g, (itr+1)
    
    def get_feature_extractor_info(self):
        """获取特征提取器信息"""
        return {
            'name': self.feature_extractor_name,
            'dim_k': self.ptnet.dim_k,
            'available_extractors': FeatureExtractorFactory.get_available_extractors()
        }


# 为了向后兼容，创建别名
PointLK = PointLKWithFeatures 
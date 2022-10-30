import torch
from torch import nn, optim
from torch.nn import functional as F
import rere_config as cnf
import ext.mish as mish
from sklearn.neighbors import KDTree
import numpy as np


# _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RereDML(torch.nn.Module):
    # the feature number of the data points fed into the net
    _d_in = 0
    # The number of classes
    _d_out = 0

    def __init__(self, d_in, d_out):
        super(RereDML, self).__init__()
        self._d_in = d_in
        self._d_out = d_out  #
        self.dml_module = nn.Sequential(
            nn.Linear(d_in, d_in, bias=True),
            mish.Mish(),
            nn.Linear(d_in, d_in, bias=True),
            mish.Mish(),
            nn.Linear(d_in, d_in, bias=True)
        )
        # self.perceptron = nn.Linear(d_in, d_out)

    def encode_dml(self, x):
        xs = self.dml_module(x)
        return xs

    # 计算数据的DML损失，适合mini-batch或者batch数据
    def compute_dml_loss(self, x, label):
        xs = self.encode_dml(x)
        loss = Triplet.compute_dml_loss(xs, label)
        return loss

    def forward(self, x):
        # print(x.dtype)
        x_dml = self.encode_dml(x)
        return x_dml

    def transform(self, x):
        x_dml = self.encode_dml(x)
        return x_dml

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, x, label, output):
        label = label.long()
        # print(label)
        # print(output)
        loss = self.compute_dml_loss(x, label)
        # print("DML loss:", loss)
        return loss


# DML三元组，封装DML距离信息数据
class Triplet:
    def __init__(self, xi, xj, xt, ij_dis, jt_dis):
        self.xi = xi  # 同数据点最近的同类数据点
        self.xj = xj  # 数据点
        self.xt = xt  # 同数据点最近的异类数据点
        self.ij_dis = ij_dis  # 最近同类距离
        self.jt_dis = jt_dis  # 最近异类距离

    @staticmethod
    def __build_triplets(xs, labels):
        triplets = []
        trs = {}  # 字典，存放了标签及其对应的数据子集
        for la in torch.unique(labels):
            trs[la] = xs[labels == la, :]
        # print(list(trs.keys()))
        for la, xs_buck in trs.items():
            # 首先求解同标签最近数据点
            xs_buck0 = xs_buck.detach().cpu().numpy()
            kdt_i = KDTree(xs_buck0, leaf_size=30, metric='euclidean')
            # 如果某类别的样本数小于k，则跳过
            if xs_buck0.shape[0] < 2:
                continue
            indices1 = kdt_i.query(xs_buck0, k=2, return_distance=False)
            xts = None
            # 搜索异类最近的数据点
            # 将所有的异类数据拼接
            for lp, xt_buck in trs.items():
                # 说明是同类，跳过
                if lp == la:
                    continue
                else:
                    if xts is None:
                        xts = xt_buck
                    else:
                        xts = torch.vstack([xts, xt_buck])
            xts0 = xts.detach().cpu().numpy()
            # print(xts.shape)
            kdt_t = KDTree(xts0, leaf_size=30, metric='euclidean')
            indices2 = kdt_t.query(xs_buck0, k=1, return_distance=False)
            # print(indices2.shape)
            for i in range(0, xs_buck.shape[0]):
                xj = xs_buck[i]
                xi = xs_buck[indices1[i, 1]]
                ij_dis = torch.norm(xj - xi, 2)
                xt = xts[indices2[i, 0]]
                jt_dis = torch.norm(xj - xt, 2)
                triplet = Triplet(xi, xj, xt, ij_dis, jt_dis)
                triplets.append(triplet)
        return triplets

    # 基于LSH构建三元组集合
    @staticmethod
    def __build_triplets_with_lsh(xs, labels):
        triplets = []
        label_table = {}
        # 仅用于LSH计算
        xs_t = xs.detach().cpu().numpy()
        # 使用LSH
        from lshashing import LSHRandom
        lsh = LSHRandom(xs_t, hash_len=4, num_tables=1, parallel=True)
        k = 5
        for i, x in enumerate(xs_t):
            # 通过LSH查询K近邻
            xsk = lsh.knn_search(xs_t, x, k, 1, parallel=True)
            sames = []  # 相同标签的近邻
            diffs = []  # 相异标签的近邻
            label_x = labels[i]
            for xx in xsk:
                # print(label_x == labels[xx.index])
                if label_x == labels[xx.index]:
                    sames.append(xx)
                else:
                    diffs.append(xx)
            # xj = xs_buck[rn]
            # 如果相同、相异标签的数据点在k中都存在，则寻找最近数据点并构建三元组
            if len(sames) != 0 and len(diffs) != 0:
                xj = xs[i]
                xi = None
                ij_dis = -1
                xt = None
                jt_dis = -1
                # 搜寻同类最近的数据点
                for rp, xp in enumerate(sames):
                    dis = torch.norm(xj - xs[xp.index], 2)
                    if dis < ij_dis or ij_dis == -1:
                        ij_dis = dis
                        xi = xs[xp.index]
                # 搜索异类最近的数据点
                for rp, xp in enumerate(diffs):
                    # xp = xt_buck[rp]
                    dis = torch.norm(xj - xs[xp.index], 2)
                    if dis < jt_dis or jt_dis == -1:
                        jt_dis = dis
                        xt = xs[xp.index]
                triplet = Triplet(xi, xj, xt, ij_dis, jt_dis)
                triplets.append(triplet)
        return triplets

    @staticmethod
    def __sort_acc(tri):
        return tri.ij_dis - tri.jt_dis

    @staticmethod
    def __compute_dml_loss(triplets, if_anti_noise=False):
        sum_ij = 0
        sum_jt = 0
        if if_anti_noise:
            triplets.sort(key=Triplet.__sort_acc)
            # print([tr.ij_dis-tr.jt_dis for tr in triplets])
            tri_len = len(triplets)
            len_t = int(tri_len * 0.95)
            if len_t < tri_len:
                triplets = triplets[0:len_t]
        # print("Triplets num:", len(triplets))
        sum_ij = sum([t.ij_dis for t in triplets])
        sum_jt = sum([t.jt_dis for t in triplets])
        # return sum_ij / sum_jt
        return sum_ij - sum_jt

    @staticmethod
    def compute_dml_loss(xs, labels, if_use_lsh=False):
        # 仅一条数据、或者无数据、或者只有一种数据类别时，返回0
        if xs.shape[0] <= 1 or len(labels) <= 1:
            return 0
        else:
            triplets = None
            if if_use_lsh:
                triplets = Triplet.__build_triplets_with_lsh(xs, labels)
            else:
                triplets = Triplet.__build_triplets(xs, labels)
            return torch.tensor(0) if len(triplets) == 0 else Triplet.__compute_dml_loss(triplets)

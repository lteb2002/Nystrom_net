import torch
from torch import nn, optim
from torch.nn import functional as F
import dl_model.rere_config as cnf
import ext.mish as mish
import dl_model.svdd.kf as kf
import numpy as np
from dl_model.rere_dml import Triplet as Trip


# _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NystromNetPure(torch.nn.Module):
    """本类用于仅训练Nystrom的变换线性矩阵，而不学习前置的DML模块"""
    # the feature number of the data points fed into the net
    _d_in = 0
    # The sampled data points used in Nystrom
    _samples = None
    # The number of classes
    _d_out = 0
    _nystrom_matrix = None

    def __init__(self, d_in, samples, d_out, if_dml_reg=False, if_fix_nystrom=False, kernel_function='rbf'):
        super(NystromNetPure, self).__init__()
        self._d_in = d_in
        # 将从外部获得的samples转换为torch的相应格式
        self._samples = samples
        sample_size = len(samples)  # calculate the number of samples
        self._sample_size = sample_size
        self._d_out = d_out  #
        self.kernel_function = kernel_function
        self.nystrom_module = nn.Sequential(
            nn.Linear(sample_size, sample_size, bias=False),
            # # nn.LeakyReLU(),
            mish.Mish()
        )
        self.if_fix_nystrom = if_fix_nystrom  # Nystrom是否为固定
        self.if_dml_reg = if_dml_reg
        # Nystrom的变换矩阵有两种办法获得，一种是通过抽样点直接计算，一种是在网络中学习获得
        # 此处设定如果为True，则通过抽样点直接计算，否则，网络自动学习
        # if if_fix_nystrom:
        #     for param in self.nys1.parameters():
        #         param.requires_grad = False
        #     for param in self.nys2.parameters():
        #         param.requires_grad = False
        self.perceptron = nn.Linear(sample_size, d_out)

    # compute the matrix for the last step of linear Nystrom transformation
    def compute_nystrom_matrix(self):
        mx = torch.ones(self._sample_size, self._sample_size).to(self._samples.device)
        sam_new = self.encode_dml(self._samples)
        for i in range(self._sample_size):
            xi = sam_new[i]
            for j in range(i + 1, self._sample_size, 1):
                xj = sam_new[j]
                rbf = torch.exp(-torch.norm(xi - xj))
                mx[i, j] = rbf
                mx[j, i] = rbf
        # print("mx:", mx)
        mx = torch.pinverse(mx)
        # print("inverse of mx:", mx)
        w, v = torch.symeig(mx, eigenvectors=True)
        # w = 1 / w
        w = torch.pow(w, 0.5)
        # print("v norm", torch.norm(v, dim=0))
        # diag = torch.zeros(self._sample_size, self._sample_size).to(self._samples.device)
        diag = torch.diag(w)
        # for i in range(len(w)):
        #     value = w[i]
        #     diag[i, i] = value
        # print("Diag:", diag)
        mx = torch.mm(diag, v.transpose(0, 1))
        # print("Projection matrix:", mx)
        return mx

    # 进行单条数据的NYSTROM映射
    def compute_nys(self, xi, sam_new):
        # Nystrom运算
        kernel_f = kf.KernelFunction.get_kernel_function(self.kernel_function)
        # 新抽样点集构成的矩阵 减 数据点向量，广播
        # 抽样数据矩阵与输入数据点向量的差矩阵按行计算二范数
        # 径向基变换，将d维（原数据维度）变换为m维（抽样数据点数）
        k_value = kernel_f(xi, sam_new)  # 核函数运算结果
        return k_value

    def compute_whole_nys(self, xx, sam_new):
        width = sam_new.shape[1]
        ss = (xx.unsqueeze(1) - sam_new.unsqueeze(0))
        # print(ss.shape)
        dis = ss.reshape(-1, width)
        norms = torch.norm(dis, dim=1)
        # print(norms)
        re = torch.exp(-norms)
        re = re.view(xx.shape[0], sam_new.shape[0])
        # print(re)
        return re

    # 进行数据的NYSTROM步变换
    def encode_nys(self, x):
        # nys_in = torch.zeros(x.shape[0], self._sample_size).to(cnf.device)
        # 每一轮都需要使用DML模块将原抽样点变换，然后再与训练数据进行Nystrom运算
        # the selected representative samples should be transformed in each round of training
        # sam_new = self._samples
        # print(sam_new)
        # rows = x.shape[0]
        # for i in range(rows):
        #     xi = x[i, :]  # 单个训练数据（DML变换后）
        #     nys_in[i, :] = self.compute_nys(xi, sam_new)
        # xs = [self.compute_nys(xi, sam_new) for xi in x.split(1)]
        # nys_in = torch.stack(xs, dim=0)
        nys_in = self.compute_whole_nys(x, self._samples)
        # list comprehension is not supported by pytorch currently yet
        z = nys_in
        if self.if_fix_nystrom:
            # 该数据形态与数学矩阵运算的行列形态相反，因素需要整体转置
            z = torch.matmul(nys_in, self.compute_nystrom_matrix().transpose(0, 1))
        else:
            z = self.nystrom_module(z)
        return z

    def percept(self, nys):
        z = self.perceptron(nys)
        # print(z)
        # z = torch.softmax(z, 1)
        z = F.log_softmax(z, 1)
        # print(z)
        return z

    def forward(self, x):
        # print(x.dtype)
        x_nys = self.encode_nys(x)  #
        z = self.percept(x_nys)
        # print(z)
        return z

    def predict(self, x):
        x_dml = self.encode_dml(x)
        x_nys = self.encode_nys(x_dml)  #
        z = self.percept(x_nys)
        return torch.argmax(z)

    def transform(self, x):
        x_nys = self.encode_nys(x)  #
        return x_nys

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, x, label, output):
        label = label.long()
        # print(label)
        # print(output)
        loss1 = F.cross_entropy(output, label)
        loss2 = 0
        # loss = F.nll_loss(output, label)
        loss = loss1 + loss2
        return loss

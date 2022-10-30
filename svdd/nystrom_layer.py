import torch
from torch import nn
import dl_model.svdd.kf as kf
import ext.mish as mish


class NystromLayer(nn.Module):
    def __init__(self, samples, kernel_function='rbf'):
        super(NystromLayer, self).__init__()
        self._d_in = 0
        self._samples = samples
        if len(samples) != 0:
            self._d_in = samples.shape[1]
        self._sample_size = len(samples)
        self.kernel_function = kernel_function
        self.dml_module = nn.Sequential(
            nn.Linear(self._d_in, self._d_in, bias=True),
            # nn.LeakyReLU(),
            mish.Mish(),
            nn.Linear(self._d_in, self._d_in, bias=True)
        )
        self.nystrom_module = nn.Sequential(
            nn.Linear(self._sample_size, self._sample_size, bias=False),
            # # nn.LeakyReLU(),
            mish.Mish(),
            nn.Linear(self._sample_size, self._sample_size, bias=False)
        )

    def encode_dml(self, x):
        xs = self.dml_module(x)
        return xs

    def encode_nys(self, x):
        # nys_in = torch.zeros(x.shape[0], self._sample_size).to(cnf.device)
        sam_new = self.encode_dml(self._samples)  # 每一轮都需要使用DML模块将原抽样点变换，然后再与训练数据进行Nystrom运算
        # print(sam_new)
        rows = x.shape[0]
        # for i in range(rows):
        #     xi = x[i, :]  # 单个训练数据（DML变换后）
        #     nys_in[i, :] = self.compute_nys(xi, sam_new)
        xs = [self.compute_nys(xi, sam_new) for xi in x.split(1)]
        nys_in = torch.stack(xs, dim=0)
        # list comprehension is not supported by pytorch currently yet
        z = nys_in
        z = self.nystrom_module(z)
        return z

    # 进行单条数据的NYSTROM映射
    def compute_nys(self, xi, sam_new):
        # Nystrom运算
        kernel_f = kf.KernelFunction.get_kernel_function(self.kernel_function)
        # 新抽样点集构成的矩阵 减 数据点向量，广播
        # 抽样数据矩阵与输入数据点向量的差矩阵按行计算二范数
        # 径向基变换，将d维（原数据维度）变换为m维（抽样数据点数）
        k_value = kernel_f(xi, sam_new)  # 核函数运算结果
        return k_value

    def compute_nystrom_matrix(self):
        mx = torch.ones(self._sample_size, self._sample_size).to(self._samples.device)
        for i in range(self._sample_size):
            xi = self.encode_dml(self._samples[i])
            for j in range(i + 1, self._sample_size, 1):
                xj = self.encode_dml(self._samples[j])
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
        diag = torch.zeros(self._sample_size, self._sample_size).to(self._samples.device)
        for i in range(len(w)):
            value = w[i]
            diag[i, i] = value
        # print("Diag:", diag)
        mx = torch.mm(diag, v.transpose(0, 1))
        # print("Projection matrix:", mx)
        return mx

    def forward(self, x):
        x_dml = self.encode_dml(x)
        x_nys = self.encode_nys(x_dml)
        return x_nys

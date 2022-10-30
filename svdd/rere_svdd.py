import torch
from torch import nn, optim
from torch.nn import functional as F
import ext.mish as mish
import dl_model.rere_nystrom as nys
import dl_model.svdd.nystrom_layer as nys_layer
import numpy as np
import dl_model.rere_config as cnf


class RereSVDD(torch.nn.Module):
    _d_in = 0
    _main_label = None
    _trans_method = "normal"
    _sample_size = 100

    def __init__(self, data, h1, h2, d_out, main_label, trans_method="normal"):
        self._main_label = main_label
        super(RereSVDD, self).__init__()
        # self.fc0 = nn.Linear(d_in, d_in)  # 第一层先进行一个随意变换
        self._data = data
        self._h1 = h1
        self._h2 = h2
        self._trans_method = trans_method
        self._d_in = data.shape[1]
        self._d_out = d_out  #
        self._middle = self._d_out
        self.trans_module = self.generate_trans_module(trans_method)

        self.fc21 = nn.Linear(self._middle, d_out)  # 左边学一个均值 mu
        self.fc22 = nn.Linear(self._middle, d_out)  # 右边学一个方差的对数 logvar

    # 生成数据变换模块
    def generate_trans_module(self, trans_method, kernel_function='rbf'):
        trans = None
        if trans_method == "kernel":
            sample_size = 100
            nystrom = nys.Nystrom(self._data.cpu().numpy(), sample_size=sample_size)
            samples = nystrom.get_samples()
            sample_size = len(samples)
            samples = torch.from_numpy(samples.astype(np.float32)).to(cnf.device)
            print("cluster number in Nystrom:", samples.shape[0])
            trans = nys_layer.NystromLayer(samples, kernel_function)
            self._middle = sample_size
        else:
            trans = nn.Sequential(
                nn.Linear(self._d_in, self._h1, bias=True),
                # nn.LeakyReLU(),
                mish.Mish(),
                nn.Linear(self._h1, self._h2, bias=True),
                mish.Mish(),
                nn.Linear(self._h2, self._d_out, bias=True)
                # nn.BatchNorm1d(num_features=d_out)
            )
        return trans

    def update_samples(self):
        if self._trans_method == "kernel":
            data = self.trans_module.encode_dml(self._data)
            nystrom = nys.Nystrom(data.detach().numpy(), sample_size=self._sample_size)
            samples=nystrom.get_samples()
            samples = torch.from_numpy(samples.astype(np.float32)).to(cnf.device)
            self.trans_module._samples = samples

    def encode(self, x, labels):
        x = x.view(-1, self._d_in)
        # x = F.relu(self.fc0(x))
        # main_loc = (labels == self._main_label)
        trans_x = self.trans_module(x)
        mu = self.fc21(trans_x)  # 均值
        logvar = self.fc22(trans_x)  # 方差的对数
        return trans_x, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # 将logvar还原为标准差
        eps = torch.randn_like(std)
        z = mu + eps * std  # 均值+标准差的某个随机倍数，倍数~(0-1)
        return z

    def decode(self, z):
        z = self.decode_module(z)
        z = torch.sigmoid(z)
        return z

    def forward(self, x, labels):
        trans_x, mu, logvar = self.encode(x.view(-1, x.shape[1]), labels)
        # normal = self.reparameterize(mu, logvar)  # 重参数
        # vae_out = self.decode_module(normal)
        return trans_x, mu, logvar

    def transform(self, x, labels):
        # mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        # y = self.trans_module(x)
        y = self.forward(x, labels)[1]
        return y

    def transform_as_nor(self, x):
        trans_x = self.trans_module(x)
        return trans_x

    def transform_as_mu(self, x):
        # mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        # y = self.trans_module(x)
        trans_x = self.trans_module(x)
        mu = self.fc21(trans_x)  # 均值
        return mu

    def transform_as_std(self, x):
        # mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        # y = self.trans_module(x)
        trans_x = self.trans_module(x)
        logvar = self.fc22(trans_x)  # 标准差对数
        std = torch.exp(0.5 * logvar)
        return std

    # SVDD + KL divergence losses summed over all elements and batch
    def loss_function(self, x, label, z, mu, logvar):
        # print(x.shape)
        main_loc = (label == self._main_label)
        x_main = z[main_loc]
        main_size = x_main.size()[0]
        x_rest = z[~main_loc]
        rest_size = x_rest.size()[0]
        # mu_cen = torch.mean(mu, dim=0)
        # logvar_cen = torch.mean(logvar, dim=0)
        # 只有正常数据参与KL散度运算
        mu_main = mu[main_loc]
        var_main = logvar[main_loc]
        # print(mu_cen.size())
        # print(x_main.size())
        # print(mu_cen.size())
        # print(x_main- mu_cen[None,:])
        # print(torch.norm((x_main - mu_cen[None, :]), dim=1).size())
        # MSE = F.mse_loss(vae_out, x[main_loc])
        # loss11 = torch.norm((self.fc21(x_main) - mu_cen[None, :]), dim=1).pow(0.5).sum() / main_size
        # loss12 = torch.norm((self.fc22(x_main) - logvar_cen[None, :]), dim=1).pow(0.5).sum() / main_size
        # loss21 = torch.norm((self.fc21(x_rest) - mu_cen[None, :]), dim=1).pow(0.5).sum() / rest_size
        # loss22 = torch.norm((self.fc22(x_rest) - logvar_cen[None, :]), dim=1).pow(0.5).sum() / rest_size
        loss11 = torch.norm((self.fc21(x_main)), dim=1).sum() / main_size
        loss12 = torch.norm((self.fc22(x_main)), dim=1).sum() / main_size
        loss21 = torch.norm((self.fc21(x_rest)), dim=1).sum() / rest_size
        loss22 = torch.norm((self.fc22(x_rest)), dim=1).sum() / rest_size
        # print(norml2.size())
        # print("Centroids:",mu_cen)
        # print("Variance:", torch.mean(logvar.exp(),dim=0))
        KLD = -0.5 * torch.sum(1 + var_main - mu_main.pow(2) - var_main.exp())
        if KLD > 1.0E16:
            KLD = KLD.pow(0.5)
        # print("Loss 1:", loss11, ",Loss 2:", loss21, ", main num:", main_size, ",KL loss:", KLD)
        # std = torch.exp(0.5 * logvar)
        # print(mu)
        # print("SVDD LOSS:", str(svdd_loss), ",KLD:", str(KLD))
        return loss11 / loss21 + loss12 / loss22 + KLD, loss11, loss12, loss21, loss22, KLD

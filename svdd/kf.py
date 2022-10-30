import torch


# 该类中定义了常用的核函数，供Nystrom网络使用
class KernelFunction:
    def __init__(self):
        pass

    @staticmethod
    def get_kernel_function(name='rbf'):
        if name == 'tanh':
            return KernelFunction.tanh_function
        elif name == 'poly':
            return KernelFunction.poly_function
        elif name == 'laplace':
            return KernelFunction.laplace_function
        else:
            return KernelFunction.radial_basis_function

    @staticmethod
    def radial_basis_function(x1, sample_bulk):
        dis = sample_bulk - x1
        norms = torch.norm(dis, dim=1)
        re = torch.exp(-norms)
        # print(re)
        return re

    @staticmethod
    def laplace_function(x1, sample_bulk):
        dis = sample_bulk - x1
        norms = torch.norm(dis, p=1, dim=1)
        return torch.exp(-norms)

    @staticmethod
    def poly_function(x1, sample_bulk, degree=3):
        d = torch.mm(sample_bulk, x1.t()).t().squeeze()
        return d.pow(degree)

    @staticmethod
    def tanh_function(x1, sample_bulk):
        d = torch.mm(sample_bulk, x1.t()).t().squeeze()
        re = torch.tanh(d)
        # print(re)
        return re

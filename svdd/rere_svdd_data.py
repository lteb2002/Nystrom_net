import torch
import torch.utils.data as tdata
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import dl_model.rere_config as cnf
import time


#

class PyTorchDataSet(tdata.Dataset):
    # 二维数据集
    data = None
    labels = None
    _labels_idx = None
    dim = 0
    label_num = 0

    def _load_data(self, features, labels, normalize):
        # 不包括标签列
        dts = features.astype(np.float32)
        if normalize:
            transformer = MinMaxScaler().fit(dts)
            dts = transformer.transform(dts)
            print("The dataset is normalized...")
        # 标签列
        lts = labels.astype(str)
        self.label_num = len(np.unique(lts))
        lts = np.array(self._convert_label_to_num(lts))
        self.data = torch.from_numpy(dts).to(cnf.device)
        # print(self._data)
        self.labels = torch.from_numpy(lts).to(cnf.device).long()
        self.dim = dts[0, :].shape[0]

    def _convert_label_to_num(self, labels0):
        self._labels_idx = {la: idx for idx, la in enumerate(np.unique(labels0))}
        return [self._labels_idx[la] for la in labels0]

    def __init__(self, features, labels, normalize=True):
        super().__init__()
        self._load_data(features, labels, normalize)

    def __getitem__(self, index: int):
        # print(self.data[index])
        # print(self.labels[index])
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]

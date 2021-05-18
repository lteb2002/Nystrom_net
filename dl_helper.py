import torch
import torch.utils.data as tdata
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import dl_model.rere_config as cnf

_log_interval = 10


def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # data = torch.tensor(data, dtype=torch.float64)
        # print(data)
        # print(label)
        optimizer.zero_grad()
        # recon_batch, mu, logvar = model(data)
        # loss = loss_function(data, recon_batch, mu, logvar)
        result = model(data)
        loss = model.loss_function(data, label, result)
        try:
            loss.backward()
        except RuntimeError as ex:
            pass
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % _log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader),loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.16f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test_reconstruct(epoch, model, loss_function, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            # data = data.to(_device)
            result = model(data)
            test_loss += loss_function(data, *result).item()
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def test_classification(epoch, model, test_loader):
    model.eval()
    right_num = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            # data = data.to(_device)
            result = model(data)
            prds = model.predict(data)
            right_num += len(prds[prds == labels])
    accuracy = right_num / len(test_loader.dataset)
    print('====> Accuracy: {:.4f}'.format(accuracy))


''''''


class ArffDataSet(tdata.Dataset):
    # 二维数据集
    data = None
    labels = None
    _labels_idx = None
    headers = None
    dim = 0
    label_num = 0

    def _load_data(self, file_path, normalize):
        temp = arff.loadarff(file_path)[0]
        temp = pd.DataFrame(temp)
        self.headers = temp.columns.values
        # print(self.headers)
        temp = temp.to_numpy()
        # 不包括标签列
        dts = temp[:, 0:- 1].astype(np.float32)
        if normalize:
            transformer = MinMaxScaler().fit(dts)
            dts = transformer.transform(dts)
            print("The dataset is normalized...")
        lts = temp[:, temp.shape[1] - 1].astype(str)
        self.label_num = len(np.unique(lts))
        lts = np.array(self._convert_label_to_num(lts))
        self.data = torch.from_numpy(dts).to(cnf.device)
        # print(self._data)
        self.labels = torch.from_numpy(lts).to(cnf.device).long()
        self.dim = dts[0, :].shape[0]

    def _convert_label_to_num(self, labels0):
        self._labels_idx = {la: idx for idx, la in enumerate(np.unique(labels0))}
        return [self._labels_idx[la] for la in labels0]

    def __init__(self, file_path, normalize=False):
        super().__init__()
        self._load_data(file_path, normalize)

    def __getitem__(self, index: int):
        # print(self.data[index])
        # print(self.labels[index])
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


def transform_save_to_csv(dataset, model, output):
    new_data = model.transform(dataset.data.to(cnf.device)).detach().cpu().numpy()
    # print(new_data)
    las = ['label_' + x for x in dataset.labels.cpu().numpy().astype(str)]
    # new_data = np.hstack([new_data, np.reshape(dataset.labels.numpy(), (-1, 1))])
    df = pd.DataFrame(new_data)
    df['labels'] = las
    df.to_csv(output, header=True, index=False)


def save_numpy_data_to_csv(data, labels, output):
    # print(new_data)
    las = ['label_' + x for x in labels.astype(str)]
    # new_data = np.hstack([new_data, np.reshape(dataset.labels.numpy(), (-1, 1))])
    df = pd.DataFrame(data)
    df['labels'] = las
    df.to_csv(output, header=True, index=False)


import torch
import torch.utils.data as tdata
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import dl_model.rere_config as cnf
import time

_log_interval = 10


def before_train(log_file):
    f_log = open(log_file, 'w')
    f_log.write("epoch,total_loss,loss1,loss2,KLD,time_cost\n")
    f_log.close()


def train(epoch, model, optimizer, train_loader, log_file=None):
    time_start = time.time()
    model.train()
    train_loss = 0
    loss11_total = 0
    loss12_total = 0
    loss21_total = 0
    loss22_total = 0
    kld_total = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        z, mu, logvar = model(data, labels)
        loss, loss11, loss12, loss21, loss22, kld = model.loss_function(data, labels, z, mu, logvar)
        try:
            loss.backward()
        except RuntimeError as ex:
            pass
        train_loss += loss.item()
        loss11_total += loss11.item()
        loss12_total += loss12.item()
        loss21_total += loss21.item()
        loss22_total += loss22.item()
        kld_total += kld.item()
        optimizer.step()
        if batch_idx % _log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    model.update_samples()
    time_end = time.time()
    if log_file is not None:
        f_log = open(log_file, 'a')
        f_log.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{:.1f}\n'.format(epoch,
                                                                     train_loss / len(train_loader),
                                                                     loss11_total / loss12_total,
                                                                     loss21_total / loss22_total,
                                                                     kld_total / len(train_loader),
                                                                     (time_end - time_start)))
        f_log.flush()


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

    def __init__(self, file_path, normalize=True):
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

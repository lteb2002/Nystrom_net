from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import numpy as np
import torch
from torch import optim
import dl_model.svdd.rere_svdd as m
import dl_model.rere_config as cnf
import dl_model.svdd.rere_svdd_trainer as hl
import dl_model.svdd.rere_svdd_data as ptd
import matplotlib.pyplot as plt
from sklearn import manifold
import dl_model.rere_tsne as tsne


class SvddClassifier(BaseEstimator):
    batch_size = 2000
    epoch_num = 10

    d_h1 = 300
    d_h2 = 100
    d_in = 0
    d_out = 0
    model = None
    max_l = 0  # 多数数据的标签
    classifier = None

    def __init__(self, trans_method="normal"):
        self.trans_method = trans_method

    # 训练神经网络
    def fit(self, features, labels):
        ddd = ptd.PyTorchDataSet(features, labels)
        train_loader = torch.utils.data.DataLoader(ddd, batch_size=self.batch_size, shuffle=True)
        self.d_in = features.shape[1]
        self.d_out = self.d_in
        ls = np.unique(labels)
        print(ls)
        print(len(labels[labels == 0]))
        max_num = 0
        for i in range(0, len(ls)):
            num = len(labels[labels == i])
            if num > max_num:
                self.max_l = i
                max_num = num
        print("max_label:", self.max_l, ",max_num:", max_num)
        # print(d_in)
        self.model = m.RereSVDD(ddd.data, self.d_h1, self.d_h2, self.d_out, self.max_l,
                                self.trans_method).to(cnf.device)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        for epoch in range(1, self.epoch_num + 1):
            hl.train(epoch, self.model, optimizer, train_loader)
        # 训练完成，测试训练效果
        xs_torch = torch.from_numpy(features).to(cnf.device)
        xts1 = self.model.transform_as_mu(xs_torch)
        xts2 = self.model.transform_as_std(xs_torch)
        avg_var = torch.mean(xts1, dim=1).detach().numpy()
        # print(avg_var, np.shape(avg_var))
        l2_mu = torch.norm(xts1, dim=1).detach().numpy()
        l2_std = torch.norm(xts2, dim=1).detach().numpy()
        xts = np.vstack((l2_mu, l2_std)).T
        print(xts, labels)
        self.classifier = svm.SVC()
        self.classifier.fit(xts, labels)
        # print(l2_mu, np.shape(l2_mu))
        # print(np.max(l2_mu), np.min(l2_mu))

    def predict(self, xs):
        # xs是输入的多个向量，本方法可以预测多个向量的标签
        # 变换输入数据
        xs_torch = torch.from_numpy(xs).to(cnf.device)
        xts1 = self.model.transform_as_mu(xs_torch)
        xts2 = self.model.transform_as_std(xs_torch)
        avg_var = torch.mean(xts1, dim=1).detach().numpy()
        # print(avg_var, np.shape(avg_var))
        l2_mu = torch.norm(xts1, dim=1).detach().numpy()
        l2_std = torch.norm(xts2, dim=1).detach().numpy()
        xts = np.vstack((l2_mu, l2_std)).T
        # prd = np.where(np.abs(l22 - 1) >= 0.01, self.max_l, self.max_l + 1)
        prd = self.classifier.predict(xts)
        print(prd)
        return prd

    def transform(self, xs):
        # xs是输入的多个向量，本方法可以将数据变换为易识别的模式
        # 变换输入数据
        xs_torch = torch.from_numpy(xs).to(cnf.device)
        xts1 = self.model.transform_as_mu(xs_torch)
        xts2 = self.model.transform_as_std(xs_torch)
        # avg_var = torch.mean(xts1, dim=1).detach().numpy()
        # print(avg_var, np.shape(avg_var))
        l2_mu = torch.norm(xts1, dim=1).detach().numpy()
        l2_std = torch.norm(xts2, dim=1).detach().numpy()
        xts = np.vstack((l2_mu, l2_std)).T
        return xts

    def transform_as_mu(self, xs):
        # xs是输入的多个向量，本方法可以将数据变换为易识别的模式
        # 变换输入数据
        xs_torch = torch.from_numpy(xs).to(cnf.device)
        xts1 = self.model.transform_as_mu(xs_torch).detach().numpy()
        xts2 = self.model.transform_as_std(xs_torch).detach().numpy()
        xts = np.hstack((xts1, xts2))
        return xts

    def save_high_dimension_img(self, img3, xts, labels):
        viz2 = tsne.RereTSNE(xts, labels)
        viz2.save_image(img3)


if __name__ == "__main__":
    fp = 'H:\\svdd_experiments\\'
    fp2 = 'H:\\svdd_experiments\\anomaly_detection\\'
    fn = 'bank_marketing'
    trans_method = "kernel"  # kernel | normal
    input_file = fp + fn + '.arff'
    output1 = fp2 + fn + "_" + trans_method + '_mu_std.csv'
    output2 = fp2 + fn + "_" + trans_method + '_trans.csv'
    output_img1 = fp2 + fn + "_" + trans_method + '_mu_std.png'
    output_img2 = fp2 + fn + "_" + trans_method + '_trans.png'
    data_set = hl.ArffDataSet(input_file)
    fts = data_set.data.detach().numpy()
    lts = data_set.labels.detach().numpy()
    clf = SvddClassifier(trans_method)
    clf.fit(fts, lts)
    # 转换为均值和方差的2范数
    trans_data1 = clf.transform(fts)
    hl.save_numpy_data_to_csv(trans_data1, lts, output1)
    clf.save_high_dimension_img(output_img1, trans_data1, lts)
    # 转换为均值的输出
    trans_data2 = clf.transform_as_mu(fts)
    hl.save_numpy_data_to_csv(trans_data2, lts, output2)
    clf.save_high_dimension_img(output_img2, trans_data2, lts)
    from sklearn.model_selection import cross_val_score
    # scores = cross_val_score(clf, fts, lts, cv=3, scoring="recall")
    # print(np.average(scores))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')
from sklearn import manifold


class RereTSNE:
    _max_samples = 1000
    _max_viz = 500
    _samples = None
    _labels = []
    _transformed_dataset = None
    _random_seed = 1  # 随机数种子，保证可重复实验

    def __init__(self, dataset, labels=[]):
        # 如果数据量太大，仅采样_max_samples个样本
        if dataset.shape[0] > self._max_samples:
            np.random.seed(seed=self._random_seed)
            ids = np.random.choice(range(dataset.shape[0]), self._max_samples, replace=False)
            self._samples = dataset[ids, :]
            if len(labels) > 0:
                self._labels = labels[ids]
        else:
            self._samples = dataset
            self._labels = labels
        # 使用TSNE降为二维，以便可视化
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=self._random_seed)
        self._transformed_dataset = tsne.fit_transform(self._samples)

    def save_image(self, file_path, if_show=False):
        RereTSNE.save_image2(self._transformed_dataset,self._labels,file_path,if_show)

    @staticmethod
    def save_image2(reduced_data, reduced_labels, file_path, if_show=False):
        plt.figure(figsize=(4, 3))
        axs = plt.gca()
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        # 设置X轴标签
        plt.xlabel('x1')
        # 设置Y轴标签
        plt.ylabel('x2')
        if len(reduced_labels) > 0:
            colors = ['RoyalBlue', 'DarkOrange', 'k', 'Indigo', 'SeaGreen', 'Olive', 'DarkSlateGray', 'Orchid',
                      'YellowGreen', 'Wheat']
            markers = ['.', '*', '+', '_', 'x', '1', '2', '3', '4', 'v']
            for i, label in enumerate(np.unique(reduced_labels)):
                cols = reduced_labels == label
                if i >= 10:
                    i = i % 10
                # print(i)
                plt.scatter(reduced_data[cols, 0], reduced_data[cols, 1],
                            c=colors[i], marker=markers[i],s=30)
        else:
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
        plt.savefig(file_path,dpi=800)
        if if_show:
            plt.show()


if __name__ == '__main__':
    import dl_model.dl_helper as hl

    fp = 'E:\\papers\\svdd\\datasets\\'
    name = 'credit_card_chi'
    fn = fp + name + '.csv'
    pfn = fp + name + '.png'
    # arff = hl.ArffDataSet(fp)
    # dataset = arff.data.cpu().numpy()
    # labels = arff.labels.cpu().numpy()
    # print(dataset.shape[1])
    # tsne = RereTSNE(dataset, labels)

    temp = pd.read_csv(fn)
    headers = temp.columns.values
    temp = temp.to_numpy()
    data = temp[:, 0:- 1]
    tsne = RereTSNE(data)
    tsne.save_image(pfn, True)

import dl_model.dl_helper as hl
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans,OPTICS,DBSCAN
import uuid
from sklearn.metrics import *
import numpy as np


# 本类封装了Nystrom变换的基本操作
# 本类使用Numpy进行数值计算，不涉及GPU运算
class Nystrom:
    _vt_matrix = None  # Nystrom的线性投影矩阵
    _dataset = None  # 进行Nystrom变换的全体数据集
    _samples = None  # 抽样点集（簇中心）
    _sample_size = 0  # 抽样点数量（变换后的数据维度）
    _threshold_of_q = 0.95 # 评估簇内数据聚集程度的阈值
    _eliminating_noise_rate = 0.05

    def __init__(self, dataset, sample_size=0):
        self._dataset = dataset
        if sample_size > 0:
            self._sample_size = sample_size
            self._samples = self.__cluster(dataset, sample_size)  # 执行聚类操作
        else:
            cls = self.__evaluate_adaptive_clusters(dataset)
            print("estimated cluster number:", len(cls))
            self._samples = self._extract_centroids(cls)
            self._sample_size = self._samples.shape[0]
            print("estimated cluster number after eliminating noises:", self._sample_size)

    # perform clustering
    def __cluster(self, dataset, cluster_num):
        # model = GaussianMixture(n_components=cluster_num,covariance_type='diag').fit(dataset)
        # return model.means_
        model = KMeans(n_clusters=cluster_num).fit(dataset)
        return model.cluster_centers_
        # model = DBSCAN(eps=0.12, min_samples=100).fit(dataset)
        # return model.components_

    # 自适应评估簇中心和簇数量
    def __evaluate_adaptive_clusters(self, dataset, clusters=[], init_c_num=2):
        # model = GaussianMixture(n_components=init_c_num).fit(dataset)
        # cluster_labels = model.predict(dataset)  # 所有数据点的归属簇编号
        # current_centers = model.means_  # 所有的簇中心点，行向量为簇中心，行数为簇数量
        model = KMeans(n_clusters=init_c_num).fit(dataset)
        cluster_labels = model.labels_  # 所有数据点的归属簇编号
        current_centers = model.cluster_centers_  # 所有的簇中心点，行向量为簇中心，行数为簇数量
        label_num = current_centers.shape[0]  # 簇的数量
        for c_no in range(label_num):
            # 通过比对数据点的归属簇编号与当前簇号，获得数据行的布尔向量，筛选出当前簇下的数据点
            sub = dataset[cluster_labels == c_no, :]
            # 对该簇下的数据点集进行奇异值分解
            svs = np.linalg.svd(sub, compute_uv=False)
            # 计算首奇异值平方所占所有奇异值平方和的比重，并估计簇数量
            q, c_num = self.compute_q_and_estimate_cluster_num(svs)
            if q >= self._threshold_of_q:  # 如果该簇内数据聚集度好,直接将簇中心加入代表性数据点
                cl = RereCluster(current_centers[c_no], sub)
                clusters.append(cl)
            else:  # 否则，递归调用本函数，将子数据集再进行聚类
                self.__evaluate_adaptive_clusters(sub, clusters, c_num)
        return clusters

    # 计算q的值并估计簇的数量
    def compute_q_and_estimate_cluster_num(self, svs):
        svs = svs ** 2
        total = svs.sum()
        q = svs[0] / total
        c_num = 0
        ss = 0
        for idx, s in enumerate(svs):
            ss += s
            if ss / total > self._threshold_of_q:
                c_num = idx + 1
                break
        return q, c_num

    # 过滤簇并将簇中心规整为矩阵
    def _extract_centroids(self, clusters):
        clusters = sorted(clusters, key=lambda cl0: cl0.dataset.shape[0], reverse=True)
        v_num = 0
        total = self._dataset.shape[0]
        ss = 0
        for ind, cl in enumerate(clusters):
            s = cl.dataset.shape[0]
            # print("samples number:", s)
            ss += s
            if ss / total >= 1 - self._eliminating_noise_rate:
                v_num = ind + 1
                break
        clusters = clusters[0:v_num]
        cs = [cl.centroid for cl in clusters]
        return np.matrix(cs)

    # get the samples used in Nystrom method
    def get_samples(self):
        return self._samples


class RereCluster:
    _id = uuid.uuid1()
    centroid = None
    dataset = None

    def __init__(self, centroid, dataset):
        self.centroid = centroid
        self.dataset = dataset


if __name__ == '__main__':
    import dl_model.dl_helper as hl

    fp = 'H:\\nystrom_experiment\\magic04.arff'
    data_set = hl.ArffDataSet(fp).data.numpy()
    print(data_set.shape[1])
    nys = Nystrom(data_set)
    samples = nys.get_samples()
    print("The number of sampled data points:", samples.shape[0])
    for i in range(len(samples)):
        sam = samples[i]
        print(sam)

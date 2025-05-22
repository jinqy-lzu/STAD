from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.clarans import clarans
from pyclustering.utils.metric import distance_metric, type_metric

import numpy as np
import pandas

class EnsembleClustering:
    def __init__(self, n_clusters, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        # Initialize different clustering algorithms
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.gmm = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
        self.dbscan = DBSCAN(eps=9, metric="euclidean",min_samples=2, algorithm="auto")
        #层次聚类
        self.clara = AgglomerativeClustering(n_clusters=self.n_clusters,metric="euclidean")
        self.initial_medoids = list(range(self.n_clusters))
        # self.mean_shift = MeanShift()#均值漂移
        
    def fit_predict(self, X):
        # Fit each clustering algorithm
        kmeans_labels = self.kmeans.fit_predict(X)
        clara_labels = self.clara.fit_predict(X)
        # dbscan_labels = self.dbscan.fit_predict(X)
        gmm_labels = self.gmm.fit_predict(X)
        #PAM
        metric = distance_metric(type_metric.EUCLIDEAN)
        pam_instance = kmedoids(X, self.initial_medoids,metric=metric)#metric="euclidean"
        pam_instance.process()
        pam_labels = np.zeros(len(X))
        for cluster_id, cluster in enumerate(pam_instance.get_clusters()):
            for index in cluster:
                pam_labels[index] = cluster_id
        
         # CLARA
        
        clarans_instance = clarans(X.tolist(), numlocal=5, maxneighbor=1, number_clusters=self.n_clusters)
        clarans_instance.process()
        clarans_labels = np.zeros(len(X))
        for cluster_id, cluster in enumerate(clarans_instance.get_clusters()):
            for index in cluster:
                clarans_labels[index] = cluster_id
        # Aggregate labels from each algorithm
        all_labels = np.vstack((kmeans_labels, clarans_labels, clara_labels, pam_labels, gmm_labels))
        
        # Voting to determine final labels
        final_labels = []
        for i in range(all_labels.shape[1]):
            labels, counts = np.unique(all_labels[:, i], return_counts=True)
            final_labels.append(labels[np.argmax(counts)])
        
        return np.array(final_labels)
def get_train_data(filepath, select_data = None):
    df = pandas.read_csv(
                filepath,
                skip_blank_lines = True,
                )
    #筛选基因
    merged_df = pandas.merge(df, select_data, left_on='gene_name', right_on='gene', how='inner')
    #保存
    merged_df.to_csv('/home/chenkp/CODA/gastric/clustering/STAD-66-geneExp.csv', index=True)
    # 删除列
    merged_df = merged_df.drop("gene_name", axis=1)
    samples_name = merged_df.columns.values
    df_exp = merged_df.values
    df_exp = df_exp.transpose()
    return samples_name, df_exp
# Example usage
if __name__ == "__main__":
   
    clusters = 2
    file_path = "/home/chenkp/CODA/gastric/clustering/STAD-counts.csv"
    gene_list = "/home/chenkp/CODA/gastric/clustering/8.6-gastric cancer-66 gene.csv"
    label_file = "/home/chenkp/CODA/gastric/clustering/STAD_66gene_label.csv"
    select_data = pandas.read_csv(gene_list)
    ensemble = EnsembleClustering(n_clusters=clusters, random_state=42)
    samples_name, exp = get_train_data(file_path, select_data)
    exp = exp[:-1]
    exp = exp.astype(float)
    exp = np.log2(exp+1)
    # Fit and predict
    final_labels = ensemble.fit_predict(exp)
    
    # 创建 DataFrame，将 samples_name 和 final_labels 合并
    data = {'SampleName': samples_name[:-1], 'Label': final_labels}
    df = pandas.DataFrame(data)

    # 写入 CSV 文件
    df.to_csv(label_file, index=False)
    # Print the final labels
    print("Final labels:", final_labels)

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
        # #Organizing in progress............
        
    def fit_predict(self, X):
        # Fit each clustering algorithm
        # #Organizing in progress............
        
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
    
    data = {'SampleName': samples_name[:-1], 'Label': final_labels}
    df = pandas.DataFrame(data)


    df.to_csv(label_file, index=False)
    # Print the final labels
    print("Final labels:", final_labels)

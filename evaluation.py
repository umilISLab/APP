import os
import time
import dill
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from pympler import asizeof
from tqdm.notebook import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from clustering import AffinityPropagation, PosterioriAffinityPropagation


def similarity(X: np.array, y: np.array = None) -> np.array:
    """Compute the similarity matrix between each pair from a vector array X and Y

    Args:
        X(np.array): input array.
        y(np.array, optional, default=None): input array. If None, y = X.

    Returns:
        np.array, similarity matrix
    """
    if y is None:
        y = X

    return -euclidean_distances(X, y)


def mapping(X: pd.DataFrame) -> pd.DataFrame:
    """
    Perform the mapping between predicted labels and actual labels.

    Args:
        X(np.array): data containing predicted label (y) and true label (y_true).

    Returns:
        pd.DataFrame, data with mapped labels (y_new)
    """
    for label in X.y.unique():
        y_cluster = pd.DataFrame(X[X.y == label].groupby('y_true').size(), columns=['size'])
        new_label = y_cluster.index[y_cluster['size'] == y_cluster['size'].max()].tolist()[0]
        rowIndex = X.index[X['y'] == label]
        X.loc[rowIndex, 'y_new'] = new_label
    return X

def purity(y: np.array, y_true: np.array) -> float:
    """
    Return clustering accuracy.

    Args:
        y(np.array): y predicted.
        y_true(np.array): gold labels.

    Returns:
        float, clustering accuracy
    """
    return round((y == y_true).sum() / y_true.shape[0], 3)

def memory_usage(obj) -> float:
    """
    Returns the size (in MB) of the object passed as argument.

    Args:
        obj(object): input object.

    Returns:
        float, size in MB"""
    return asizeof.asizeof(obj) * 10 ** -6

def sum_of_similarity(y: np.array, centers: np.array, S: np.array) -> float:
    """
    Compute sum of similarity between objects and their exemplar.

    Args:
        y(np.array): object labels.
        centers(np.array): labels of the object examplars.
        S(np.array): similarity matrix

    Returns:
        float, sum of similarity
    """
    # clustering didn't converge
    if len(centers) < 1:
        return 0

    return sum([S[i, centers[y[i]]] for i in range(S.shape[0])])

# https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
#def purity(y_pred, y_true):
    # compute contingency matrix (also called confusion matrix)
    #contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    #return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def evaluation(clustering: object, dataset: str, step: int, X: pd.DataFrame, stats: dict):
    """
    Collect several evaluation data on the clustering step.

    Args:
        clustering(object): the clustering instance that performed the clustering activity.
        dataset(str): the name of the dataset.
        step(int): the i-th step of the clustering.
        X(pd.DataFrame): the dataset.
        stats(dict): the object in which to store the statistics.
    """

    stats[dataset]['computational_time'][step].append(clustering.computational_time)

    stats[dataset]['number_of_iteration'][step].append(clustering.n_iter_)

    purity_score = purity(X.y_new, X.y_true)
    stats[dataset]['purity'][step].append(purity_score)

    # we remove memory_usage(clustering.to_remove) it keeps track of the information removed for analysis purposes.
    # This variable would not be used otherwise
    mega_bytes = memory_usage(clustering) - memory_usage(clustering.to_remove)
    stats[dataset]['memory_usage'][step].append(mega_bytes)

    mutual_information_score = metrics.normalized_mutual_info_score(X.y, X.y_true)
    stats[dataset]['normalized_mutual_info_score'][step].append(mutual_information_score)

    # try:
    #    sum_of_similarity_score = sum_of_similarity(X.y.values, clustering.cluster_centers_indices_, clustering.S)
    # except:
    #    sum_of_similarity_score = 0
    # stats[dataset]['sum_of_similarity'][step].append(sum_of_similarity_score)

    # tmp = X.drop(['y', 'y_new', 'y_true'], axis=1)
    # try:
    #    silhoutte_score = metrics.silhouette_score(tmp, X.y_new, metric='sqeuclidean')
    # except:
    #    silhoutte_score = 0
    # stats[dataset]['silhouette'][step].append(silhoutte_score)

    # homogeneity_score = metrics.homogeneity_score(X.y_true, X.y_new)
    # stats[dataset]['homogeneity_score'][step].append(homogeneity_score)

    # completeness_score = metrics.completeness_score(X.y_true, X.y_new)
    # stats[dataset]['completeness_score'][step].append(completeness_score)

    # adjusted_rand_score = metrics.adjusted_rand_score(X.y_true, X.y_new)
    # stats[dataset]['adjusted_rand_score'][step].append(adjusted_rand_score)

    # v_measure_score = metrics.v_measure_score(X.y_true, X.y_new)
    # stats[dataset]['v_measure_score'][step].append(v_measure_score)

    stats[dataset]['number_of_clusters'][step].append(np.unique(clustering.labels_).shape[0])

    tmp = X.copy()
    tmp['iter'] = step
    tmp = tmp[['iter', 'y', 'y_true', 'y_new']]
    
    if len(stats[dataset]['all']) == 0:
        stats[dataset]['all'] = list() 
   
    if step == 0:
        stats[dataset]['all'].append(tmp)
    else:
        last = stats[dataset]['all'].pop()
        stats[dataset]['all'].append(pd.concat([last, tmp]))


def ap_clustering(dataset: str, dfs: list, n_step, stats: dict, columns_to_scale:list=None, columns_to_encode:list=None):
    """
    Perform AP clustering with stream data.

    Args:
        dataset(str): the name of the dataset.
        dfs(list): list of DataFrame, i.e. pre-computed dataset partitions.
        stats(dict): the object in which to store the statistics.
    """

    for step in range(0, n_step + 1):

        # get data
        df = pd.concat(dfs[:step + 1]).fillna(0)

        # get features
        X = df.drop('label', axis=1).to_numpy()

        # pre-compute similarity
        S = similarity(X, X)

        # compute preference
        if dataset == 'yeast':
            preference = S.min() - 0.011 * S.shape[0]
        else:
            preference = None

        # clustering
        AP = AffinityPropagation(damping=damping, affinity=similarity, preference=preference,
                                 max_iter=max_iter, convergence_iter=convergence_iter,
                                 columns_to_scale=columns_to_scale, columns_to_encode=columns_to_encode)

        # clustering
        start_time = time.time()
        AP.fit(X)
        AP.computational_time = time.time() - start_time

        # X_global
        X_global = df
        X_global = X_global.rename(columns={"label": "y_true"})
        X_global['y'] = AP.labels_

        # mapping
        X_global = mapping(X_global)

        evaluation(AP, dataset, step, X_global, stats)

def iapna_clustering(dataset: str, dfs: list, n_step, stats: dict, freezing: bool = False,
                     aging_index: int = 0, columns_to_scale:list=None, columns_to_encode:list=None):
    """
    Perform IAPNA(freeze) clustering with stream data.

    Args:
        dataset(str): the name of the dataset.
        dfs(list): list of DataFrame, i.e. pre-computed dataset partitions.
        stats(dict): the object in which to store the statistics.
        freezing(bool, optional, default=False): If True, clusters will be frozen.
        aging_index(TODO)
    """

    # clustering
    IAPNA = AffinityPropagation(damping=damping, affinity=similarity, freezing=freezing,
                                max_iter=max_iter, convergence_iter=convergence_iter,
                                aging_index=aging_index, columns_to_encode=columns_to_encode,
                                columns_to_scale=columns_to_scale)

    for step in range(0, n_step + 1):

        # get data
        df = dfs[step].fillna(0)

        # get features
        X = df.drop('label', axis=1).to_numpy()

        # pre-compute similarity
        S = similarity(X, X)

        # compute preference
        if dataset == 'yeast':
            preference = S.min() - 0.011 * S.shape[0]
        else:
            preference = None

        IAPNA.preference = preference

        # clustering
        start_time = time.time()
        IAPNA.fit(X)
        IAPNA.computational_time = time.time() - start_time

        # X_global
        X_global = pd.concat(dfs[:step + 1]).reset_index(drop=True)
        X_global = X_global.rename(columns={"label": "y_true"})
        # tmp
        for i in IAPNA.to_remove:
            X_global = X_global.drop(i).reset_index(drop=True)
        X_global['y'] = IAPNA.labels_

        # mapping
        X_global = mapping(X_global)

        evaluation(IAPNA, dataset, step, X_global, stats)

def app_clustering(dataset: str, dfs: list, n_step, stats: dict, freezing: bool = False,
                   aging_index: int = 0, columns_to_scale:list=None, columns_to_encode:list=None):
    """
    Perform IAPNA(freeze) clustering with stream data.

    Args:
        dataset(str): the name of the dataset.
        dfs(list): list of DataFrame, i.e. pre-computed dataset partitions.
        stats(dict): the object in which to store the statistics.
        freezing(bool, optional, default=False): If True, clusters will be frozen.
        aging_index(TODO)
    """
    # clustering
    APP = PosterioriAffinityPropagation(damping=damping, affinity=similarity, freezing=freezing,
                                        max_iter=max_iter, convergence_iter=convergence_iter,
                                        aging_index=aging_index, columns_to_scale=columns_to_scale,
                                        columns_to_encode=columns_to_encode, exemplar_pack=exemplar_pack)

    for step in range(0, n_step + 1):

        # get data
        df = dfs[step].fillna(0)

        # get features
        X = df.drop('label', axis=1).to_numpy()

        # pre-compute similarity
        S = similarity(X, X)

        # preference
        # APP.preference = S.min() - pc * S.shape[0]
        APP.preference = None

        # clustering
        start_time = time.time()
        APP.fit(X)
        APP.computational_time = time.time() - start_time

        # X_global
        X_global = pd.concat(dfs[:step + 1]).reset_index(drop=True)
        X_global = X_global.rename(columns={"label": "y_true"})
        # tmp
        for i in APP.to_remove:
            X_global = X_global.drop(i).reset_index(drop=True)
        X_global['y'] = APP.labels_

        # mapping
        X_global = mapping(X_global)

        evaluation(APP, dataset, step, X_global, stats)

def clustering_evaluation(dataset:str, mode:str, samples:list, n_step=5, freezing=True, aging_index: int = 0,
                          columns_to_scale:list=None, columns_to_encode:list=None) -> dict:
    """
    Evaluate a clustering algorithm on a set of precomputed random samples. The evaluation is performed over different dataset.

    Args:
        mode(str): the name of the clustering algorithm you want evaluate.
        fair(bool, optional, default=True): if False, the evaluation is performed in a evolution scenario.
        freezing(bool, optional, default=True): this parameter enables the cluster freezing. It works only for mode='iapna' and mode='app'.
        aging_index(TODO).
        start_size(int, default=1): this parameters allows to bias the sample. The first time a group appears, it will appear with at least start_size items
        min_size_local(int, default=2): this parameters allows to bias the sample. At each time step there will be at least min_size objects or 0 for each group
        min_size_global(int, default=2): this parameters allows to bias the sample. At each time step there will be at least min_size objects in total

    Returns:
        dict, statistics for each dataset, each time step, and each sample.
    """

    # statistics fo each step and for each dataset
    # {dataset: {measure: {step: list}}}
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    freezing = True if freezing is True else False

    for dfs in tqdm(samples, desc=f'Evaluation {dataset.upper()} dataset'):

        if mode == 'ap':
            ap_clustering(dataset, dfs, n_step, stats, columns_to_scale, columns_to_encode)
        elif mode == 'iapna':
            iapna_clustering(dataset, dfs, n_step, stats, freezing, aging_index, columns_to_scale, columns_to_encode)
        elif mode == 'app':
            app_clustering(dataset, dfs, n_step, stats, freezing, aging_index, columns_to_scale, columns_to_encode)

    return stats

def dump(result, path:str):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    dill.dump(result, open(path, mode='wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_step", default=5, type=int, help="time steps")
    parser.add_argument("--iap", default='ap', type=str, help="the incremental clusterin algorithm")
    parser.add_argument("--freezing", default=None, type=bool, help="true for freezing clusters")
    parser.add_argument("--aging_index", default=0, type=int, help="aging index")
    parser.add_argument("--convergence_iter", type=int, default=15, help="convergence iter")
    parser.add_argument("--max_iter", default=200, type=int, help="max_iter of AP")
    parser.add_argument("--damping", type=float, default=0.9, help="damping factor")
    parser.add_argument("--dataset", type=str, help="the dataset on which the clustering is performed")
    parser.add_argument("--output", type=str, help="the path to the file where the clustering results will be stored")
    parser.add_argument("--input", type=str, help="the path to containing the samples")
    parser.add_argument("--pack0", type=str, default="exemplar", help="True to replace the centroids with the exemplars in 0th step")
    args = parser.parse_args()

    # Dataset with preference coefficients
    dataset = args.dataset

    # clustering parameters
    damping = args.damping
    max_iter = args.max_iter
    convergence_iter = args.convergence_iter
    freezing = args.freezing
    aging_index = args.aging_index
    algo = args.iap
    exemplar_pack = (args.pack0 == 'exemplar')

    # io
    inp, out = args.input, args.output

    # sample parameters
    samples = dill.load(open(inp, mode='rb'))
    n_step = args.n_step

    columns = [c for c in samples[0][0].columns.values if c!='label']

    if dataset in ['iris', 'wine', 'yeast', 'wdbc', 'fc']:
        columns_to_scale = [i for i, _ in enumerate(columns)]
        columns_to_encode = None
    elif dataset == 'car':
        columns_to_scale = None
        columns_to_encode = [i for i, _ in enumerate(columns)]
    elif dataset == 'kddcup':
        columns_to_scale = [i for i, c in enumerate(columns) if c not in [1,2,3,6,11,20,21]]
        columns_to_encode = [i for i, c in enumerate(columns) if c in [1,2,3,6,11,20,21]]

    stats = clustering_evaluation(dataset=dataset, mode=algo, samples=samples, n_step=n_step,
                                  freezing=freezing, aging_index=aging_index,
                                  columns_to_scale=columns_to_scale,
                                  columns_to_encode=columns_to_encode)
    
    #for i in range(100):
    #    stats[dataset]['all'][i].to_csv(f'csv/{dataset}/{i}.csv', index=False)
    del stats[dataset]['all']
    dump(stats, out)

import os
import argparse
import dill
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Manager, cpu_count

def sampling(df: pd.DataFrame, labels: list, n_obj: int) -> pd.DataFrame:
    """
    Sample n objects for each label of a dataset.

    Args:
        df(pd.DataFrame): dataset wrapped in a pandas DataFrame.
        labels(list): input labels.
        n_obj(int): number of objects for label.

    Returns:
        pd.DataFrame, a reduced dataset.
    """

    dfs = [df[df.label == label][:n_obj] for label in labels]
    return pd.concat(dfs)

def iris() -> pd.DataFrame:
    """IRIS dataset

    Returns:
        pd.DataFrame, iris dataset"""

    # read data from csv
    df = pd.read_csv('datasets/iris/iris.data',
                     sep=",",
                     names=["sepal_length", "sepal_width", "petal_length", "petal_width", "label"])

    # map categorical values to numeric ids
    unique_label = df['label'].unique()
    label2id = {label: i for i, label in enumerate(unique_label)}
    df['label'] = df['label'].apply(lambda x: label2id[x])

    # shuffle data
    df = df.sample(frac=1)

    #mms = MinMaxScaler()
    #continuous_columns = [c for c in df.columns.values if c != 'label']
    #df[continuous_columns] = mms.fit_transform(df[continuous_columns])

    return df

def wine() -> pd.DataFrame:
    """WINE dataset

    Returns:
        pd.DataFrame, wine dataset"""

    # read data from csv
    df = pd.read_csv('datasets/wine/wine.data', sep=",", names=["label", "alcohol", "malic_acid",
                                                       "ash", "alcalinity_of_ash",
                                                       "magnesium", "total_phenols",
                                                       "flavanoids", "nonflavanoid_phenols",
                                                       "proanthocyanins", "color_intensity",
                                                       "hue", "OD280/OD315_of_diluted wines",
                                                       "proline"])
    # map categorical values to numeric ids
    unique_label = df['label'].unique()
    label2id = {label: i for i, label in enumerate(unique_label)}
    df['label'] = df['label'].apply(lambda x: label2id[x])

    # shuffle data
    df = df.sample(frac=1)

    #mms = MinMaxScaler()
    #continuous_columns = [c for c in df.columns.values if c != 'label']
    #df[continuous_columns] = mms.fit_transform(df[continuous_columns])

    return df

def car() -> pd.DataFrame:
    """CAR dataset

    Returns:
        pd.DataFrame, car dataset"""

    # read data from csv
    df = pd.read_csv('datasets/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])

    # number of record for class
    n = 65

    df = sampling(df, df.label.unique(), n)

    # map categorical values to numeric ids
    for column in df.columns.values:
        unique_label = df[column].unique()
        label2id = {label: i for i, label in enumerate(unique_label)}
        df[column] = df[column].apply(lambda x: label2id[x])

    # shuffle data
    df = df.sample(frac=1)

    return df

def kddcup() -> pd.DataFrame:
    """KDDCUP dataset

    Returns:
        pd.DataFrame, yeast dataset"""

    # read data from csv
    df = pd.read_csv('datasets/kddcup/kddcup.csv', names=list(range(41)) + ["label"]).reset_index(drop=True)

    # top 11 label
    top_label = ['smurf.', 'neptune.', 'normal.', 'satan.', 'ipsweep.', 'portsweep.',
                 'nmap.', 'back.', 'warezclient.', 'teardrop.', 'pod.']

    df = df[np.isin(df.label.values, top_label)]

    df_tmp = list()
    for label in top_label:
        df_tmp.append(df[df.label == label].sample(frac=1)[:264])

    df = pd.concat(df_tmp).sample(frac=1)

    # map categorical values to numeric ids
    for column in [1,2,3,6,11,20,21,'label']:
        unique_label = df[column].unique()
        label2id = {label: i for i, label in enumerate(unique_label)}
        df[column] = df[column].apply(lambda x: label2id[x])

    # shuffle data
    # df = df.sample(frac=1)
    #mms = MinMaxScaler()
    #continuous_columns = [c for c in df.columns.values if c not in [1,2,3,6,11,20,21,'label']]
    #df[continuous_columns] = mms.fit_transform(df[continuous_columns])

    return df


def incremental_split(df: pd.DataFrame, n_step: int, start: int, n_obj: int) -> list:
        """
        Split a dataset in multiples partiton.

        Args:
            df(pd.DataFrame): dataset wrapped in a pandas DataFrame.
            start(int): objects available at time-step 0.
            n_obj(int): objects available at each successive time-step.
            n_step(int): number of successive time-step.

        Returns:
            list, list of dataframe.
        """
        return [df[:start]] + [df[start + (i * n_obj): start + (i * n_obj) + n_obj] for i in range(0, n_step)]

def evolutionary_split_label(df: pd.DataFrame, n_step: int, desc: bool, start_size: int = 1, min_size: int = 2):
    """Split the object of a specific label in n partition. The number of object of a specific category is
        stable, ascending, or descending over the successive partitions.

    Args:
        df(pd.DataFrame): objects of a specific label wrapped in a pandas DataFrame.
        n_step(int): number of successive time-step.
        desc(boolean): True for descending splitting. False for ascending splitting. None for stable splitting.
        start_size(int, default=1): this parameters allows to bias the sample. The first time a group appears, it will appear with at least start_size items
        min_size(int, default=2): this parameters allows to bias the sample. At each time step there will be at least min_size objects or 0
    Returns:
        list, list of dataframe.
    """

    n = df.shape[0]

    while True:

        pieces = list()
        illegal = False

        # stable
        if desc is None:
            m = n // n_step
            r = n % n_step
            pieces = [m] * n_step
            for i in range(r):
                pieces[i] += 1
        else:
            # evolution/extinction
            for idx in range(n_step - 1):
                # Number between 0 and n
                # minus the current total so we don't overshoot
                number = abs(int(np.random.uniform(0, n - sum(pieces)) - 1))
                pieces.append(number)
            pieces.append(n - sum(pieces))
            pieces = sorted(pieces, reverse=desc)

        if desc is not None:
            t = [i for i in sorted(pieces, reverse=desc) if i > 0]

            # 0 (not yet evolved) or at least start_threshold elements (new group)
            if t[0] < start_size and not ablation or t[0] > start_size and ablation:
                continue

            for i in t:
                # at least min_size elements
                if i < min_size:
                    illegal = True
                    break

        if not illegal:
            return [df[sum(pieces[:i]): sum(pieces[:i]) + p] for i, p in enumerate(pieces)]

def evolutionary_split(df: pd.DataFrame, n_step: int, start_size: int = 2, min_size_local: int = 2, min_size_global: int = 2) -> list:
    """
    Split a dataset in multiples partiton. The number of object of a specific category is
    stable, ascending, or descending over the successive partitions.

    Args:
        df(pd.DataFrame): dataset wrapped in a pandas DataFrame.
        n_step(int): number of successive time-step.
        start_size(int, default=1): this parameters allows to bias the sample. The first time a group appears, it will appear with at least start_size items
        min_size_local(int, default=2): this parameters allows to bias the sample. At each time step there will be at least min_size objects or 0 for each group
        min_size_global(int, default=2): this parameters allows to bias the sample. At each time step there will be at least min_size objects in total
    Returns:
        list, list of dataframe.
    """

    n_step += 1  # additional step (start)
    split = defaultdict(list)

    # flag: True/False, descending, ascending
    desc = None

    while True:

        # remove the generated sample
        for step in range(n_step):
            split[step] = list()

        for label in df.label.unique():
            rand = np.random.random()

            # extinction: descending
            if 0 <= rand < 1 / 3:
                desc = True

            # evolution: ascending
            elif 1 / 3 <= rand < 2 / 3:
                desc = False

            # stable: stable
            else:
                desc = None

            for step, df_pieces in enumerate(evolutionary_split_label(df=df[df.label == label], n_step=n_step, desc=desc,
                                                                   start_size=start_size, min_size=min_size_local)):
                split[step].append(df_pieces)

        # reset
        legal_split = True

        for step in split:

            # at least two group for samples
            if len([i for i in split[step] if i.shape[0] > 0]) < 2:
                legal_split = False
                break

            df_tmp = pd.concat(split[step])

            # the number of object is less than threshold min_size_global
            if df_tmp.shape[0] < min_size_global:
                legal_split = False
                break

        if legal_split:
            return [pd.concat(split[step]) for step in split]

def dump(split: list, path:str):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    dill.dump(split, open(path, mode='wb'))

def sample(path:str, df:pd.DataFrame, n_step:int, s:int, mg:int, ml:int=None):
    samples = list()

    for i in range(n_iteration):
        if ml is None:
            samples.append(incremental_split(df, n_step, s, mg))
        else:
            samples.append(evolutionary_split(df, n_step, s, ml, mg))

    dump(samples, path)

def sample_tmp(samples, n_iter, df:pd.DataFrame, n_step:int, s:int, mg:int, ml:int=None):

    for i in range(n_iter):
        if ml is None:
            samples.append(incremental_split(df, n_step, s, mg))
        else:
            samples.append(evolutionary_split(df, n_step, s, ml, mg))

def sample_multiprocessing(path, df, n_step, s, mg, ml):
    # something wrong???
    manager = Manager()
    samples = manager.list()
    workers = cpu_count()

    n_iter = 100
    n_iter_split = n_iter//workers
    tmp = [n_iter_split]*workers
    for i in range(n_iter % workers):
        tmp[i]+=1
    n_iter_split = tmp

    processes = []
    for i in range(cpu_count()):
        processes.append(Process(target=sample_tmp, args=(samples, n_iter_split[i], df, n_step, s, ml, mg)))
        processes[i].start()

    for i in range(cpu_count()):
        processes[i].join()

    samples = list(samples)
    dump(samples, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=100, help="number of independent experiment")
    parser.add_argument("--n_step", type=int, default=5, help="number of time steps")
    parser.add_argument("--start_size", type=int, help="the first time a group appears, it will appear with at least start_size items")
    parser.add_argument("--min_size_local", default=None, type=int, help="at each time step there will be at least min_size objects or 0 for each group")
    parser.add_argument("--min_size_global", type=int, help="at each time step there will be at least min_size objects in total")
    parser.add_argument("--dataset", type=str, help="the dataset on which the sample has to be performed")
    parser.add_argument("--output", type=str, help="the path to the file where the sample will be stored")
    parser.add_argument("--ablation", type=bool, default=False, help="ablation study start size")
    args = parser.parse_args()

    ablation = args.ablation

    n_iteration = args.n_iter
    n_step = args.n_step

    s = args.start_size
    ml = args.min_size_local
    mg = args.min_size_global

    out = args.output

    df = eval(f"{args.dataset}()")

    sample(out, df, n_step, s, mg, ml)
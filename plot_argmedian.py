import numpy as np
import dill
import argparse
from matplotlib import pyplot as plt
import numpy as np

def plot_metric():
    sample_ids = list()
    medians = [np.median(dump[dataset][metric][0]).round(3)]

    for step in range(1, 6):
        data = dump[dataset][metric][step]
        medians.append(np.median(data).round(3))
        argmedian = np.argsort(data)[len(data) // 2]
        sample_ids.append(argmedian)

    lines = list()
    for iter_ in sample_ids:
        line = [dump[dataset][metric][step][iter_] for step in range(0, n_step)]
        lines.append(line)

    color = ['green', 'blue', 'red', 'yellow', 'orange', 'purple']
    for i, line in enumerate(lines):
        plt.plot(list(range(0, n_step)), line, color=color[i], linestyle='--', marker='o', label=f'{i+1}-step')
    plt.xlabel('Time step')
    ylabel = 'NMI' if metric.startswith('normalized') else 'Pur'
    plt.ylabel(ylabel)
    #plt.legend()
    
    for x, y in zip(list(range(0, n_step)), medians):
        plt.text(x = x, y = y, s = f"{y}")
    plt.savefig(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", default=100, type=int, help="number of experiments")
    parser.add_argument("--n_step", default=5, type=int, help="number of time steps")
    parser.add_argument("--metric", default='normalized_mutual_info_score', type=str, help="metric to analyse")
    parser.add_argument("--output", type=str, help="the path to the file where the clustering results will be stored")
    parser.add_argument("--input", type=str, help="the path to containing the samples")
    parser.add_argument("--dataset", type=str, help="the path to containing the samples")
    args = parser.parse_args()

    n_iter = args.n_iter
    n_step = args.n_step + 1
    dataset = args.dataset
    metric = args.metric

    output = args.output

    dump = dill.load(open(args.input, mode='rb'))

    plot_metric()

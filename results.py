import dill
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default='normalized_mutual_info_score')
    parser.add_argument("--path")
    args = parser.parse_args()

    path = args.path
    dataset = path.split('/')[1]
    metric = args.metric

    for i in ['ap', 'iapna', 'app']:
        try:
            dump = dill.load(open(path.format(i), mode='rb'))
            print(i, [round(np.median(i),3) for i in dump[dataset][metric].values()])
        except:
            pass
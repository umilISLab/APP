# Scalable Incremental Affinity Propagation based on Cluster Consolidation and  Evolutionary Stratification

To reproduce our results:
- unzip <i>datasets.zip</i> and <i>samples.zip</i>: ```unzip -q filename.zip```
- comment the 3rd, 5th, and 15th lines in the bash files <i>iris.sh</i>, <i>car.sh</i>, <i>wine.sh</i>, <i>kddcup.sh</i>. By commenting you will perform exactly the same experiments as ours. Otherwise, you will randomly resample the datasets.
- run all the bash script: ```bash dataset.sh```
- run the python script <i>results.py</i> specifying the stats file and the metric to be analysed: e.g., ```python3 results.py --path stats/kddcup/s1904_mg200.dill --metric purity``` returns the purity for the KDD-CUP'99 dataset in the incremental setting.
- run the python script <i>plot_argmedian.py</i> specifying the stats file, the dataset, the metric to be analysed, and where you want to save the plot as image: 
e.g., ```python3 plot_argmedian.py --input stats/kddcup/app_s53_ml0_mg0_abl.dill --dataset kddcup --output ablation_study.png --metric purity``` returns the purity trend for the KDD-CUP'99 ablation setting.

Check our paper for further details on the experiments.


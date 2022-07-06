###########################
#incremental setup
python3 sampling.py --n_iter 100 --n_step 5 --start_size 100 --min_size_global 10 --dataset iris --output samples/iris/s100_mg10.dill
python3 evaluation.py --iap ap --dataset iris --input samples/iris/s100_mg10.dill --output stats/iris/ap_s100_mg10.dill & 
python3 evaluation.py --iap iapna --dataset iris --input samples/iris/s100_mg10.dill --output stats/iris/iapna_s100_mg10.dill &
python3 evaluation.py --iap app --dataset iris --input samples/iris/s100_mg10.dill --output stats/iris/app_s100_mg10.dill --aging_index 1 --pack0 centroid &
###########################
#evolutionary setup
python3 sampling.py --n_iter 100 --n_step 5 --start_size 5 --min_size_local 5 --min_size_global 10 --dataset iris --output samples/iris/s5_ml5_mg10.dill
python3 evaluation.py --iap ap --dataset iris --input samples/iris/s5_ml5_mg10.dill --output stats/iris/ap_s5_ml5_mg10.dill &
python3 evaluation.py --iap iapna --dataset iris --input samples/iris/s5_ml5_mg10.dill --output stats/iris/iapna_s5_ml5_mg10.dill &
python3 evaluation.py --iap app --dataset iris --input samples/iris/s5_ml5_mg10.dill --output stats/iris/app_s5_ml5_mg10.dill --aging_index 1 &
###########################
#ablation study
python3 sampling.py --n_iter 100 --n_step 5 --start_size 10 --min_size_local 0 --min_size_global 0 --dataset iris --output samples/iris/s10_ml0_mg0_abl.dill --ablation True
python3 evaluation.py --iap ap --dataset iris --input samples/iris/s10_ml0_mg0_abl.dill --output stats/iris/ap_s10_ml0_mg0_abl.dill &
python3 evaluation.py --iap iapna --dataset iris --input samples/iris/s10_ml0_mg0_abl.dill --output stats/iris/iapna_s10_ml0_mg0_abl.dill &
python3 evaluation.py --iap app --dataset iris --input samples/iris/s10_ml0_mg0_abl.dill --output stats/iris/app_s10_ml0_mg0_abl.dill --aging_index 1 &

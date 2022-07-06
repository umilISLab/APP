###########################
#incremental setup
python3 sampling.py --n_iter 100 --n_step 5 --start_size 128 --min_size_global 10 --dataset wine --output samples/wine/s128_mg10.dill
python3 evaluation.py --iap ap --dataset wine --input samples/wine/s128_mg10.dill --output stats/wine/ap_s128_mg10.dill &
python3 evaluation.py --iap iapna --dataset wine --input samples/wine/s128_mg10.dill --output stats/wine/iapna_s128_mg10.dill &
python3 evaluation.py --iap app --dataset wine --input samples/wine/s128_mg10.dill --output stats/wine/app_s128_mg10.dill --aging_index 1 &
###########################
#incremental setup
python3 sampling.py --n_iter 100 --n_step 5 --start_size 6 --min_size_local 6 --min_size_global 12 --dataset wine --output samples/wine/s6_ml6_mg12.dill
python3 evaluation.py --iap ap --dataset wine --input samples/wine/s6_ml6_mg12.dill --output stats/wine/ap_s6_ml6_mg12.dill &
python3 evaluation.py --iap iapna --dataset wine --input samples/wine/s6_ml6_mg12.dill --output stats/wine/iapna_s6_ml6_mg12.dill &
python3 evaluation.py --iap app --dataset wine --input samples/wine/s6_ml6_mg12.dill --output stats/wine/app_s6_ml6_mg12.dill --aging_index 1 &
###########################
#ablation study
python3 sampling.py --n_iter 100 --n_step 5 --start_size 15 --min_size_local 0 --min_size_global 0 --dataset wine --output samples/wine/s15_ml0_mg0_abl.dill --ablation True
python3 evaluation.py --iap ap --dataset wine --input samples/wine/s15_ml0_mg0_abl.dill --output stats/wine/ap_s15_ml0_mg0_abl.dill &
python3 evaluation.py --iap iapna --dataset wine --input samples/wine/s15_ml0_mg0_abl.dill --output stats/wine/iapna_s15_ml0_mg0_abl.dill &
python3 evaluation.py --iap app --dataset wine --input samples/wine/s15_ml0_mg0_abl.dill --output stats/wine/app_s15_ml0_mg0_abl.dill --aging_index 1
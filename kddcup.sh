###########################
#incremental setup
python3 sampling.py --n_iter 100 --n_step 5 --start_size 1904 --min_size_global 200 --dataset kddcup --output samples/kddcup/s1904_mg200.dill
python3 evaluation.py --iap ap --dataset kddcup --input samples/kddcup/s1904_mg200.dill --output stats/kddcup/ap_s1904_mg200.dill &
python3 evaluation.py --iap iapna --dataset kddcup --input samples/kddcup/s1904_mg200.dill --output stats/kddcup/iapna_s1904_mg200.dill &
python3 evaluation.py --iap app --dataset kddcup --input samples/kddcup/s1904_mg200.dill --output stats/kddcup/app_s1904_mg200.dill --aging_index 1 &
###########################
#incremental setup
python3 sampling.py --n_iter 100 --n_step 5 --start_size 26 --min_size_local 26 --min_size_global 52 --dataset kddcup --output samples/kddcup/s26_ml26_mg52.dill
python3 evaluation.py --iap ap --dataset kddcup --input samples/kddcup/s26_ml26_mg52.dill --output stats/kddcup/ap_s26_ml26_mg52.dill &
python3 evaluation.py --iap iapna --dataset kddcup --input samples/kddcup/s26_ml26_mg52.dill --output stats/kddcup/iapna_s26_ml26_mg52.dill &
python3 evaluation.py --iap app --dataset kddcup --input samples/kddcup/s26_ml26_mg52.dill --output stats/kddcup/app_s26_ml26_mg52.dill --aging_index 1 &
###########################
#ablation study
python3 sampling.py --n_iter 100 --n_step 5 --start_size 53 --min_size_local 0 --min_size_global 0 --dataset kddcup --output samples/kddcup/s53_ml0_mg0_abl.dill --ablation True
python3 evaluation.py --iap ap --dataset kddcup --input samples/kddcup/s53_ml0_mg0_abl.dill --output stats/kddcup/ap_s53_ml0_mg0_abl.dill &
python3 evaluation.py --iap iapna --dataset kddcup --input samples/kddcup/s53_ml0_mg0_abl.dill --output stats/kddcup/iapna_s53_ml0_mg0_abl.dill &
python3 evaluation.py --iap app --dataset kddcup --input samples/kddcup/s53_ml0_mg0_abl.dill --output stats/kddcup/app_s53_ml0_mg0_abl.dill --aging_index 1
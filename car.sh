###########################
#incremental setup
python3 sampling.py --n_iter 100 --n_step 5 --start_size 210 --min_size_global 10 --dataset car --output samples/car/s210_mg10.dill 
python3 evaluation.py --iap ap --dataset car --input samples/car/s210_mg10.dill --output stats/car/ap_s210_mg10.dill &
python3 evaluation.py --iap iapna --dataset car --input samples/car/s210_mg10.dill --output stats/car/iapna_s210_mg10.dill &
python3 evaluation.py --iap app --dataset car --input samples/car/s210_mg10.dill --output stats/car/app_s210_mg10.dill --aging_index 1 &
###########################
#evolutionary setup
python3 sampling.py --n_iter 100 --n_step 5 --start_size 7 --min_size_local 7 --min_size_global 14 --dataset car --output samples/car/s7_ml7_mg14.dill
python3 evaluation.py --iap ap --dataset car --input samples/car/s7_ml7_mg14.dill --output stats/car/ap_s7_ml7_mg14.dill &
python3 evaluation.py --iap iapna --dataset car --input samples/car/s7_ml7_mg14.dill --output stats/car/iapna_s7_ml7_mg14.dill &
python3 evaluation.py --iap app --dataset car --input samples/car/s7_ml7_mg14.dill --output stats/car/app_s7_ml7_mg14.dill --aging_index 1 &
###########################
#ablation study
python3 sampling.py --n_iter 100 --n_step 5 --start_size 13 --min_size_local 0 --min_size_global 0 --dataset car --output samples/car/s13_ml0_mg0_abl.dill --ablation True
python3 evaluation.py --iap ap --dataset car --input samples/car/s13_ml0_mg0_abl.dill --output stats/car/ap_s13_ml0_mg0_abl.dill &
python3 evaluation.py --iap iapna --dataset car --input samples/car/s13_ml0_mg0_abl.dill --output stats/car/iapna_s13_ml0_mg0_abl.dill &
python3 evaluation.py --iap app --dataset car --input samples/car/s13_ml0_mg0_abl.dill --output stats/car/app_s13_ml0_mg0_abl.dill --aging_index 1 &
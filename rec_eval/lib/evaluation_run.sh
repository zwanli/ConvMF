#!/usr/bin/env bash
rootpath=/home/alzoghba/citeulike_data/terms_keywords_based
for exp in CTR_Rocchio_user_not_normalized_Cosine_k
do
    for folder in 2k_10_P3_reduced 2k_10_P5_reduced 2k_10_P10_reduced 2k_50_P3_reduced 2k_50_P5_reduced 2k_100_P3_reduced 2k_100_P5_reduced 2k_20_P3_reduced 2k_20_P5_reduced 2k_20_P10_reduced 2k_30_P3_reduced 2k_30_P5_reduced 2k_30_P10_reduced 2k_40_P3_reduced 2k_40_P5_reduced 
    do
        for k in 200
        do  
            screen -S evaluator_${exp}_${k}_${folder} -dm bash -c " source /home/alzoghba/HyPRec/py3.5_hyprec/bin/activate;  \
            python3 /home/alzoghba/IFUP2018/Recommender_evaluator/lib/evaluator.py -u ${rootpath}/${folder}/users.dat \
            -p ${rootpath}/${folder}/in-matrix-item_folds \
            -x ${rootpath}/${folder}/in-matrix-item_folds/${exp}_${k} \
            -f 5"
        done
    done
done

# python3.5 evaluator.py --score --folds 5 --split_directory /vol2/wanliz/data/Extended_ctr/citeulike_a_extended/in-matrix-item_folds/ -x /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/inmatrix/0.01000-1.10000-300-cnn_cae -nu 5551
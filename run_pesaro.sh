#!/usr/bin/env bash

#Pesaro
## preprocess documents and ratings
python3 process_citeulike.py --data_dir /vol2/wanliz/data/Extended_ctr/convmf/ --dataset citeulike-a


# Pre-process

python run.py --do_preprocess true --raw_rating_data_path /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/ratings.txt \
--raw_item_document_data_path /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/papers.txt \
-d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/in-matrix_convmf/ \
-a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ \
--splits_dir /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/in-matrix-item_folds/ -f 5
# Train CNN
python run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix/ -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/ -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/w2v_200.txt --content_mode cnn -u 0.0001 -v 1000 -f 5

# Train CNN_CAE
python run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix/ \
-a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ \
-o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/ \
-e 200 -k 200 \
-p /vol2/wanliz/data/cbow_w2v/w2v_200.txt \
--content_mode cnn_cae -u 0.01 -v 10 --att_dim 10 -f 5

python run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/in-matrix_convmf/ -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/22-9_in-matrix-200_no-val_0.01-100_w-alpha_cnn-100_cae-50_concat  -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/normalized_word_embeddings.txt --content_mode cnn_cae -u 0.01 -v 100 --num_kernel_per_ws 100  -f 5 --max_iter 200 --att_dim 50 --give_item_weight true
#Grid search
python run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix/ -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/grid_search -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/w2v_200.txt --grid_search true --splits_dir /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/in-matrix-item_folds/

# train raw_att_cnn
python run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/in-matrix_convmf/ -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/10-9_in-matrix-200_no-val_0.01-100-w-cnn-100_att_raw -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/normalized_word_embeddings.txt --content_mode raw_att_cnn -u 0.01 -v 100 --num_kernel_per_ws 100  --give_item_weight True  --join_mode transfer -f 5 --max_iter 200 -lr 0.01
## Local
# preprocess
python run.py --do_preprocess true --raw_rating_data_path /home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/ratings.txt --raw_item_document_data_path /home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/papers.txt -d /home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix -a /home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ --splits_dir /home/zaher/data/Extended_ctr/citeulike_a_extended/in-matrix-item_folds/ -f 10
--do_preprocess
true
--raw_rating_data_path
/home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/ratings.txt
--raw_item_document_data_path
/home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/papers.txt
-d
/home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix
-a
/home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/
--splits_dir
/home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/raw/in-matrix-item_folds/
-f
5
# train local

# CAE
-d
/home/zaher/data/Extended_ctr/convmf/dummy/inmatrix/
-a
/home/zaher/data/Extended_ctr/convmf/dummy/preprocessed/
-o
/home/zaher/data/Extended_ctr/convmf/dummy/results/
-k
200
--content_mode
cae
-u
0.01
-v
10
--att_dim
10

#MF

-d
/home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix/
-a
/home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/
-o
/home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/results/
-k
200
--content_mode
mf
-u
0.01
-v
0.01
-f
10

-d /home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix -a /home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/results/25-7_0.01-1000-c-mf -e 200 -k 200 --content_mode mf -u 0.01 -v 1000 -f 5

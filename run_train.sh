#!/usr/bin/env bash

#Pesaro
# Pre-process

python run.py --do_preprocess true --raw_rating_data_path /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/ratings.txt --raw_item_document_data_path /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/papers.txt -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ --splits_dir /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/in-matrix-item_folds/

# Train CNN
python run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix/ -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/ -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/w2v_200.txt --content_mode cnn -u 0.0001 -v 1000

# Train CNN_CAE
python run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix/ -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/ -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/w2v_200.txt --content_mode cnn_cae -u 0.01 -v 10 --att_dim 10

#Grid search
python run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix/ -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/grid_search -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/w2v_200.txt --grid_search true --splits_dir /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/in-matrix-item_folds/


## Local
# preprocess
python run.py --do_preprocess true --raw_rating_data_path /home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/ratings.txt --raw_item_document_data_path /home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/papers.txt -d /home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix -a /home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ --splits_dir /home/zaher/data/Extended_ctr/citeulike_a_extended/in-matrix-item_folds/ -f 10
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


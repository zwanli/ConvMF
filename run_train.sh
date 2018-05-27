#!/usr/bin/env bash

# Pre-process

python run.py --do_preprocess true --raw_rating_data_path /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/ratings.txt --raw_item_document_data_path /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/papers.txt -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ --splits_dir /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/in-matrix-item_folds/

# Train CNN
python run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix/fold-1 -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/ -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/w2v_200.txt --content_mode cnn -u 0.0001 -v 1000

# Train CNN_CAE
python run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix/fold-1 -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/ -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/w2v_200.txt --content_mode cnn_cae -u 0.01 -v 10 --att_dim 10

#Grid search
python run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix/ -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/grid_search -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/w2v_200.txt --grid_search true --splits_dir /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/in-matrix-item_folds/
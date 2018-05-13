python ./run.py \
-d ../data/preprocessed/movielens_10m/cf/0.2_1/ \
-a ../data/preprocessed/movielens_10m/ \
-o ./test/movielens_10m/result/1_100_200 \
-e 200 \
-p ../data/preprocessed/glove/glove.6B.200d.txt \
-u 10 \
-v 100 \
-g True



%
-d
/home/wanli/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/inmatrix
-a
/home/wanli/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/
-o
/home/wanli/data/Extended_ctr/convmf/citeulike_a_extended/results/inmatrix/
-e
200
-k
200
-p
/home/wanli/data/cbow_w2v/w2v_200.txt
-u
10
-v
100
-g
True
%

% GRID SEARCH
-d
/home/wanli/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/inmatrix/
-a
/home/wanli/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/
-o
/home/wanli/data/Extended_ctr/convmf/citeulike_a_extended/results/
-e
200
-k
200
-p
/home/wanli/data/cbow_w2v/w2v_200.txt
-u
10
-v
100
-g
True
%

## preprocessing step for fold 1
#run.py
--do_preprocess
true
--raw_rating_data_path
/home/zaher/data/Extended_ctr/convmf/dummy/ratings.txt
--raw_item_document_data_path
/home/zaher/data/Extended_ctr/convmf/dummy/papers.txt
-d
/home/zaher/data/Extended_ctr/convmf/dummy/preprocessed/inmatrix
-a
/home/zaher/data/Extended_ctr/convmf/dummy/preprocessed/
--splits_dir
/home/zaher/data/Extended_ctr/convmf/dummy/in-matrix-item_folds/fold-1
-o
/home/zaher/data/Extended_ctr/convmf/dummy/results/inmatrix/


## train
run.py
-d
/home/zaher/data/Extended_ctr/convmf/dummy/preprocessed/inmatrix/fold-1
-a
/home/zaher/data/Extended_ctr/convmf/dummy/preprocessed/
-o
/home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/results/
-e
200
-k
200
-p
/home/zaher/data/cbow_w2v/w2v_200.txt
-u
10
-v
100


# /usr/bin/python2.7 run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/inmatrix/fold-5 -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/inmatrix/fold-5/ -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/w2v_200.txt -u 0.01 -v 10 -g True --content_mode cnn_cae
run.py --splits_dir /home/wanli/data/Extended_ctr/citeulike_a_extended/in-matrix-item_folds -d /home/wanli/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/inmatrix/ -a /home/wanli/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /home/wanli/data/Extended_ctr/convmf/citeulike_a_extended/results/inmarix -e 200 -k 200 -p /home/wanli/data/cbow_w2v/w2v_200.txt -u 10 -v 100 -g True --grid_search

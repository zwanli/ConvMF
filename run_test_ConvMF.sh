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

# /usr/bin/python2.7 run.py -d /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/inmatrix/fold-5 -a /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/ -o /vol2/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/inmatrix/fold-5/ -e 200 -k 200 -p /vol2/wanliz/data/cbow_w2v/w2v_200.txt -u 0.01 -v 10 -g True --content_mode cnn_cae

python ./run.py \
-d ./test/ml-10m/0.2/ \
-a ./test/ml-10m/ \
-c True \
-r ./data/movielens/ml-10m_ratings.dat \
-i ./data/movielens/Plot.idmap \
-m 1

%

-f
5
--do_preprocess
true
--raw_rating_data_path
/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/ratings.txt
--raw_item_document_data_path
/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/papers.txt
-d
/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix
-a
/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed_2/
--splits_dir
/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/raw/in-matrix-item_folds/


##
-d
/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/inmatrix/
-a
/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/preprocessed/
-o
/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/delete_me
-e
200
-k
200
-p
/home/wanliz/data/cbow_w2v/normalized_word_embeddings.txt
--content_mode
cnn
-u
0.01
-v
100
--num_kernel_per_ws
100
-f
5
--max_iter
200
-lr
0.01
import os
import csv
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import argparse

def convert_ratings(in_path, out_path):
    print('Reading ratings ...')
    with open(in_path, 'r') as f:
        with open(out_path,'w') as outfile:
            u_id =0
            for line in f.readlines():
                items_idx = line.split()[1:]
                items_idx = [int(x) for x in items_idx]
                user_id = u_id  # 0 base index
                rating = 1
                for i in items_idx:
                    outfile.write('{}::{}::{}\n'.format(user_id,i,rating))
                u_id +=1
    print('File {} is generated. Each raw is user_id::item_id::rating\n'.format(out_path))

def convert_abstracts(in_path, out_path,dataset='citeulike-a'):

    delimiter = ','
    if dataset == 'citeulike-t':
        delimiter = '\t'
    first_line = True
    # read raw data
    print('Reading documents ...')
    with open(in_path, "r", encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter=delimiter)
        with open(out_path,'w') as outfile:
            for line in reader:
                if first_line:
                    labels = line
                    row_length = len(line)
                    first_line = False
                    continue
                doc_id = line[0]
                if dataset == 'citeulike-t':
                    paper = line[1]
                elif row_length > 2:
                    paper = line[1]+' '+line[4]
                sentences = sent_tokenize(paper)
                document = []
                sentences = [document.extend(word_tokenize(x)) for x in sentences]

                outfile.write('{}::{}|\n'.format(int(doc_id) - 1, ' '.join(document)))
    print('File {} is generated. Each raw is item_id::abstract|'.format(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/wanli/data/Extended_ctr/convmf',
                        help='data directory containing input.txt')
    parser.add_argument("--dataset", "-d", type=str, default='citeulike-a',
                        help="Which dataset to use", choices=['dummy', 'citeulike-a', 'citeulike-t'])
    args = parser.parse_args()

    if args.dataset == 'citeulike-a':
        dataset_folder = os.path.join(args.data_dir ,'citeulike_a_extended')
    elif args.dataset == 'citeulike-t':
        dataset_folder = os.path.join(args.data_dir ,'citeulike_t_extended')
    elif args.dataset == 'dummy':
        dataset_folder = os.path.join(args.data_dir , 'dummy')
    else:
        print("Warning: Given dataset not known, setting to dummy")
        dataset_folder = os.path.join(args.data_dir ,'citeulike_a_extended')

    ratings_path = os.path.join(dataset_folder,'users.dat')
    ratings_outfile =  os.path.join(dataset_folder,'ratings.txt')
    convert_ratings(ratings_path, ratings_outfile)

    abstracts_path = os.path.join(dataset_folder, 'raw-data.csv')
    abstracts_outfile = os.path.join(dataset_folder, 'papers.txt')
    convert_abstracts(abstracts_path, abstracts_outfile)

if __name__ == '__main__':
     main()

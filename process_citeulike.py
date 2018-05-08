import os
import csv
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import argparse
import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing

def convert_ratings(in_path, out_path):
    print('Reading ratings ...')

    items_id = set()
    max_item_id = 0
    with open(in_path, 'r') as f:
        with open(out_path,'w') as outfile:
            u_id =0
            for line in f.readlines():
                item_idx = line.split()[1:]
                item_idx = [int(x) for x in item_idx]
                user_id = u_id  # 0 base index
                rating = 1
                for i in item_idx:
                    if i > max_item_id:
                        max_item_id = i
                    items_id.add(i)
                    outfile.write('{}::{}::{}\n'.format(user_id,i,rating))
                u_id +=1
    print('Number of distinctive items {}, Max item id {}'.format(len(items_id),max_item_id))
    print('File {} is generated. Each raw is user_id::item_id::rating\n'.format(out_path))
    return items_id

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

                #todo: check the 0 base indexing
                outfile.write('{}::{}|\n'.format(int(doc_id), ' '.join(document)))
    print('File {} is generated. Each raw is item_id::abstract|'.format(out_path))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def proccess_paper_info(path, outfile,items_id, paper_count):
    null_token = 'NaN'
    now = datetime.datetime.now()

    clean_file_path = path +'.cleaned'
    if os.path.exists(clean_file_path):
        os.remove(clean_file_path)
    with open(path, "r", encoding='utf-8', errors='ignore') as infile:
        reader = csv.reader(infile, delimiter=',')
        i = 0
        first_line = True

        with open(clean_file_path, 'w', newline='') as f2:
            writer = csv.writer(f2, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            for line in reader:
                if first_line:
                    row_length = len(line)
                    first_line = False
                    writer.writerow(line)
                    continue
                if len(line) > row_length:
                    line[row_length] = ' '.join(line[row_length-1:]).replace('\t', ' ')
                    line = line[:row_length]
                paper_id = int(line[0]) - 1
                if paper_id != i and int(paper_id) != paper_count:
                    for _ in range(int(paper_id) - i):
                        empty_row = [str(i)]
                        empty_row.extend([null_token] * (row_length-1))
                        # empty_row = '\t'.join(empty_row)
                        writer.writerow(empty_row)
                        i += 1
                for j, _ in enumerate(line):
                    if line[j] == '\\N':
                        line[j] = null_token
                writer.writerow(line)
                i += 1
            if i < paper_count:
                # todo: check the paper count
                for _ in range(int(paper_count - 1) - i):
                    empty_row = [str(i)]
                    empty_row.extend([null_token] * (row_length - 1))
                    # empty_row = '\t'.join(empty_row)
                    writer.writerow(empty_row)
                    i += 1

    # Month converter
    months = ['apr','aug', 'dec' ,'feb', 'jan' ,'jul' ,'jun' ,'mar' ,'may', 'nov', 'oct', 'sep']
    month_convert_func = lambda x: x if x in months else null_token

    def number_convert_func (x):
        if x == '-1':
            return null_token
        if is_number(x):
            if x == 'NaN':
                return null_token
            x = float(x)
            x = int(x)
            return x
        return null_token

    def year_convert_func (x):
        if is_number(x):
            if x == 'NaN':
                return null_token
            x = float(x)
            x = int(x)
            if x == -1:
                return null_token
            if x < 1000:
                return null_token
            return now.year - x
        else:
            return null_token

    labels = ['doc_id', 'citeulike_id', 'type', 'pages', 'year']
    labels_dtype = {'doc_id': np.int32, 'citeulike_id': np.int32, 'type': str, 'pages': np.int32}
    convert_func= {'pages': number_convert_func, 'doc_id': number_convert_func, 'year': year_convert_func,
                   'citeulike_id': number_convert_func}
    # labels = ['doc_id', 'citeulike_id', 'type', 'journal', 'booktitle', 'series', 'pages', 'year', 'month', 'address']
    # labels_dtype = {'doc_id': np.int32, 'citeulike_id': np.int32, 'type': str, 'journal': str, 'booktitle': str,
    #                 'series': str,
    #                 'pages': np.int32, 'month': str, 'address': str}
    # convert_func = {'month': month_convert_func, 'pages': number_convert_func, 'doc_id': number_convert_func,
    #                 'citeulike_id': number_convert_func}

    df = pd.read_table(clean_file_path, delimiter='\t', index_col = 'doc_id', usecols=labels,dtype=labels_dtype,
                         na_values='\\N',na_filter=False,
                         converters=convert_func)

    # special case
    # df.year[df.year == -20058083] = 2006

    # Filter values with frequency less than min_freq
    def filter(df, tofilter_list, min_freq):
        for col in tofilter_list:
            to_keep = df[col].value_counts().reset_index(name="count").query("count > %d" % min_freq)["index"]
            to_keep = to_keep.values.tolist()
            df[col] = [x if x in to_keep else 'NaN' for x in df[col]]
        return df

    tofilter_list = []
    df = filter(df, tofilter_list, 2)

    # Convert catigorical feature into one-hot encoding
    def dummmy_df(df, todummy_list):
        for x in todummy_list:
            dummies = pd.get_dummies(df[x], prefix=x, dummy_na=True)
            df = df.drop(x, 1)
            df = pd.concat([df, dummies], axis=1)
        return df

    todummy_list = ['type']
    df = dummmy_df(df, todummy_list)

    # Remove the last generated column type_nan, it's always zero
    df = df.drop('type_nan',1)
    # # Remove 'citeulike_id' column, it will be added after normalization
    # df = df.drop('citeulike_id',1)

    # Replace NaN values with mean before normalization
    # TODO: Make sure that the cells in the page column are positive
    # Get the data as np array
    values = df.values
    # Impute the data using the mean value
    imputer = preprocessing.Imputer(missing_values=null_token, strategy='mean')
    transformed_values = imputer.fit_transform(values)

    # Normalize the year and page columns
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(transformed_values)

    # convert the scaled data into pandas dataframe
    df_normalized = pd.DataFrame(x_scaled)

    # rename columns
    df_normalized.columns = df.columns

    # add 'citeulike_id' column
    df_normalized = df_normalized.assign(citeulike_id=df.citeulike_id.values)

    #remove items that havn't appeared in the ratings matrix
    df_normalized = df_normalized.iloc[list(items_id)]

    # outfile = os.path.join(os.path.dirname(path),'paper_info_processed.csv')
    df_normalized.to_csv(outfile, sep='\t', na_rep='' )


    print('Processed features saved to %s' % outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/zaher/data/Extended_ctr/convmf',
                        help='data directory containing input.txt')
    parser.add_argument("--dataset", "-d", type=str, default='citeulike-a',
                        help="Which dataset to use", choices=['dummy', 'citeulike-a', 'citeulike-t'])
    args = parser.parse_args()

    if args.dataset == 'citeulike-a':
        dataset_folder = os.path.join(args.data_dir ,'citeulike_a_extended')
        paper_count = 16980
    elif args.dataset == 'citeulike-t':
        dataset_folder = os.path.join(args.data_dir ,'citeulike_t_extended')
        paper_count = 25976
    elif args.dataset == 'dummy':
        dataset_folder = os.path.join(args.data_dir , 'dummy')
        paper_count = 1929
    else:
        print("Warning: Given dataset not known, setting to citeulike_a_extended")
        dataset_folder = os.path.join(args.data_dir,'citeulike_a_extended')

    ratings_path = os.path.join(dataset_folder,'users.dat')
    ratings_outfile = os.path.join(dataset_folder,'ratings.txt')
    items_id = convert_ratings(ratings_path, ratings_outfile)

    abstracts_path = os.path.join(dataset_folder, 'raw-data.csv')
    abstracts_outfile = os.path.join(dataset_folder, 'papers.txt')
    convert_abstracts(abstracts_path, abstracts_outfile)

    paper_info_path= os.path.join(dataset_folder, 'paper_info.csv')
    paper_info_outfile = os.path.join(dataset_folder, 'preprocessed/paper_attributes.tsv')
    proccess_paper_info(paper_info_path, paper_info_outfile, items_id =items_id, paper_count=paper_count)

if __name__ == '__main__':
     main()

'''
Parts of the code belong to Fabien Daniel (July 2017)
https://www.kaggle.com/fabiendaniel/film-recommendation-engine

'''
import os
# import csv
# from nltk.tokenize import sent_tokenize
# from nltk.tokenize import word_tokenize
import argparse
# import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


def load_paper_info(path,fix_years=False):

    df = pd.read_csv(path,sep=',')
    if fix_years:
        df['year']=pd.to_datetime(df['year'].astype(int),format='%Y',errors='coerce').dt.year

    return df

def print_info(df_initial):
    tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0: 'null values'}))
    tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum() / df_initial.shape[0] * 100).T.
                               rename(index={0: 'null values (%)'}))
    tab_info = tab_info.append(pd.DataFrame(df_initial.nunique()).T.rename(index={0: 'unique values'}))
    print(tab_info)

def plot_frequeny(df, col_label,ax=None):

    plot = False
    if not ax:
        fig, ax = plt.subplots()
        plot = True

    df = df[col_label].value_counts().reset_index()
    df = df.sort_values(by='index')
    ax.set_title(col_label)
    # ax.set_xticklabels(df['index'], rotation=45)
    # ax.set_xticks(df[col_label])
    # ax.set_xticks(df['index'])
    df.plot(ax=ax,kind='bar',x='index',y=col_label,fontsize=5,figsize=[20,20])

    if plot:
        plt.show()

def plot_frequencies(df,dataset_folder):
    # fig, ax = plt.subplots()
    # df[col_label].value_counts().plot(ax=ax,x_compat=True)

    nrows = df.shape[1]
    fig, axes = plt.subplots(nrows=nrows)

    for j in range(df.shape[1]):
        # axes[j].set_title(df.columns[j])
        # df[df.columns[j]].value_counts().plot(ax=axes[j], x_compat=True)
        plot_frequeny(df,df.columns[j],axes[j])
    # plt.show()
    plt.savefig(os.path.join(dataset_folder,'paper_info_frequencies'),format='png')
#

def count_missing_values(df):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['filling_factor'] = (df.shape[0]
                                    - missing_df['missing_count']) / df.shape[0] * 100
    missing_df =missing_df.sort_values('filling_factor',ascending=False).reset_index(drop=True)
    print(missing_df)

def correlation(df):
    f, ax = plt.subplots(figsize=(12, 9))
    # _____________________________
    # calculations of correlations
    # ignore first 2 columns (index, doc_id)
    corrmat =df.iloc[:,2:].dropna(how='any').corr()
    # ________________________________________
    k = 17  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'pages_norm')['pages_norm'].index
    cm = np.corrcoef(df[cols].dropna(how='any').values.T)
    sns.set(font_scale=1.25)

    hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size': 10}, linewidth=0.1, cmap='coolwarm',
                 yticklabels=cols.values, xticklabels=cols.values)
    f.text(0.5, 0.93, "Correlation coefficients", ha='center', fontsize=18, family='fantasy')
    plt.show()

def convert_year_to_categorical(df):
    df['year_cat'] = df.year.astype(str)
    df.loc[df.year < 1990, 'year_cat'] = 'befor_90'
    df.loc[(df.year >= 1990) & (df.year < 1995), 'year_cat'] = '90-94'
    df.loc[(df.year >= 1995) & (df.year < 2000), 'year_cat'] = '95-1999'
    df.loc[(df.year >= 2000) & (df.year < 2005), 'year_cat'] = '2000-2004'
    df.loc[df.year >= 2005, 'year_cat'] = '2005-2016'
    df.loc[df.year == 0, 'year_cat'] = 'null'
    df.year_cat = pd.Categorical(df.year_cat, ["null",'befor_90','90-94','95-1999','2000-2004', '2005-2016'])
    return df.year_cat

def min_max_scale(df):
    '''

    :param series: series that we want to scale
    :return:
    Scaled series
    '''
    scaler = preprocessing.MinMaxScaler()
    df.pages *= -1
    df['pages_norm'] = scaler.fit_transform(df[['pages']])
    return df.pages_norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/wanliz/data/Extended_ctr/convmf',
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

    paper_info_path= os.path.join(dataset_folder,'raw', 'papers_info_corrected_pages.csv')
    paper_info_outfile = os.path.join(dataset_folder, 'preprocessed/paper_attributes.tsv')
    df_initial = load_paper_info(paper_info_path)
    print('Shape:', df_initial.shape)

    print_info(df_initial)

    # plot_frequeny(df_initial,'pages')
    # plot_frequencies(df_initial,dataset_folder)

    count_missing_values(df_initial)

    fig, ax = plt.subplots(dpi=250)

    ''' Add zeros for the the documents that has no attributes  '''
    # raw_data = '/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/raw/raw-data.csv'
    raw_data_path= os.path.join(dataset_folder,'raw', 'raw-data.csv')

    raw_data_df = pd.read_csv(raw_data_path, sep=',', encoding='iso-8859-1')
    # get doc_ids of completlty missing rows
    missing_ids = sorted(list(set(range(1, 16981)).difference(set(df_initial.doc_id.values))))
    # subtract 1 because index is zero based, but doc_id is 1 based
    missing_ids = [x - 1 for x in missing_ids]
    missing_raws_df = raw_data_df.iloc[missing_ids]
    print (missing_raws_df.columns)
    missing_raws_df = missing_raws_df[['doc_id', 'citeulike_id', 'title', 'raw_abstract']]
    print (missing_raws_df.columns)
    missing_raws_df = missing_raws_df.rename(index=str, columns={"raw_abstract": "abstract"})
    missing_raws_df.abstract.str.lower()
    print (missing_raws_df.columns)
    no_missing_rows_df = pd.concat([df_initial, missing_raws_df], ignore_index=True, sort=False)
    no_missing_rows_df = no_missing_rows_df.sort_values('doc_id')
    df = no_missing_rows_df.reset_index(drop=True)

    ''' Processing Year '''
    # replace NaN with zeros for 'year' and 'pages'
    values = {'year': 0}
    df = df.fillna(value=values)
    df.year = df.year.astype(int)

    ## convert 'year' form numerical into categorical
    df['year_cat'] = convert_year_to_categorical(df)
    ## plot years histogram
    df.year.value_counts().sort_index().plot(ax=ax, kind='bar')
    plt.xlabel('year', fontsize=20)
    plt.xticks(fontsize=5)
    df.year_cat.value_counts().sort_index().plot(ax=ax, kind='bar')
    plt.savefig(os.path.join(dataset_folder,'year_frequencies'),format='png')
    plt.show()


    # ''' Processing pages'''
    # df.pages.value_counts().sort_index().plot(ax=ax, kind='bar')
    # plt.xlabel('pages', fontsize=20)
    # plt.xticks(fontsize=4)
    # plt.savefig(os.path.join(dataset_folder,'pages_frequencies'),format='png')
    # plt.show()
    # fill NaN with mean value in order for scale to work, using mean values won't effect the scaling
    values = {'pages': df.pages.mean()}
    df = df.fillna(value=values)
    df['pages_norm'] = min_max_scale(df[['pages']])
    # replace nan values with zeros
    df.loc[df.pages == df.pages.mean(), 'pages_norm'] = 0
    correlation(df)


    ''' Select columns to bel used in training the model later'''
    selected_attribues = ['doc_id', 'citeulike_id', 'type', 'pages_norm', 'year_cat']

    selected_df = df[selected_attribues]

    '''Convert catigorical feature into one-hot encoding '''
    def dummmy_df(df, todummy_list):
        for x in todummy_list:
            dummies = pd.get_dummies(df[x], prefix=x, dummy_na=True)
            df = df.drop(x, 1)
            df = pd.concat([df, dummies], axis=1)
        return df



    todummy_list = ['type','year_cat']
    df = dummmy_df(selected_df, todummy_list)
    df.to_csv(paper_info_outfile, sep='\t', na_rep='')
    # proccess_paper_info(paper_info_path, paper_info_outfile, items_id =items_id, paper_count=paper_count)
    correlation(df)

    # get duplicates
    # a = pd.concat(g for _, g in df_initial.groupby('abstract') if len(g) > 1)
if __name__ == '__main__':
     main()

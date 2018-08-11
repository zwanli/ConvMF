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
# from sklearn import preprocessing
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
    corrmat =df.iloc[:,3:].dropna(how='any').corr()
    # ________________________________________
    k = 17  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'pages')['pages'].index
    cm = np.corrcoef(df[cols].dropna(how='any').values.T)
    sns.set(font_scale=1.25)

    hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size': 10}, linewidth=0.1, cmap='coolwarm',
                 yticklabels=cols.values, xticklabels=cols.values)
    f.text(0.5, 0.93, "Correlation coefficients", ha='center', fontsize=18, family='fantasy')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/wanliz/data/Extended_ctr/convmf',
                        help='data directory containing input.txt')
    parser.add_argument("--dataset", "-d", type=str, default='citeulike-a',
                        help="Which dataset to use", choices=['dummy', 'citeulike-a', 'citeulike-t'])
    args = parser.parse_args()

    if args.dataset == 'citeulike-a':
        dataset_folder = os.path.join(args.data_dir ,'citeulike_a_extended','raw')
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

    paper_info_path= os.path.join(dataset_folder, 'papers_info_corrected_pages.csv')
    paper_info_outfile = os.path.join(dataset_folder, 'preprocessed/paper_attributes.tsv')
    df_initial = load_paper_info(paper_info_path)
    print('Shape:', df_initial.shape)

    print_info(df_initial)

    # plot_frequeny(df_initial,'pages')
    # plot_frequencies(df_initial,dataset_folder)

    count_missing_values(df_initial)
    correlation(df_initial)

    # proccess_paper_info(paper_info_path, paper_info_outfile, items_id =items_id, paper_count=paper_count)

if __name__ == '__main__':
     main()

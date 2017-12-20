"""
Author: Anas Alzogbi
Description: this module provides the functionality of:
 - Generate time-based training/test split for a set of timestamped ratings
 - Generate statistics about the given timestamped ratings and the generated split
Date: December 3rd, 2017
"""
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from util.files_utils import read_ratings_into_array, read_mappings, write_ratings, write_list_to_file, read_docs_vocabs_file, load_list_from_file, save_mult_list, save_list
from util.stats import Stats
from util.split_utils import format_grid

class TimeBasedSplitter(object):

    def __init__(self, base_dir='citeulike-crawled', split_duration=6, out_of_matrix = False):
        self.base_dir = base_dir

        # Read ratings
        self.ratings = read_ratings_into_array(os.path.join(self.base_dir, "users.dat"))

        # load the mult file:
        self.docs_vocabs = read_docs_vocabs_file(os.path.join(self.base_dir, "mult.dat"))

        # Load the terms file:
        self.terms = load_list_from_file(os.path.join(self.base_dir, "terms.dat"))

        # Read id mappings:
        # Read userid- citeulikeid mapping
        self.users_dict = read_mappings(os.path.join(self.base_dir, "citeulikeUserHash_userId_map.dat"))
        # Read citeulike_paperid mapping:
        self.papers_dict = read_mappings(os.path.join(self.base_dir, "citeulikeId_docId_map.dat"))
        self.split_duration = split_duration
        self.stat_list = []
        self.out_of_matrix = out_of_matrix


    def plot_split_lines(self, tr_rs, tr_us, tr_ps, ts_rs, ts_us, ts_ps, rat, dates):
        x = np.arange(len(dates))
        fig, axes = plt.subplots(nrows=4, figsize=(15, 15))
        ax = axes[0]
        ax.plot(x, tr_rs, '--', linewidth=2, label='Training ratings')
        ax.plot(x, ts_rs, '--', linewidth=2, label='Test ratings')
        ax.set_xticks(x)
        ax.set_xticklabels([datetime.strftime(d, "%m.%y") for d in dates], rotation=45)
        ax.set_yticks(range(0, max(tr_rs) + 1, 100000))
        # ax.semilogy()
        ax.set_yscale("log")
        ax.set_ylabel("count (log-scale)", fontsize=12)
        ax.grid(True)
        ax.legend()

        ax = axes[1]
        ax.plot(x, tr_us, '--', linewidth=2, label='Training users')
        ax.plot(x, ts_us, '--', linewidth=2, label='Test users')
        ax.set_xticks(x)
        ax.set_xticklabels([datetime.strftime(d, "%m.%y") for d in dates], rotation=45)
        ax.set_yticks(range(0, max(tr_us), 5000))
        ax.set_yscale("log")
        ax.set_ylabel("count (log-scale)", fontsize=12)
        ax.grid(True)
        ax.legend()

        ax = axes[2]
        ax.plot(x, ts_ps, '--', linewidth=2, label='Test papers')
        ax.plot(x, tr_ps, '--', linewidth=2, label='Training papers')
        ax.set_xticks(x)
        ax.set_xticklabels([datetime.strftime(d, "%m.%y") for d in dates], rotation=45)
        ax.set_ylabel("count (log-scale)", fontsize=12)
        ax.set_yticks(range(0, max(tr_ps), 20000))

        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()

        ax = axes[3]
        ax.plot(x, rat, '--', linewidth=2, label='Test/Training ratings ratio ')
        ax.set_xticks(x)
        ax.set_xticklabels([datetime.strftime(d, "%m.%y") for d in dates], rotation=45)
        ax.set_yticks(range(0, int(max(rat)) + 1), max(rat)//10 )
        ax.set_ylabel("Test/Training %", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        # plt.ylabel("Test/Training %", fontsize=12)
        ax.grid(True)
        ax.legend()

        for i in range(4):
            format_grid(axes[i])

        plt.savefig(os.path.join(self.base_dir,'time-based_split_out-of-matrix' if self.out_of_matrix else 'time-based_split_in-matrix', 'time-based_splits_statistics.png'))

    def integrate_raings_timestamp(self, users_dict, papers_dict):
        """
        Integrate the ratings with their timestamps from the current file.
        :return:  a list of (user_id, paper_id, date)
        """

        # 2- Read current
        currentFile = os.path.join(self.base_dir, 'current')
        # user_id, paper_id -> date
        ratings_array = np.zeros(self.ratings.shape)
        # user_id, paper_id, date:
        ratings_list = []
        with open(currentFile) as f:
            hit = 0
            ratings_hit = 0
            for line in f:
                citeulike_paperid, citeulikeHash, timestamp, _ = line.split("|")
                if citeulikeHash in users_dict and citeulike_paperid in papers_dict:
                    hit += 1
                    user_id = users_dict[citeulikeHash]
                    paper_id = papers_dict[citeulike_paperid]
                    if self.ratings[user_id, paper_id]:
                        date = timestamp.split(' ')[0]
                        if ratings_array[user_id, paper_id] == 0:
                            ratings_hit += 1
                            ratings_array[user_id, paper_id] = 1
                            ratings_list.append((user_id, paper_id, datetime.strptime(date, "%Y-%m-%d").date()))
        return ratings_list

    def generate_fold(self, split_date, train_df, test_df):
        # 1- Extract the users and assign ids
        users_df = pd.DataFrame(data=pd.concat([train_df.user, test_df.user]).unique(), columns=['user'])
        # index -> user_id:
        users_dict = users_df.to_dict()['user']

        # Save it in a dataframe:
        useridx_user_id_map_list = [(i, j) for (i, j) in users_dict.items()]
        useridx_user_id_map_list.sort(key=lambda x: x[0])
        users_df = pd.DataFrame(data=useridx_user_id_map_list, columns=['user_idx', 'user'])
        useridx_user_id_map_list = [j for _, j in useridx_user_id_map_list]

        # The same for papers:
        papers_df = pd.DataFrame(data=pd.concat([train_df.paper, test_df.paper]).unique(), columns=['paper'])
        papers_dict = papers_df.to_dict()['paper']
        paperidx_paper_id_map_list = [(i, j) for (i, j) in papers_dict.items()]
        paperidx_paper_id_map_list.sort(key=lambda x: x[0])
        papers_df = pd.DataFrame(data=[(i, j) for (i, j) in papers_dict.items()], columns=['paper_idx', 'paper'])
        paperidx_paper_id_map_list = [j for _, j in paperidx_paper_id_map_list]

        # 2- Replace the old ids with the new ids (the index, new name: for user(paper) is: user(paper)_idx)
        # For users:
        merged_df = pd.merge(users_df, train_df, on=['user'])[['user_idx', 'paper', 'date']]
        # For papers:
        merged_train_df = pd.merge(merged_df, papers_df, on=['paper'], how='outer')[['user_idx', 'paper_idx', 'date']].sort_values(['user_idx', 'date'])

        # Repeat the same for the test data, (note that not all users and papers from training appear in the test data, therefore we are using outer join here):
        # Outer join with the papers_df to replace the paper ids with the new ids, and keep the new_ids which don't have matches in the test data:
        merged_df = pd.merge(test_df, papers_df, on=['paper'], how='right')[['user', 'paper_idx', 'date']]
        # Outer join with the users_df to replace the user ids with the new ids, and keep the new_ids which don't have matches in the test data:
        merged_test_df = pd.merge(users_df, merged_df, on=['user'], how='outer')[['user_idx', 'paper_idx', 'date']].sort_values(['user_idx', 'date'])

        # 3- Group items ratings in a single line:
        train_l_items = merged_train_df.groupby('paper_idx')[['user_idx']].agg(lambda x: x.values.tolist()).values.tolist()
        train_l_items = [[] if math.isnan(i[0][0]) else i[0] for i in train_l_items]
        test_l_items = merged_test_df.groupby('paper_idx')[['user_idx']].agg(lambda x: x.values.tolist()).values.tolist()
        test_l_items = [[] if math.isnan(i[0][0]) else [int(j) for j in i[0]] for i in test_l_items]

        # 4- Replace the date with the rating age in the training data:
        merged_train_df['age'] = (split_date - merged_train_df['date']).apply(lambda x: "{:4.2f}".format(abs(x.days/30)))
        #merged_train_df['rating_age'] = merged_train_df['paper_idx'].apply(lambda x: str(x)) + ":" + merged_train_df['age'].apply(lambda x: "{:4.2f}".format(abs(x)))
        #merged_train_df = merged_train_df[['user_idx', 'rating_age']]

        # 5- Group users ratings in a single line:
        train_l_users_age = merged_train_df.groupby('user_idx')[['age']].agg(lambda x: x.values.tolist())['age'].values.tolist()
        train_l_users_age = [[] if i[0]== 'nan' else i for i in train_l_users_age]

        #train_l_users = merged_train_df.groupby('user_idx')[['age']].agg(lambda x: x.values.tolist())['age'].values.tolist()
        #train_l_users = [[] if i[0] == 'nan:nan' else i for i in train_l_users]        

        train_l_users = merged_train_df.groupby('user_idx')[['paper_idx']].agg(lambda x: x.values.tolist())['paper_idx'].values.tolist()
        train_l_users = [[] if math.isnan(i[0]) else i for i in train_l_users]

        test_l_users = merged_test_df.groupby('user_idx')[['paper_idx', 'date']].agg(lambda x: x.values.tolist()).values.tolist()
        test_l_users = [[] if math.isnan(i[0][0]) else [int(j) for j in i[0]] for i in test_l_users]
        n_users = len(users_dict)
        n_papers = len(papers_dict)
        return train_l_users, train_l_users_age, train_l_items, test_l_users, test_l_items, useridx_user_id_map_list, paperidx_paper_id_map_list, n_users, n_papers

    def generate_docs_terms(self, docs_vocabs, remaining_p_ids, terms, output_dir):
        """
        Generating the new mult.dat file and terms.dat file, which contains only the terms that appear in the selected papers (p_ids)
        """
        # 1-Get the term ids of the remaining papers:
        term_ids = set()
        for p in remaining_p_ids:
            for e in docs_vocabs[p]:
                term_ids.add(e[0])
        term_ids = sorted(list(term_ids))
        t_dict = dict(zip(term_ids, range(len(term_ids))))

        # 2- Generate new mult (reduced)
        new_papers = []
        for p in remaining_p_ids:
            new_papers.append([(t_dict[t], tf) for (t, tf) in docs_vocabs[p]])

        # 3- Save the new mult:
        save_mult_list(new_papers, os.path.join(output_dir, "mult.dat"))

        # 4- Generate the new terms file:
        # Filter the terms:
        new_terms = terms[term_ids]

        # 5- Save the new terms:
        save_list(new_terms, os.path.join(output_dir, "terms.dat"))

    def split(self):
        # Get the mapping as a list of user_hash where the key is the corresponding index:
        userhash_userid_map_list = list(self.users_dict.items())
        userhash_userid_map_list.sort(key=lambda x: x[1])
        user_id_userhash_map_list = np.array([i for (i, _) in userhash_userid_map_list])

        # Get the mapping as a list of doc_ids where the key is the corresponding index:
        docid_paperid_map_list = list(self.papers_dict.items())
        docid_paperid_map_list.sort(key=lambda x: x[1])
        paper_id_docid_map_list = np.array([i for (i, _) in docid_paperid_map_list])

        # Get the ratings list integrated with time stamps:
        ratings_list = self.integrate_raings_timestamp(self.users_dict, self.papers_dict)

        fr = pd.DataFrame(data=ratings_list, columns=['user', 'paper', 'date'])
        print("Ratings: {}, users: {}, papers: {}.".format(len(fr), fr.user.nunique(), fr.paper.nunique()))

        # First split date:
        d1 = datetime.strptime('2005-03-31', "%Y-%m-%d").date()

        # Last date:
        last_date = fr.date.max()
        ratings_period = (last_date.year - d1.year) * 12 + last_date.month

        # These lists are used for plotting:
        tr_rs, tr_us, tr_ps, ts_rs, ts_us, ts_ps, rat, dates = [], [], [], [], [], [], [], []

        folds_num = ratings_period // self.split_duration

        # For split stats:
        stats_header = ['{:4}'.format('Fold'), '{:20}'.format('#Usrs(Tot,R,S)'),'{:23}'.format('#Itms(Tot,R,S)'),'{:23}'.format('#Rtng(Tot,R,S)'),\
                        '{:23}'.format('PRU(min/max/avg/std)'), '{:22}'.format('PSU(min/max/avg/std)'), '{:20}'.format('PRI(min/max/avg/std)'), '{:20}'.format('PSI(min/max/avg/std)')]
        self.stat_list.append(stats_header)
        stats = Stats()
        for fold in range(folds_num):
            d2 = d1 + relativedelta(months=self.split_duration)

            # Training ratings:
            f1 = fr[fr['date'] < d1]

            # Test ratings:
            if self.out_of_matrix:
                f2 = fr[(fr['date'] >= d1) & (fr['date'] < d2) & fr['user'].isin(f1['user'])]
            else:
                f2 = fr[(fr['date'] >= d1) & (fr['date'] < d2) & fr['user'].isin(f1['user']) & (fr['paper'].isin(f1['paper']))]
            print("{}->{}, Tr:[Rs: {:6}, Us: {:5}, Ps: {:6}], Te:[Rs: {:5}, Us: {:5}, Ps: {:6}], Ratio: {:04.2f}%"\
                  .format(d1, d2, len(f1), f1.user.nunique(), f1.paper.nunique(), len(f2), f2.user.nunique(), f2.paper.nunique(), len(f2) / len(f1) * 100))

            # Generate data for the folds:
            train_l_users, train_l_users_age, train_l_items, test_l_users, test_l_items, useridx_user_id_map_list, paperidx_paper_id_map_list, n_users, n_papers = self.generate_fold(d1,f1, f2)
            stats.add_fold_statistics(fold+1, train_l_users, test_l_users, train_l_items, test_l_items)
            """
            tru = [len(i) for i in train_l_users]
            tsu = [len(i) for i in test_l_users]
            tri = [len(i) for i in train_l_items]
            tsi = [len(i) for i in test_l_items]
            self.stat_list.append(['{:4}'.format(fold + 1), '{:5d} / {:5d} / {:4d}'.format(n_users, f1.user.nunique(), f2.user.nunique()),
                                   '{:6d} / {:6d} / {:5d}'.format(n_papers, f1.paper.nunique(), f2.paper.nunique()),\
                                   '{:6d} / {:6d} / {:5d}'.format(f1.shape[0]+ f2.shape[0], f1.shape[0], f2.shape[0]), \
                                   '{:1d} / {:4d} / {:4.1f} / {:5.1f}'.format(np.min(tru), np.max(tru), np.mean(tru), np.std(tru)),\
                                   '{:1d} / {:4d} / {:4.1f} / {:4.1f}'.format(np.min(tsu), np.max(tsu), np.mean(tsu), np.std(tsu)),\
                                   '{:1d} / {:3d} / {:4.1f} / {:3.1f}'.format(np.min(tri), np.max(tri), np.mean(tri), np.std(tri)),\
                                   '{:1d} / {:3d} / {:4.1f} / {:3.1f}'.format(np.min(tsi), np.max(tsi), np.mean(tsi), np.std(tsi))])
            """

            # Write to file:
            fold_folder = os.path.join(self.base_dir,'time-based_split_out-of-matrix' if self.out_of_matrix else 'time-based_split_in-matrix', 'fold-{}'.format(fold+1))
            if not os.path.exists(fold_folder):
                os.makedirs(fold_folder)

            write_ratings(train_l_users, os.path.join(fold_folder, 'train-users.dat'))
            write_ratings(train_l_users_age, os.path.join(fold_folder, 'train-users-ages.dat'))
            write_ratings(test_l_users, os.path.join(fold_folder, 'test-users.dat'))
            write_ratings(train_l_items, os.path.join(fold_folder, 'train-items.dat'))
            write_ratings(test_l_items, os.path.join(fold_folder, 'test-items.dat'))

            print("Generating the new mult file...")
            self.generate_docs_terms(self.docs_vocabs, paperidx_paper_id_map_list, self.terms, fold_folder)

            # Write users and papers mappings to files:
            useridx_userhash = user_id_userhash_map_list[useridx_user_id_map_list]
            write_list_to_file([(j, i) for (i, j) in enumerate(useridx_userhash)], os.path.join(fold_folder, 'citeulikeUserHash_userId_map.dat'), header=['citeulikeUserHash', 'user_id'])

            paperidx_docid = paper_id_docid_map_list[paperidx_paper_id_map_list]
            write_list_to_file([(j, i) for (i, j) in enumerate(paperidx_docid)], os.path.join(fold_folder, 'citeulikeId_docId_map.dat'), header=['citeulikeId', 'paper_id'])

            # For plotting:
            dates.append(d2)
            tr_rs.append(len(f1))
            tr_us.append(f1.user.nunique())
            tr_ps.append(f1.paper.nunique())
            ts_rs.append(len(f2))
            ts_us.append(f2.user.nunique())
            ts_ps.append(f2.paper.nunique())
            rat.append(len(f2) / len(f1) * 100)
            d1 = d2
        self.plot_split_lines(tr_rs, tr_us, tr_ps, ts_rs, ts_us, ts_ps, rat, dates)

        # Write split statistics to file:
        stats.save_stats_to_file(os.path.join(self.base_dir,'time-based_split_out-of-matrix' if self.out_of_matrix else 'time-based_split_in-matrix', 'stats.txt'))

if __name__ == '__main__':
    data_folder = "../../citeulike-crawled/converted_data/2k_1_P3/terms_keywords_based"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", "-d", help="The folder containing data, the users' ratings file")
    parser.add_argument("--split_duration", "-du", type=int, required=True, help="The split duration in months")
    parser.add_argument("--outofmatrix", "-o", action="store_true", default=False,
                        help="A flag orders the code to allow out of matrix ratings: papers that appear only in the test set.")

    args = parser.parse_args()
    if args.data_directory:
        data_folder = args.data_directory
    if args.split_duration:
        split_duration = args.split_duration
    out_of_matrix = False
    if args.outofmatrix:
        out_of_matrix = True
    time_splitter = TimeBasedSplitter(data_folder, split_duration, out_of_matrix)
    time_splitter.split()
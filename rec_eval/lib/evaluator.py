"""
Author: Anas Alzogbi
Description: this module provides the functionality of:
 - Calculating predictions given two factor matrices
 - Calculating evaluation metrics given score predictions and a test set
Date: October 22nd, 2017
"""
import sys
import os
import glob
import pandas as pd
import numpy as np
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rec_eval.util.files_utils import read_ratings
from rec_eval.util.files_utils import print_list

breaks = range(20, 201, 20)
delimiter = " "


class Evaluator(object):
    def __init__(self, num_users, experiment_directory):
        self.num_users = num_users
        self.experiment_directory = experiment_directory
        # Get the experiment name (the filder name of the experiment_directory path)
        self.experiment_name = os.path.basename(os.path.normpath(self.experiment_directory))
        self.results_list =[]

    def calculate_metrics_user(self, hits, num_user_test_positives, recall_breaks, mrr_breaks, ndcg_breaks):
        # Adjust the breaks lists to be 0-based:
        recall_breaks = [i - 1 for i in recall_breaks]
        mrr_breaks = [i - 1 for i in mrr_breaks]
        ndcg_breaks = [i - 1 for i in ndcg_breaks]
        iDCGs = np.cumsum(np.array([1 / np.log2(i + 2) for i in range(len(hits))]))

        # Calculate recall:
        recall = np.cumsum(hits)
        recall_at_breaks = (np.array(recall)[recall_breaks] / float(num_user_test_positives)).tolist()

        # Calculate MRR
        mrrs = [hits[i] / float(i + 1) for i in range(len(hits))]
        for i in range(1, len(mrrs)):
            mrrs[i] = max(mrrs[i], mrrs[i - 1])
        mrrs_at_breaks = np.array(mrrs)[mrr_breaks].tolist()

        # Calculate nDCG
        dcgs = [hits[i] / np.log2(i + 2) for i in range(len(hits))]
        dcgs = np.array(dcgs)
        dcgs = np.cumsum(dcgs) / iDCGs
        ndcgs_at_breaks = dcgs[ndcg_breaks].tolist()
        return recall_at_breaks + mrrs_at_breaks + ndcgs_at_breaks


    def calculate_metrics_fold(self, users_test, candidate_items, scores, results_users_file, results_header, top, recall_breaks, mrr_breaks, ndcg_breaks, fold=-1):
        num_users, num_items = scores.shape
        results = np.zeros(shape=(num_users, len(recall_breaks) + len(mrr_breaks) + len(ndcg_breaks)))
        users_with_zero_test = 0
        print("Computing {}...\n".format(results_users_file))
        with open(results_users_file, 'w') as f:
            row_header = ['{:7}'.format('user_id')] + results_header
            f.write('{}'.format(' '.join(map(str, row_header)) + '\n'))
            for user in range(num_users):
                # Get the test positive items
                user_test_positive = users_test[user]
                if len(user_test_positive) == 0:
                    users_with_zero_test += 1
                    continue

                # Get the prediction scores for the candidate items
                scores_u = scores[user, candidate_items[user]]

                # Identify the top recommendations
                recommended_items_idx = np.argsort(scores_u)[::-1][0:top]
                # recommended_items_ids = candidate_items[user][recommended_items_idx]
                recommended_items_ids = np.array(candidate_items[user])[recommended_items_idx]
                # Identify the hits:
                hits = [1 if i in user_test_positive else 0 for i in recommended_items_ids]

                # Calculate the metrics:
                metrics_values = self.calculate_metrics_user(hits, len(user_test_positive), recall_breaks, mrr_breaks, ndcg_breaks)

                # Write results to the file
                l = map(str, ['{:7d}'.format(user), '{:7d}'.format(fold + 1)] + ["{:7.3f}".format(i) for i in metrics_values])
                f.write('{}'.format(' '.join(l) + '\n'))
                results[user, :] = metrics_values
        print(" Fold {} Results, users with zero test items = {}: ".format(fold + 1, users_with_zero_test))
        print_list(results_header)
        print_list(['{:7d}'.format(fold + 1)] + ["{:7.3f}".format(i) for i in np.average(results, axis=(0)).tolist()])
        return results


    def compute_metrics_all_folds(self, test_paths, exp_paths, splits, recall_breaks=[5, 10] + list(range(20, 201, 20)), mrr_breaks=[10], ndcg_breaks=[5, 10], folds_num=5, top=200):
        # Check if the breaks size is suitable to the splits array, the max number in splits must be less or equal the
        # minimum number of papers in the test set for each user, otherwise it is not applicable!
        len_fun = np.vectorize(lambda x: len(x))
        min_test_size = len_fun(splits).min()
        recall_breaks =[i for i in recall_breaks if i <=min_test_size]
        mrr_breaks = [i for i in mrr_breaks if i <= min_test_size]
        ndcg_breaks = [i for i in ndcg_breaks if i <= min_test_size]

        # Initialize the 3d results matrix (folds x users, metrics)
        results = np.zeros(shape=(folds_num, self.num_users, len(recall_breaks) + len(mrr_breaks) + len(ndcg_breaks)))
        results_list = []
        results_header = ["Rec@" + str(i) for i in recall_breaks] + ["MRR@" + str(i) for i in mrr_breaks] + ["nDCG@" + str(i) for i in ndcg_breaks]
        results_header = ['{:7}'.format('fold')] + ['{:7}'.format(h) for h in results_header]
        results_list.append(results_header)

        for fold in range(folds_num):
            score_path = "{}/score.npy".format(exp_paths[fold])
            if not os.path.exists(score_path):
                print("Scores file {} is not available".format(score_path))
                return
            scores = np.load(score_path)
            users_test = read_ratings(os.path.join(test_paths[fold], "test-fold_{}-users.dat".format(fold + 1)))
            results_users_file = os.path.join(exp_paths[fold], "results-users.dat")

            # Get the candidate items for this fold
            candidate_items = np.array(splits[:, fold])

            # Calculate the results and save them in the tensor
            results[fold] = self.calculate_metrics_fold(users_test, candidate_items, scores, results_users_file, results_header, top, recall_breaks, mrr_breaks, ndcg_breaks, fold)
            results_list.append(['{:7d}'.format(fold + 1)] + ["{:7.3f}".format(i) for i in np.average(results[fold], axis=(0)).tolist()])

        print("Average Results over all folds: ")
        print_list(results_header[1:])
        print_list(["{:7.3f}".format(i) for i in np.average(results, axis=(0, 1)).tolist()])
        results_list.append([('-' * (9 * len(results_header) - 1))])
        results_list.append(['{:7}'.format('avg')] + ["{:7.3f}".format(i) for i in np.average(results, axis=(0, 1)).tolist()])

        self.results_list = results_list

        np.save(os.path.join(self.experiment_directory, "results_matrix"), results)
        # Writing the results to a file:
        # with open(os.path.join(self.experiment_directory, "Exp_" + self.experiment_name + "_eval_results.txt"), 'w', newline='') as f:
        with open(os.path.join(self.experiment_directory, "Exp_" + self.experiment_name + "_eval_results.txt"), 'w') as f:
            for s in results_list:
                f.write('[%s]' % ', '.join(map(str, s)) + '\n')

    def score(self, folder_path):
        score_path = os.path.join(folder_path, "score")
        if os.path.exists(score_path + ".npy"):
            print("Score file [{}] already exists, exiting.".format(score_path + ".npy"))
            return

        u_path = os.path.join(folder_path, "final-U.dat")
        if not os.path.exists(u_path):
            print("U file {} is not found".format(u_path))
            return

        print("Reading U file...{}".format(u_path))
        U = pd.read_csv(u_path, sep=' ', header=None).iloc[:, 0:-1]

        v_path = os.path.join(folder_path, "final-V.dat")
        if not os.path.exists(v_path):
            print("V file {} is not found".format(v_path))
            return

        print("Reading V file...{}".format(v_path))
        V = pd.read_csv(v_path, sep=' ', header=None).iloc[:, 0:-1]

        print("Multiplication...")
        scores = np.dot(U, V.T)

        print("Saving scores file...{}".format(score_path + ".npy"))
        np.save(score_path, scores)

    def score_all(self):
        folds_folders = glob.glob(self.experiment_directory + "/fold-*")
        folds_folders.sort()
        for f in [f for f in folds_folders]:
            print("Scoring {} ...".format(f))
            self.score(f)


    def eval_experiment(self, split_directory, folds=1, score=True):

        if not os.path.exists(split_directory):
            print("Split directory not found: {}".format(split_directory))
            raise NameError("Split directory not found")

        if score:
            self.score_all()
        # 2- Evaluate:
        splits_file = os.path.join(split_directory, "splits.npy")

        if not os.path.exists(splits_file):
            print("Splits file not found: {} ".format(splits_file))
            raise NameError("Splits file not found")

        print("loading {} ...\n".format(splits_file))
        splits = np.load(splits_file)

        test_paths = glob.glob(split_directory + "/fold-*")
        test_paths.sort()
        exp_paths = glob.glob(self.experiment_directory + "/fold-*")
        exp_paths.sort()
        self.compute_metrics_all_folds(test_paths, exp_paths, splits, folds_num=folds)
        # avg_results = list(map(float, self.results_list[-1][1:]))
        return self.results_list


if __name__ == '__main__':
    users_rtings_file = "../../data/users.dat"
    split_directory = '../../data/in-matrix-item_folds'
    experiment_directory = '../../data/in-matrix-item_folds/CTR_k_200'
    folds = 5
    parser = argparse.ArgumentParser()
    #parser.add_argument("--users_ratings", "-u", help="The complete users ratings file (not split) users.dat")
    parser.add_argument("--score", "-s", action="store_true", default=False,
                        help="A flag orders the code to calculate the score matrix: U.VT, default: the score is not calculated, assuming the existence of the files [experiment_directory]/fold-[1-5]/score.npy")
    parser.add_argument("--split_directory", "-p", required = True,
                        help="The directory that contains the folds. The folds folders are named as 'fold[1-5]', each one should contain the test files")
    parser.add_argument("--experiment_directory", "-x", required = True,
                        help="The directory that contains the experiment data, one folder for each fold (fold-[1-5]), each of them contains two files: final-U.dat and final-V.dat")
    parser.add_argument("--folds", "-f", help="The number of folds", type=int)
    parser.add_argument("--num_users", "-nu", required = True, help="The number of users", type=int)
    parser.add_argument("--evaluate_single_fold", "-es", help="The fold name to be evaluated. This causes the evaluator to evaluate this fold only.")
    args = parser.parse_args()
    """
    if args.users_ratings:
        users_rtings_file = args.users_ratings
        if not os.path.exists(users_rtings_file):
            print("Ratings file not found: {}".format(users_rtings_file))
            raise NameError("Ratings file not found")
    """
    if args.num_users:
        num_users = int(args.num_users)

    if args.split_directory:
        split_directory = args.split_directory
        if not os.path.exists(split_directory):
            print("Split directory not found: {}".format(split_directory))
            raise NameError("Split directory not found")

    if args.experiment_directory:
        experiment_directory = args.experiment_directory
        if not os.path.exists(experiment_directory):
            print("Experiment directory not found: {}".format(experiment_directory))
            raise NameError("Experiment directory not found")

    if args.folds:
        folds = args.folds

    evaluator = Evaluator(num_users, experiment_directory)

    if args.evaluate_single_fold:

        # 1- score:
        if args.score:
            evaluator.score(experiment_directory)

        # 2- evaluate:
        score_path = "{}/score.npy".format(experiment_directory)
        if os.path.exists(score_path):

            # Load the prediction scores:
            scores = np.load(score_path)
            num_items = scores.shape[1]


            # Load users ratings (training and test), the training is needed to build the candidate set later:
            users_test = read_ratings(glob.glob(os.path.join(args.evaluate_single_fold, "test-fold_*-users.dat"))[0])
            users_train = read_ratings(glob.glob(os.path.join(args.evaluate_single_fold, "train-fold_*-users.dat"))[0])

            # Generate the candidate items for each user (all items except those appear in the user's training)
            candidate_items = []
            all_items_ids = set(range(num_items))
            for i,u in enumerate(users_test):
                candidate_items.append(list(all_items_ids - set(users_train[i])))
            candidate_items = np.array(candidate_items)

            # Evaluation metrics:
            recall_breaks = [5, 10] + list(range(20, 201, 20))
            mrr_breaks = [10]
            ndcg_breaks = [5, 10]
            top = 200

            # Result header:
            results_header = ["Rec@" + str(i) for i in recall_breaks] + ["MRR@" + str(i) for i in mrr_breaks] + ["nDCG@" + str(i) for i in ndcg_breaks]
            results_header = ['{:7}'.format(h) for h in results_header]

            # Calculate the results
            results_users_file = os.path.join(args.evaluate_single_fold, "results-users.dat")
            evaluator.calculate_metrics_fold(users_test, candidate_items, scores, results_users_file, results_header, top, recall_breaks, mrr_breaks, ndcg_breaks)

        # The score file doesn' exist
        else:
            print("Scores file {} is not available".format(score_path))

    # Evaluate multiple folds together:
    else:
        # 1- Score if needed:
        if args.score:
            evaluator.score_all()

        # 2- Evaluate:
        splits_file = os.path.join(split_directory, "splits.npy")

        if not os.path.exists(splits_file):
            print("Splits file not found: {} ".format(splits_file))
            raise NameError("Splits file not found")

        print("loading {} ...\n".format(splits_file))
        splits = np.load(splits_file)

        test_paths = glob.glob(split_directory + "/fold-*")
        test_paths.sort()
        exp_paths = glob.glob(experiment_directory + "/fold-*")
        exp_paths.sort()
        evaluator.compute_metrics_all_folds(test_paths, exp_paths, splits, folds_num=folds)

from utils import *
from sklearn.utils import resample
from matrix_factorization import als
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt

def bootstrap_dict(dict_data, random_state):
    n = len(dict_data["user_id"])
    boot_questions = resample(dict_data["question_id"], replace=True, n_samples=n, random_state=random_state)
    boot_users = resample(dict_data["user_id"], replace=True, n_samples=n, random_state=random_state)
    boot_correct = resample(dict_data["is_correct"], replace=True, n_samples=n, random_state=random_state)

    boot = dict()
    boot["question_id"] = boot_questions
    boot["user_id"] = boot_users
    boot["is_correct"] = boot_correct

    return boot

def bagged_als(num_estimators, K, LR, n_iter, lam, train_data, val_data, bootstrap=False):
    """
    Train 10 matrix factorization models and get their average result.
    """

    pred = None
    for i in tqdm(range(num_estimators)):
        if not bootstrap:
            result, *_ = als(bootstrap_dict(train_data, i), K, LR, n_iter, lam, val_data, collect=False, verbose=False)
        else:
            result, *_ = als(train_data, K, LR, n_iter, lam, val_data, collect=False, verbose=False)
        if pred is None:
            pred = result
        else:
            pred += result

    return pred / num_estimators

def regularized_als(K, LR, n_iter, lam, train_data, val_data):
    """
    Train 1 matrix factorization models and get the result.
    """
    result, *_ = als(train_data, K, LR, n_iter, lam, val_data, collect=False, verbose=False)

    return result


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    n = 10
    K = 50
    LR = 0.05
    n_iter = 200000
    lam = 0.01
    
    # not regularized, not ensemble
    pred_vot_reg = bagged_als(n, K, LR, n_iter, lam, train_data, val_data)
    # ensemble but not regularized
    pred_vot = bagged_als(n, K, LR, n_iter, 0, train_data, val_data)
    # regularized but no ensemble
    pred_reg = regularized_als(K, LR, n_iter, lam, train_data, val_data)
    # both regularized and ensemble
    pred_none = regularized_als(K, LR, n_iter, 0, train_data, val_data)


    print('START HERE \n Not Regularized, No Ensemble:')
    print("Training perf: {}".format(sparse_matrix_evaluate(train_data, pred_none)))
    print("Validation perf: {}".format(sparse_matrix_evaluate(val_data, pred_none)))
    print("Test: {}".format(sparse_matrix_evaluate(test_data, pred_none)))
    print('\n\n /////////////////////////////////// \n\n')

    print('Not Regularized, but with Ensemble:')
    print("Training perf: {}".format(sparse_matrix_evaluate(train_data, pred_vot)))
    print("Validation perf: {}".format(sparse_matrix_evaluate(val_data, pred_vot)))
    print("Test: {}".format(sparse_matrix_evaluate(test_data, pred_vot)))
    print('\n\n /////////////////////////////////// \n\n')

    print('Regularized, but no Ensemble:')
    print("Training perf: {}".format(sparse_matrix_evaluate(train_data, pred_reg)))
    print("Validation perf: {}".format(sparse_matrix_evaluate(val_data, pred_reg)))
    print("Test: {}".format(sparse_matrix_evaluate(test_data, pred_reg)))
    print('\n\n /////////////////////////////////// \n\n')

    print('Regularized, with Ensemble:')
    print("Training perf: {}".format(sparse_matrix_evaluate(train_data, pred_vot_reg)))
    print("Validation perf: {}".format(sparse_matrix_evaluate(val_data, pred_vot_reg)))
    print("Test: {}".format(sparse_matrix_evaluate(test_data, pred_vot_reg)))
    print('\n\n /////////////////////////////////// \n\n')


if __name__ == "__main__":
    main()
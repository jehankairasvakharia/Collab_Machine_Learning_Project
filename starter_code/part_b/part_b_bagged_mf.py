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


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    n = 10
    K = 50
    LR = 0.05
    n_iter = 200000
    lam = 0.01
    pred = bagged_als(n, K, LR, n_iter, lam, train_data, val_data)

    print("Training perf: {}".format(sparse_matrix_evaluate(train_data, pred)))
    print("Validation perf: {}".format(sparse_matrix_evaluate(val_data, pred)))
    print("Test: {}".format(sparse_matrix_evaluate(test_data, pred)))


if __name__ == "__main__":
    main()
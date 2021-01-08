from utils import *
from scipy.linalg import sqrtm
from sklearn.utils import resample

from matrix_factorization import als

import numpy as np
import matplotlib.pyplot as plt


def main():
    """Train 10 matrix factorization models and get their average result"""

    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    n = 10
    pred = None
    for i in range(n):
        result, train_losses, val_losses = als(train_data, 50, 0.05,
                                               200000, val_data, False, 0.01)
        if pred is None:
            pred = result
        else:
            pred += result

    pred = pred / n

    print("Training perf: {}".format(sparse_matrix_evaluate(train_data, pred)))
    print("Validation perf: {}".format(sparse_matrix_evaluate(val_data, pred)))
    print("Test: {}".format(sparse_matrix_evaluate(test_data, pred)))


if __name__ == "__main__":
    main()
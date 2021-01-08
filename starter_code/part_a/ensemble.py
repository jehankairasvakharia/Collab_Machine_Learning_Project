# TODO: complete this file.
from sklearn.impute import KNNImputer
from matrix_factorization import *
from item_response import *
from utils import *
from knn import *
from sklearn.utils import resample
import numpy as np

def final_evaluate(data, predictions, threshold=0.5):
    """ Return the accuracy of the predictions given the data.
    Multiple imported files use a function called evaluate, so this clears that up for this file.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param predictions: list
    :param threshold: float
    :return: float
    """
    if len(data["is_correct"]) != len(predictions):
        raise Exception("Mismatch of dimensions between data and prediction.")
    if isinstance(predictions, list):
        predictions = np.array(predictions).astype(np.float64)
    return (np.sum((predictions >= threshold) == data["is_correct"])
            / float(len(data["is_correct"])))



# Bootstrapped training data
def bootstrap_dict(dict_data, random_state=1):
    """Return a bootstrapped version of the dictionary data given, using a particular random state
    :param dict_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param random_state: a random state that defines how the random sampling occurs

    :return: a bootstrapped version of the dictionary data
    """

    n = len(dict_data["user_id"])
    boot_questions = resample(dict_data["question_id"], replace=True, n_samples=n, random_state=random_state)
    boot_users = resample(dict_data["user_id"], replace=True, n_samples=n, random_state=random_state)
    boot_correct = resample(dict_data["is_correct"], replace=True, n_samples=n, random_state=random_state)

    boot = dict()
    boot["question_id"] = boot_questions
    boot["user_id"] = boot_users
    boot["is_correct"] = boot_correct

    return boot

def predict_irt(data, theta, beta):
    """ Evaluate the model given data and return the accuracy, using IRT.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.array(pred)


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Bootstrapping for ALS and IRT
    boot_2 = bootstrap_dict(train_data, 3)
    boot_3 = bootstrap_dict(train_data, 5)



    # KNN by User, optimal k: 11
    k = 11
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(train_matrix)
    prediction_1_val = np.array(sparse_matrix_predictions(val_data, mat))
    prediction_1_test = np.array(sparse_matrix_predictions(test_data, mat))

    acc_1_val = sparse_matrix_evaluate(val_data, mat)
    print(f'Prediction 1 KNN Validation Accuracy:{acc_1_val}')
    acc_1_test = sparse_matrix_evaluate(test_data, mat)
    print(f'Prediction 1 KNN Test Accuracy:{acc_1_test}')



    # matrix factorization ALS
    mk = 50
    result_val, train_loss_val, val_loss = als(boot_2, mk, 0.05, 200000, val_data)
    # result_test, train_loss_test, test_loss = als(boot_2, mk, 0.05, 200000, test_data)

    prediction_2_val = np.array(sparse_matrix_predictions(val_data, result_val))
    prediction_2_test = np.array(sparse_matrix_predictions(test_data, result_val))

    acc_2_val = sparse_matrix_evaluate(val_data, result_val)
    print(f'Prediction 2: ALS Validation Accuracy:{acc_2_val}')
    acc_2_test = sparse_matrix_evaluate(test_data, result_val)
    print(f'Prediction 2: ALS Test Accuracy:{acc_2_test}')



    # IRT
    LR = 0.01
    N_iter = 40
    theta_val, beta_val, acc_3_val = irt(boot_3, val_data, LR, N_iter)
    theta_test, beta_test, acc_3_test = irt(boot_3, test_data, LR, N_iter)

    prediction_3_val = np.array(predict_irt(val_data, theta_val, beta_val))
    prediction_3_test = np.array(predict_irt(test_data, theta_test, beta_test))

    print(f'Prediction 3 IRT Validation Accuracy:{acc_3_val[-1]}')
    print(f'Prediction 3 IRT Test Accuracy:{acc_3_test[-1]}')



    # ENSEMBLE: Knn + ALS + IRT
    prediction_bagged_avg_val = prediction_1_val + prediction_2_val + prediction_3_val
    prediction_bagged_avg_val = np.round(prediction_bagged_avg_val / 3)

    prediction_bagged_avg_test = prediction_1_test + prediction_2_test + prediction_3_test
    prediction_bagged_avg_test = np.round(prediction_bagged_avg_test / 3)

    acc_bagged_avg_val = final_evaluate(val_data, prediction_bagged_avg_val)
    print(f'Bagged Ensemble Validation Accuracy:{acc_bagged_avg_val}')

    acc_bagged_avg_test = final_evaluate(test_data, prediction_bagged_avg_test)
    print(f'Bagged Ensemble Test Accuracy:{acc_bagged_avg_test}')


if __name__ == '__main__':
    main()
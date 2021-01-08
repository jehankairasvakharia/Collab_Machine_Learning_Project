# TODO: complete this file.
from collections import Counter
from sklearn.impute import KNNImputer
from tqdm.auto import tqdm
from matrix_factorization import *
from item_response import *
from utils import *
from knn import *
from scipy.linalg import sqrtm
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import partb_randomforest

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def final_evaluate(data, predictions, threshold=0.5):
    """ Return the accuracy of the predictions given the data.

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


train_matrix = load_train_sparse("../data").toarray()
train_data = load_train_csv("../data")

# val_data = load_valid_csv("../data")
val_data = load_public_test_csv("../data")
# val_data = load_train_csv("../data")


test_data = load_public_test_csv("../data")

print(f'type: {type(train_matrix)}, shape: {train_matrix.shape}')

N_1 = len(train_data["question_id"])
N_2 = len(train_data["user_id"])
N_3 = len(train_data["is_correct"])

print(f'N: {[N_1,N_2,N_3]}')

# Bootstrapped training data
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

boot_2 = bootstrap_dict(train_data, 3)
boot_3 = bootstrap_dict(train_data, 5)
boot_4 = bootstrap_dict(train_data, 1)

#
# def sparse_ify(data_dict):
#     n_Q = len(Counter(data_dict["question_id"]).keys())
#     n_U = len(Counter(data_dict["user_id"]).keys())
#     new_matrix = sp.sparse.dok_matrix((n_Q,n_U))#, dtype=np.int8)
#
#     for i in range(len(data_dict["question_id"])):
#         question_id = data_dict["question_id"][i]
#         user_id = data_dict["user_id"][i]
#         new_matrix[question_id,user_id] = 1
#
#     new_matrix = new_matrix.transpose().tocsr()
#     return new_matrix.toarray()
#
# matrix_1 = sparse_ify(boot_1)
# print(f'boot type: {type(matrix_1)}, shape: {matrix_1.shape}')
# matrix_2 = sparse_ify(boot_2)
# matrix_3 = sparse_ify(boot_3)




# Random Forest
LR = 0.05
N_iter = int(2e5)
K = 50

# def new_train_model(train_data, als_LR, latent_dimension, num_iteration, verbose=True):
#
#     # Get representation matrix with ALS.
#
#     if verbose:
#         print(f"Performing ALS with k = {latent_dimension}, lr = {als_LR}, {num_iteration} iterations.")
#
#     u = np.random.uniform(low=0, high=1 / np.sqrt(latent_dimension),
#                           size=(len(set(train_data["user_id"])), latent_dimension))
#     z = np.random.uniform(low=0, high=1 / np.sqrt(latent_dimension),
#                           size=(len(set(train_data["question_id"])), latent_dimension))
#     for i in tqdm(range(num_iteration)):
#         u, z = update_u_z(train_data, als_LR, u, z)
#
#     if verbose:
#         print(f"ALS Completed.")
#         print("Converting data into matrices")
#
#     train_matrix = []
#     for i, is_correct in tqdm(enumerate(train_data["is_correct"])):
#         user = train_data["user_id"][i]
#         question = train_data["question_id"][i]
#
#         train_matrix.append(np.r_[u[user], z[question], np.array([is_correct])])
#
#     train_matrix = np.stack(train_matrix)
#     X, y = train_matrix[:, :-1], train_matrix[:, -1]
#     print("Training RandomForestClassifier.")
#     RFC = RandomForestClassifier(n_estimators=120,verbose= 2 if verbose else 0)
#     RFC.fit(X, y.reshape(-1, ))
#
#     return (RFC, u, z)
#
#
# model = new_train_model(boot_4, LR, K, N_iter)
# prediction_4 = np.array(partb_randomforest.predict(model, val_data))
#
# acc_4 = final_evaluate(val_data, prediction_4)
#
# print(f'Prediction 4 size:{prediction_4.shape}')
# print(f'Prediction 4 Accuracy:{acc_4}')





# KNN: by User, optimal k: 11
k = 11
nbrs = KNNImputer(n_neighbors=k)
# We use NaN-Euclidean distance measure.
mat = nbrs.fit_transform(train_matrix)
prediction_1 = np.array(sparse_matrix_predictions(val_data, mat))

print(f'Prediction 1 size:{prediction_1.shape}')

acc_1 = sparse_matrix_evaluate(val_data, mat)
print(f'Prediction 1 Accuracy:{acc_1}')







# matrix factorization ALS
mk = 25
result, train_loss, val_loss = als(boot_2, mk, 0.05, 200000, val_data)

prediction_2 = np.array(sparse_matrix_predictions(val_data, result))
acc_2 = sparse_matrix_evaluate(val_data, result)

print(f'Prediction 2: ALS size:{prediction_2.shape}')
print(f'Prediction 2: ALS Accuracy:{acc_2}')




# SVD
sk = 8
svd_result = svd_reconstruct(train_matrix, sk)

prediction_svd = np.array(sparse_matrix_predictions(val_data, svd_result))
acc_svd = sparse_matrix_evaluate(val_data, result)

print(f'Prediction SVD size:{prediction_2.shape}')
print(f'Prediction SVD Accuracy:{acc_2}')




prediction_bagged = prediction_1 + prediction_2

partial_predicc = np.round(prediction_bagged/2)
acc_bagged_partial = final_evaluate(val_data, partial_predicc)
print(f'Bagged Accuracy:{acc_bagged_partial}')

print('/////////////////////////')





# IRT

def predict_irt(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
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

LR = 0.01
N_iter = 40
theta, beta, acc_3 = irt(boot_3, val_data, LR, N_iter)
prediction_3 = np.array(predict_irt(val_data, theta, beta))


print(f'Prediction 3 size:{prediction_3.shape}')
print(f'Prediction 3 Accuracy:{acc_3[-1]}')



# ALS + IRT
# prediction_bagged_avg = prediction_2 + prediction_3
# prediction_bagged_avg = np.round(prediction_bagged_avg / 2)


# Knn + ALS + IRT
# prediction_bagged_avg = prediction_1+ prediction_2+ prediction_3
# prediction_bagged_avg = np.round(prediction_bagged_avg / 3)

# SVD + ALS + IRT
prediction_bagged_avg = prediction_svd + prediction_2+ prediction_3
prediction_bagged_avg = np.round(prediction_bagged_avg / 3)



#ALL
# prediction_bagged_avg = prediction_1+ prediction_2+ prediction_3 + prediction_4
# prediction_bagged_avg = np.round(prediction_bagged_avg / 4)



acc_bagged_avg = final_evaluate(val_data, prediction_bagged_avg)
print(f'Bagged AVG Accuracy:{acc_bagged_avg}')





# coeffs = [0, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 7.5, 10]
# for c1 in coeffs:
#     for c2 in coeffs:
#         for c3 in coeffs:
#             prediction_bagged_search = (c1*prediction_1) + (c2*prediction_2) + (c3*prediction_3)
#             N = c1+c2+c3
#             prediction_bagged_search = np.round(prediction_bagged_search / N)
#
#             acc_bagged_search = final_evaluate(val_data, prediction_bagged_search)
#             print(f'Bagged Accuracy ({c1}*knn) + ({c2}*matrix_fac) + ({c3}*irt : {acc_bagged_search}) ')


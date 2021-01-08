from utils import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from tqdm.auto import tqdm

@np.vectorize
def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    log_lklihood = 0.
    diff_matrix = theta.reshape(-1, 1) - beta.reshape(1, -1)
    inside_matrix = np.log(1 + np.exp(diff_matrix))
    for i, is_correct in enumerate(data["is_correct"]):
        u = data["user_id"][i]
        q = data["question_id"][i]
        log_lklihood += (is_correct * diff_matrix[u][q]) - inside_matrix[u][q]

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood

def get_grad(data, theta, beta, type):

    diff_matrix = sigmoid(theta.reshape(-1, 1) - beta.reshape(1, -1))
    if type == "theta":
        grad = np.zeros(theta.shape)
        for i, is_correct in enumerate(data["is_correct"]):
            u = data["user_id"][i]
            q = data["question_id"][i]
            grad[u] += (is_correct - diff_matrix[u][q])

    else: # type == "beta"
        grad = np.zeros(beta.shape)
        for i, is_correct in enumerate(data["is_correct"]):
            u = data["user_id"][i]
            q = data["question_id"][i]
            grad[q] += (diff_matrix[u][q] - is_correct)

    return grad


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # We want to MAXIMIZE the log likelihood, so we're actually doing gradient ascent.

    theta_grad = get_grad(data, theta, beta, "theta")
    new_theta = theta + (lr * theta_grad)

    beta_grad = get_grad(data, new_theta, beta, "beta")
    new_beta = beta + (lr * beta_grad)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return new_theta, new_beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    # TODO: Test initialization with normal? With beta?
    theta = np.random.normal(0, 1, 542)
    beta = np.random.normal(0, 1, 1774)

    val_acc_lst = []
    train_neg_llds = []
    val_neg_llds = []


    for i in tqdm(range(iterations)):
        if i % 1 == 0:
            train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
            val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
            train_neg_llds.append(train_neg_lld)
            val_neg_llds.append(val_neg_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        # print("Train NLLK: {} \t Val NNLK: {} \t  Val Score: {}".format(train_neg_lld, val_neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)


    plt.plot(train_neg_llds, label="Training")
    plt.plot(val_neg_llds, label = "Validation")
    plt.title("Val and training NLLK"), plt.xlabel("iterations"), plt.ylabel("NLLK"), plt.legend()
    plt.show()

    print(train_neg_llds)
    print(val_neg_llds)
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
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
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    LR = 0.01
    N_iter = 40
    theta, beta, val_acc = irt(train_data, val_data, LR, N_iter)
    plt.plot(val_acc)
    plt.title("Validation Accuracy"), plt.xlabel("iter"), plt.ylabel("Acc")
    plt.show()
    print(val_acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)

    test_acc = evaluate(test_data, theta, beta)
    print(f"Validation accuracy: {val_acc[-1]}")
    print(f"Test accuracy: {test_acc}")

    question_ids = [574, 219, 241, 739, 12]
    theta_plot = np.sort(theta)
    for q in question_ids:
        responses = sigmoid(theta_plot - beta[q])
        plt.plot(theta_plot, responses, label = f"Question {q}")

    plt.title("Curves of 5 questions"), plt.xlabel("Theta"), plt.ylabel("p(Cij) = 1"), plt.legend()
    plt.show()



    # #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

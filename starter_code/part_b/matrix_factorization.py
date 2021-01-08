import numpy as np

def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z, lam=0.0):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    u[n] = (1 - lr * lam) * u[n] + lr * (c - np.dot(u[n], z[q])) * z[q]
    z[q] = (1 - lr * lam) * z[q] + lr * (c - np.dot(u[n], z[q])) * u[n]

    return u, z


def als(train_data, k, lr, num_iteration, lam, val_data, collect=False, verbose=True):

    """ Performs ALS algorithm. Return reconstructed matrix.
    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param lam: float
    :param val_data: validation data dictionary
    :param collect: bool for whether to collect training and validation losses
    :param verbose: controls verbosity of the algorithm.
    :return: 2D reconstructed Matrix, arrays of training and validation losses
    during training
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(542, k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(1774, k))

    if verbose:
        print("\nAlternating Least Squares. k={}, lr={}, lambda={}, iterations={}".format(
            k, lr, lam, num_iteration))

    training_loss = []
    val_loss = []
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z, lam)

        if collect and not i % 1000:
            # Compute training and validation loss
            training_loss.append(squared_error_loss(train_data, u, z))
            val_loss.append(squared_error_loss(val_data, u, z))

    if verbose:
        print("Final training loss: {}".format(squared_error_loss(train_data, u, z)))

    mat = u @ z.T
    return mat, training_loss, val_loss

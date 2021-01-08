import numpy as np
from tqdm.auto import tqdm
from utils import *
from matrix_factorization import update_u_z
from sklearn.ensemble import RandomForestClassifier


def train_model(train_data, als_LR, latent_dimension, num_iteration, verbose=True):

    # Get representation matrix with ALS.

    if verbose:
        print(f"Performing ALS with k = {latent_dimension}, lr = {als_LR}, {num_iteration} iterations.")

    u = np.random.uniform(low=0, high=1 / np.sqrt(latent_dimension),
                          size=(len(set(train_data["user_id"])), latent_dimension))
    z = np.random.uniform(low=0, high=1 / np.sqrt(latent_dimension),
                          size=(len(set(train_data["question_id"])), latent_dimension))
    for i in tqdm(range(num_iteration)):
        u, z = update_u_z(train_data, als_LR, u, z)

    if verbose:
        print(f"ALS Completed.")
        print("Converting data into matrices")

    train_matrix = []
    for i, is_correct in tqdm(enumerate(train_data["is_correct"])):
        user = train_data["user_id"][i]
        question = train_data["question_id"][i]

        train_matrix.append(np.r_[u[user], z[question], np.array([is_correct])])

    train_matrix = np.stack(train_matrix)
    X, y = train_matrix[:, :-1], train_matrix[:, -1]
    print("Training RandomForestClassifier.")
    RFC = RandomForestClassifier(verbose= 2 if verbose else 0)
    RFC.fit(X, y.reshape(-1, ))

    return (RFC, u, z)

def predict(model, data):

    RFC, u, z = model

    matrix = []
    for i, is_correct in tqdm(enumerate(data["is_correct"])):
        user = data["user_id"][i]
        question = data["question_id"][i]

        matrix.append(np.r_[u[user], z[question]])

    matrix = np.stack(matrix)

    return RFC.predict(matrix)

def main():
    LR = 0.05
    N_iter = int(2e5)
    K = 50

    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")

    model = train_model(train_data, LR, K, N_iter)
    print(predict(model, val_data))

if __name__ == "__main__":
    main()






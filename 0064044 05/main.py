import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_data():
    data = np.genfromtxt("hw04_data_set.csv", delimiter=",", skip_header=1)
    X = np.array(data[:, 0])
    Y = np.array(data[:, 1])
    train_x = np.array(X[0:100])
    train_y = np.array(Y[0:100])
    test_x = np.array(X[100:])
    test_y = np.array(Y[100:])
    N = train_x.shape[0]
    return train_x, train_y, test_x, test_y, N

def train(N):
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}

    # put all training instances into the root node
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True
    while True:
        split_nodes = [key for key, value in need_split.items() if value == True]
        if len(split_nodes) == 0:
            break
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            if len(data_indices) <= N:
                node_splits[split_node] = np.mean(y_train[data_indices])
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False
                unique_values = np.sort(np.unique(X_train[data_indices]))
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
                split_scores = np.repeat(0.0, len(split_positions))
                for s in range(len(split_positions)):
                    left_indices = data_indices[X_train[data_indices] <= split_positions[s]]
                    right_indices = data_indices[X_train[data_indices] > split_positions[s]]
                    split_scores[s] = 1/(left_indices.size+right_indices.size)*(
                        np.sum(np.square(y_train[left_indices] - np.mean(y_train[left_indices])))
                        + np.sum(np.square(y_train[right_indices] - np.mean(y_train[right_indices]))))
                best_scores = np.min(split_scores)
                best_splits = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_splits

                left_indices = data_indices[X_train[data_indices] <= best_splits]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                right_indices = data_indices[X_train[data_indices] > best_splits]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
    return node_splits, is_terminal

def predict(X, node_splits, is_terminal):
    N = X.size
    y_predicted = np.repeat(0, N)
    for i in range(N):
        index = 1
        while True:
            if is_terminal[index] == True:
                y_predicted[i] = node_splits[index]
                break
            else:
                if X[i] <= node_splits[index]:
                    index = index * 2
                else:
                    index = index * 2 + 1
    return y_predicted

def plot_data(node_splits, is_terminal):
    data_interval = np.linspace(0, 60, 1601)
    predicted = predict(data_interval, node_splits, is_terminal)

    plt.figure(figsize=(10, 6))
    plt.plot(X_train[:], y_train[:], "b.", markersize=10)
    plt.plot(test_x[:], test_y[:], "r.", markersize=10)
    plt.plot(data_interval, predicted, "k-")
    plt.title(" P = 15")
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.show()

def find_RMSE(real, pred):
    return np.sqrt(np.mean(np.square(real - pred)))

def plot_RMSE():
    RMSE = np.zeros((10,1))
    interval = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for index, i in enumerate(interval):
        node_splits, is_terminal = train(i)
        pred = predict(test_x, node_splits, is_terminal)
        RMSE[index] = find_RMSE(test_y,pred)

    plt.figure(figsize=(10, 6))
    plt.plot(interval, RMSE, "k-")
    plt.plot(interval, RMSE, "b.", markersize=20)
    plt.xlabel("Pre-pruning size (P)")
    plt.ylabel("RMSE")
    plt.show()

if __name__ == '__main__':
    X_train, y_train, test_x, test_y, N_train = parse_data()

    node_splits, is_terminal = train(15)
    plot_data(node_splits, is_terminal)
    RMSE = find_RMSE(test_y, predict(test_x, node_splits, is_terminal))
    print("RMSE is ", RMSE, "when P is 15")
    plot_RMSE()


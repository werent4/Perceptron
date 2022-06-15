import numpy as np
import pandas as pd

def sigmoid(w_sum):
    return 1 / (1+ np.exp(-w_sum))

def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x,w in zip(feature, weights):
        new_w = w + l_rate *(target- prediction)*x
        new_weights.append(new_w)
    return new_weights

def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate*(target-prediction)

bias = 0.5
l_rate = 0.01
epochs = 1000

X = np.array([[1, 1, 0],
                    [0, 1, 1],
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]])
y = np.array([[1, 0, 0, 1, 1, 0, 0]]).T

data = pd.DataFrame(X, columns=["x0", "x1", "x2"])
data["targets"] = y

print('Ours data')
print(data)


rg = np.random.default_rng()
#weights = rg.random((1, 3))[0]
weights = np.array([1, -1, 0])

print("our random weights")
print(weights)

trained_weights = []

def train_model(trained_weights, data, weights, bias, l_rate, epochs):
    for e in range(epochs):

        for i in range(len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            w_sum =  np.dot(feature, weights) + bias
            prediction = sigmoid(w_sum)

            weights = update_weights(weights, l_rate, target, prediction, feature)
            bias = update_bias(l_rate, bias, prediction, target)

        print('*' * 30)
        print('epoch:',e,'/',epochs - 1)
    print('our trained weights')
    trained_weights.extend(weights)
    print(trained_weights)
    print('weighted sum')
    print(w_sum)


def predict(new_data, trained_weights):
    return sigmoid(np.dot(new_data, trained_weights) + bias)

train_model(trained_weights, data, weights, bias, l_rate, epochs)

new_data = np.array([0,  0, 1])

print('*' * 30)
print('prediction')
print(predict(new_data, trained_weights))

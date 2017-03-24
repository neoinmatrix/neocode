# coding=utf-8
import numpy as np
from sklearn.datasets import load_iris

class Config:
    nn_input_dim = 4   # input layer dimensionality
    nn_output_dim = 3  # output layer dimensionality
    epsilon = 0.01     # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength

def loss(model, X, y):
    num_examples = len(X)  # training set size
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # print probs
    return np.argmax(probs, axis=1)

def fit_model(X, y, nn_hdim, num_passes=200, print_loss=False):
    num_examples = len(X)
    # np.random.seed(0)
    W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, Config.nn_output_dim))

    model = {}
    # Gradient descent
    for i in range(0, num_passes):
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += Config.reg_lambda * W2
        dW1 += Config.reg_lambda * W1
        # Gradient descent parameter update
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2
        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, loss(model, X, y)))
    return model

if __name__ == "__main__":
    # X = np.array([[0,0],
    #             [0,1],
    #             [1,0],
    #             [1,1] ])         
    # y = np.array([[0,0,1,1]]).T
    data = load_iris()
    features = data['data']
    labels = data['feature_names']
    target = data['target']

    index_train=np.random.randint(150,size=20)
    index_test=np.random.randint(150,size=20)
    X=features[index_train]
    y=target[index_train]
    X_t=features[index_test]
    y_t=target[index_test]

    model = fit_model(X, y, 4, print_loss=True)
    print predict(model,X_t)
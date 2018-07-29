import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder


def add_ones(X):
    """
    Add vector of ones (bias) at the left of the matrix X.

    :param X: input data
    :return: matrix with added ones
    """
    ones = np.ones(shape=(X.shape[0], 1))
    return np.hstack((ones, X))


def sigmoid_activation(X):
    """
    Sends data X through activation sigmoid function.

    :param X: input data
    :return: activated data
    """
    return expit(X)


def cross_validation(k=5):
    """
    Compare accuracy of my implementation with Logistic Regression and Gradient Boosting with cross validation.

    :param k: k used for k-fold cross validation
    :return: list of estimates F1 with cross validation on Iris data
    """
    X, y = load_iris(True)
    nn = cross_val_predict(NeuralNetwork([5, 3], 0.001), X, y, cv=k)
    lr = cross_val_predict(LogisticRegression(), X, y, cv=k)
    gb = cross_val_predict(GradientBoostingClassifier(), X, y, cv=k)
    nn_f1 = round(f1_score(y, nn, average='weighted'), 4)
    lr_f1 = round(f1_score(y, lr, average='weighted'), 4)
    gb_f1 = round(f1_score(y, gb, average='weighted'), 4)
    return {'MyNeuralNetwork': nn_f1, 'LogisticRegression': lr_f1, 'GradientBoostingClassifier': gb_f1}


class NeuralNetwork(MLPClassifier):
    """
    Class NeuralNetwork implements neural network with backpropagation using L-BFGS-B optimization for gradient descent.
    """

    def __init__(self, hidden_layer_sizes, alpha):
        """
        :param hidden_layer_sizes: list with numbers of neurons in hidden layers (without bias - constant activation)
        :param alpha: regularization constant
        """
        super().__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha

        self.coefs_ = None  # The ith element in the list represents the weight matrix corresponding to layer i.

        self.weights_per_layer = None
        self.total_weights = None
        self.regularization_mask = None
        self.indices2split = None
        self.number_of_unique_targets = None
        self.X = None
        self.y = None

    def set_data_(self, X, y):
        """
        Saves data for faster calculations (weights structure for layers, bitmask for regularization).

        :param X: input data
        :param y: target values
        :return: self
        """
        self.X = X
        self.y = y

        first_layer_size = X.shape[1] + 1   # +1 for bias
        self.number_of_unique_targets = np.unique(y).size

        self.weights_per_layer = [[first_layer_size, self.hidden_layer_sizes[0]]]   # first layer
        for i in range(len(self.hidden_layer_sizes) - 1):   # hidden layers
            n_layers = self.hidden_layer_sizes[i]
            n_1_layers = self.hidden_layer_sizes[i+1]
            self.weights_per_layer.append([n_layers + 1, n_1_layers])
        self.weights_per_layer.append([self.hidden_layer_sizes[-1] + 1, self.number_of_unique_targets])   # last layer

        self.weights_per_layer = list(map(lambda x: [x[0], x[1], x[0] * x[1]], self.weights_per_layer))
        curr_index = 0
        for i in range(len(self.weights_per_layer)):
            curr_index += self.weights_per_layer[i][2]
            self.weights_per_layer[i][2] = curr_index

        self.total_weights = self.weights_per_layer[-1][2]
        self.regularization_mask = np.ones(self.total_weights)
        for i in range(self.weights_per_layer[0][1]):     # first layer
            self.regularization_mask[i] = 0
        for i in range(1, len(self.hidden_layer_sizes)):    # hidden layers
            n_layers = self.hidden_layer_sizes[i]
            for j in range(n_layers):
                self.regularization_mask[self.weights_per_layer[i - 1][2] + j] = 0
        for j in range(self.number_of_unique_targets):    # last layer
            self.regularization_mask[self.weights_per_layer[-2][2] + j] = 0

        self.indices2split = list(map(lambda x: x[2], self.weights_per_layer))[:-1]

        return self

    def init_weights_(self):
        """
        Returns random vector of weights (activation unit 1 added to each layer except the last).

        :return: vector of normally distributed weights
        """
        np.random.seed(777)
        return np.random.randn(self.total_weights) * 0.1

    def flatten_coefs(self, coefs):
        """
        Flattens coefficients from list of matrices to 1D vector.

        :param coefs: list of matrices of weights
        :return: vector of weights
        """
        for i in range(len(coefs)):
            coefs[i] = coefs[i].flatten()
        return np.concatenate(coefs)

    def unflatten_coefs(self, coefs):
        """
        Inverse procedure than in function flatten_coefs.

        :param coefs: flattened vector of weights
        :return: non flattened list of matrices of weights
        """
        subarrays = np.split(coefs, self.indices2split)
        for i in range(len(subarrays)):
            subarrays[i] = subarrays[i].reshape(self.weights_per_layer[i][0], self.weights_per_layer[i][1])
        return subarrays

    def fit(self, X, y):
        """
        Fit the model to data matrix X and target(s) y.
        Saves vector of weights to coefs_.

        :param X: input data
        :param y: target values (class labels in classification, real numbers in regression)
        :return: self (trained model)
        """
        self.set_data_(X, y)
        coefs = self.init_weights_()
        self.coefs_ = self.unflatten_coefs(coefs)

        coefs, _, d = fmin_l_bfgs_b(self.cost, coefs, fprime=self.grad)
        self.coefs_ = self.unflatten_coefs(coefs)
        return self

    def predict(self, X):
        """
        Predict using the multi-layer perceptron classifier.

        :param X: input data
        :return: array of the predicted classes
        """
        output, _ = self.feedforward(X, self.coefs_)
        return np.argmax(output, axis=1)

    def predict_proba(self, X):
        """
        Probability estimates.

        :param X: input data
        :return: predicted probability of the sample for each class in the model
        """
        output, _ = self.feedforward(X, self.coefs_)
        return np.apply_along_axis(lambda r: r / sum(r), 1, output)

    def cost(self, coefs):
        """
        Calculates the value of the cost function.

        :param coefs: flattened vector of weights
        :return: value of the cost function
        """
        self.coefs_ = self.unflatten_coefs(coefs)
        last_activations, _ = self.feedforward(self.X, self.coefs_)

        J = (1 / (2 * len(self.y))) * np.sum((last_activations - self.one_hot_encoding_vector(self.y)) ** 2)
        regularization = (self.alpha / 2) * (coefs ** 2).dot(self.regularization_mask)
        cost = J + regularization
        return cost

    def grad(self, coefs):
        """
        Calculates gradients of the weights.

        :param coefs: flattened vector of weights
        :return: vector of weights' gradients
        """
        coefs = self.unflatten_coefs(coefs)
        _, activations = self.feedforward(self.X, coefs)

        gradients = []

        L = len(activations) - 1
        A_L = activations[L]
        d_L = (A_L - self.one_hot_encoding_vector(self.y)) * A_L * (1 - A_L)
        D_L = (1 / len(self.y)) * np.dot(activations[L - 1].T, d_L)
        gradients.append(D_L)

        l = L - 1
        d_l = d_L[:]
        while l > 0:
            d_l = np.dot(d_l, coefs[l].T) * activations[l] * (1 - activations[l])
            d_l = d_l[:, 1:]  # remove first column to match dimensions
            D_l = (1 / len(self.y)) * np.dot(activations[l - 1].T, d_l)
            gradients.append(D_l)
            l -= 1

        gradient = self.flatten_coefs(list(reversed(gradients)))
        regularization_gradient = self.alpha * self.flatten_coefs(coefs) * self.regularization_mask
        return gradient + regularization_gradient

    def grad_approx(self, coefs, e):
        """
        Calculates numeric approximation of the gradient vector.

        :param coefs: flattened vector of weights
        :param e: small numeric value
        :return: numeric approximation of the gradient vector
        """
        gradient = np.zeros(coefs.shape[0])
        for i in range(coefs.shape[0]):
            old_coef = coefs[i]
            coefs[i] += e
            cost_add = self.cost(coefs)
            coefs[i] -= 2 * e
            cost_subtract = self.cost(coefs)
            gradient[i] = (cost_add - cost_subtract) / (2 * e)
            coefs[i] = old_coef
        return gradient

    def feedforward(self, X, coefs):
        """
        Performs feedforward in the neural network.

        :param X: input data
        :param coefs: list of matrices of weights
        :return: activations at the output layer, activations on all layers
        """
        A = X
        activations = []
        for w_matrix in coefs:
            A = add_ones(A)
            activations.append(A)
            Z = A @ w_matrix
            A = sigmoid_activation(Z)
        activations.append(A)
        return A, activations

    def one_hot_encoding_vector(self, vector):
        """
        One Hot Encoding of a vector.

        :param vector: vector of target variables
        :return: encoded matrix
        """
        enc = OneHotEncoder(sparse=False, n_values=self.number_of_unique_targets)
        matrix = enc.fit_transform([vector]).reshape((len(vector), enc.n_values))
        return matrix


if __name__ == '__main__':
    print(cross_validation())

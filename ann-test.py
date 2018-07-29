import unittest
import numpy as np

from ann import NeuralNetwork
from sklearn import datasets, utils, metrics


class NeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        iris = datasets.load_iris()
        self.X, self.y = iris.data, iris.target

    def test_set_data(self):
        ann = NeuralNetwork([5, 3], alpha=1e-5)
        ann.set_data_(self.X, self.y)
        coefs = ann.init_weights_()
        self.assertEqual(len(coefs), 55)

    def test_weighs_structure(self):
        ann = NeuralNetwork([5, 3], alpha=1e-5)
        ann.set_data_(self.X, self.y)
        coefs = ann.unflatten_coefs(ann.init_weights_())
        shapes = np.array([coef.shape for coef in coefs])
        np.testing.assert_array_equal(shapes, np.array([[5, 5], [6, 3], [4, 3]]))

    def test_gradient_computation(self):
        ann = NeuralNetwork([2, 2], alpha=1e-5)
        ann.set_data_(self.X, self.y)
        coefs = ann.init_weights_()
        g1 = ann.grad_approx(coefs, e=1e-5)
        g2 = ann.grad(coefs)
        np.testing.assert_array_almost_equal(g1, g2, decimal=10)

    def test_fit_and_predict(self):
        ann = NeuralNetwork([4, 2], alpha=1e-5)
        ann.fit(self.X, self.y)
        T = self.X[[10, 60, 110]]
        predictions = ann.predict(T)
        np.testing.assert_array_equal(predictions, np.array([0, 1, 2]))

    def test_predict_probabilities(self):
        ann = NeuralNetwork([4, 2], alpha=1e-5)
        ann.fit(self.X, self.y)
        T = self.X[[15, 65, 115, 117]]
        ps = ann.predict_proba(T)
        margin = np.min(np.max(ps, axis=1))
        self.assertGreater(margin, 0.90)

    def test_on_digits(self):
        data_full = datasets.load_digits()
        data, resp = utils.shuffle(data_full.data, data_full.target)
        m = data.shape[0]
        X, y = data[:m // 2], resp[:m // 2]
        X_test, y_test = data[m // 2:], resp[m // 2:]

        ann = NeuralNetwork([20, 5], alpha=1e-5)
        ann.fit(X, y)
        y_hat = ann.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_hat)
        self.assertGreater(acc, 0.85)

    def test_with_crossvalidation(self):
        from sklearn.model_selection import cross_validate

        clf = NeuralNetwork([10, 2], alpha=1e-5)
        scores = cross_validate(clf, self.X, self.y, scoring='accuracy', cv=5)
        acc = np.sum(scores["test_score"]) / 5
        self.assertGreater(acc, 0.94)


if __name__ == "__main__":
    unittest.main(verbosity=2)

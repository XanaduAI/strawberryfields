r"""
Submodule for embedding trainable GBS parameters.

Training algorithms that rely on the WAW parametrization, where W is a diagonal matrix of weights
and A is a symmetric matrix, proceed by updating the weights using gradient descent. This process
is facilitated when trainable parameters are embedded in the weights, i.e., when the weights
are functions of the parameters. This ensures that the updated weights lead to valid matrices and
provides additional flexibility in training strategies.

This submodule contains methods to implement these embeddings and provides derivatives of the
weights with respect to the trainable parameters."""


class Exp:

    def weights(self, params):
        return np.exp(-params)

    def derivative(self, params):
        return -1*self.weights(params)


class ExpFeatures(features):

    def __init__(self):
        self.features = features

    def weights(self, params):
        m, d = np.shape(params)
        for i in range(m):
            w[i] = np.exp(-np.dot(self.features[i], params[i]))
        return w

    def derivative(self, params):
        m, d = np.shape(params)
        w = self.weights(self, params)
        dw = np.zeros(m, d)
        for i in range(m):
            for j in range(d):
                dw[i, j] = -self.features[i, j]*w[i]
        return dw




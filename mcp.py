import numpy as np


class MCP(object):
    """A Multi-Class Perceptron class.
    """

    def __init__(self, num_inputs=3, num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs
        Args:
            num_inputs (int): Number of inputs
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # create zero weights for the layer connections
        self.weights = np.zeros(shape=(num_outputs, num_inputs))

    def _sigmoid(self, x):
        """Sigmoid activation function
        Parameters:
            x : Value to be processed
        Returns:
            y : Output after processing
        """
        # Basic sigmoid function
        y = 1.0 / (1 + np.exp(-x))
        return y

    def train(self, data, labels):
        """Perceptron Training Function
        Parameters:
            data : the training dataset
            labels : the correct label for each training instance
        """
        epochs = 10
        for i in range(epochs):
            # Get the number of total training instances
            instances = data.shape[0]
            error = 0
            for i in range(instances):

                # obtain the data and label of the current instance
                x = data[i]
                y = np.argmax(labels[i])

                # Make a prediction based on current weight vector using sigmoid function
                yi = np.argmax(self._sigmoid(np.dot(self.weights, x)))

                # Update weight vectors if the prediction is incorrect
                if y != yi:
                    self.weights[y] = self.weights[y] + x
                    self.weights[yi] = self.weights[yi] - x

    def predict(self, x):
        """Perceptron Prediction Function
        Parameters:
            x : The data instance to classify
        Returns:
            y : The predicted class of data instance
        """

        # Use the sigmoid function to make prediction based on weight vectors
        y = np.argmax(self._sigmoid(np.dot(self.weights, x)))
        return y

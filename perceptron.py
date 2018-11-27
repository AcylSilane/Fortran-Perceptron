#!/usr/bin/env python

import numpy as np


class Neuron(object):
    def __init__(self, weights, bias):
        """
        A Neuron in the network. Takes in a list of weights and its bias

        :param weights: An array containing all the weights
        :type weights: np.ndarray
        :param bias: The bias for the neuron
        :type bias: float
        """
        self.bias = bias
        self.weights = weights

    def percept(self, input):
        """
        Given an input array, returns a 1 if the neuron was activated and 0 otherwise.

        :param input: The input array.
        :type input: np.ndarray
        :return: An integer containing the neuron's output.
        """
        potential = 0
        for count,value in enumerate(input):
            potential *= self.weights[count]
        potential += self.bias
        if potential >= 0:
            return 1
        else:
            return 0

class Perceptron(object):
    def __init__(self, input_length, output_length, weightfile = "weights.csv", biasfile = "bias.csv"):
        """
        A collection of neuron objects giving a collection of responses
        :param input_length: How many values will be in the input vector
        :type input_length: int
        :param output_length: How many neurons are in the perceptron. The number of classes it can predict.
        :type output_length: int
        """
        # Read in input weights
        self.weightfile = weightfile
        self.weights = np.zeros((output_length, input_length))
        self.read_weightfile()
        # Read in input biases
        self.biasfile = biasfile
        self.biases = np.zeros(output_length)
        self.read_biasfile()
        # Save the length of the input and output vectors
        self.input_length = input_length
        self.output_length = output_length
        # Populate with neurons
        self.neurons = [None] * self.output_length
        for count, neuron in enumerate(self.neurons):
            self.neurons[count] = Neuron(self.weights[count], self.biases[count])

    def read_weightfile(self):
        with open(self.weightfile, "r") as inp:
            for row,line in enumerate(inp):
                line = line.split(",")
                values = list(map(float, map(lambda x: x.strip(), line)))
                for col, weight in enumerate(values):
                    self.weights[row, col] = weight

    def read_biasfile(self):
        with open(self.biasfile, "r") as inp:
            for row, line in enumerate(inp):
                line = line.split(",")
                values = list(map(float, map(lambda x: x.strip(), line)))
                for count, value in enumerate(values):
                    self.biases[count] = value

    def percieve(self, input_vector):
        responses = [0] * self.output_length
        assert(len(input_vector) == self.input_length)
        for count, neuron in enumerate(self.neurons):
            responses[count] = neuron.percept(input_vector)
        responded = False
        for count, response in enumerate(responses):
            if response:
                responded = True
                print("I recognize %d", count)
        if not responded:
            print("I do not recognize this.")

if __name__ == "__main__":
    x = Perceptron(784, 10)
    test = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



from typing import Callable
import numpy as np
import gzip
import pickle
import random

LEARNING_RATE = 0.001
EXPECTED_RESULT = 0.95
CONTINOUS_SUCCESS = 10


class Network:
    weights: np.array
    biases: np.array

    def __init__(self, weights: np.array, biases: np.array):
        self.weights = weights
        self.biases = biases


class IterationResult:
    expectedOutcome: int
    actualOutcome: int
    activation: np.array
    layerErrors: np.array
    biasesErrors: np.array

    def __init__(self, activation: np.array, layerErrors: np.array, biasesErrors: np.array, expectedOutcome: int, actualOutcome: int):
        self.activation = activation
        self.layerErrors = layerErrors
        self.biasesErrors = biasesErrors
        self.expectedOutcome = expectedOutcome
        self.actualOutcome = actualOutcome


class TrainingData:
    picture: np.array     # A matrix with a single line representing the pixels of the image
    expected_output: int

    def __init__(self, data: np.array, expected_output: int):
        self.picture = data
        self.expected_output = expected_output


class DataSets:
    train_set: list[TrainingData]
    valid_set: list[TrainingData]
    test_set: list[TrainingData]

    def __init__(self, train_set: list[TrainingData], valid_set: list[TrainingData], test_set: list[TrainingData]):
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set


def map_data_to_training(raw_data_set) -> list[TrainingData]:
    zipped = list(zip(raw_data_set[0], raw_data_set[1]))
    return list(map(lambda raw_data: TrainingData(raw_data[0], raw_data[1]), zipped))


def get_data_sets() -> DataSets:
    with gzip.open('mnist.pkl.gz', 'rb') as fd:
        train_set, valid_set, test_set = pickle.load(fd, encoding='latin')
        return DataSets(
            map_data_to_training(train_set),
            map_data_to_training(valid_set),
            map_data_to_training(test_set)
        )


def get_network() -> Network:
    weights = np.array([
        np.random.randn(784, 100) / np.sqrt(784),
        np.random.randn(100, 10) / np.sqrt(100)
    ])

    biases = np.array([
        np.random.randn(100),
        np.random.randn(10)
    ])

    return Network(
        weights,
        biases
    )


def sigmoid(array: np.array) -> np.array:
    return 1 / (1 + np.exp(-array))


def iterationOutput(results: np.array) -> int:
    maxim = results[0]
    index = 0

    for i in range(0, len(results)):
        r = results[i]
        if r > maxim:
            maxim = r
            index = i

    return index


def sigm_der(layer: np.array) -> np.array:
    return layer * (-layer + 1)


def errors_on_last_layer(lastLayer: np.array, expectedOutput: int) -> np.array:
    aux = np.array([0 for index in range(0, len(lastLayer))])
    aux[expectedOutput] = 1

    return lastLayer - aux


def errors_on_layer(layer: np.array, next_layer_errors: np.array, network: Network, current_layer_index: int) -> np.array:
    return sigm_der(layer) * network.weights[current_layer_index + 1].dot(next_layer_errors)


def iteration(image: TrainingData, network: Network, onIterationFinished: Callable[[Network, IterationResult], None], learn=True):
    second_layer = sigmoid(image.picture.dot(network.weights[0]) + network.biases[0])
    last_layer = sigmoid(second_layer.dot(network.weights[1]) + network.biases[1])

    output = iterationOutput(last_layer)

    if learn:
        err_last = errors_on_last_layer(last_layer, image.expected_output)
        err_second = errors_on_layer(second_layer, err_last, network, 0)

        reshaped_err_second = image.picture.reshape(-1, 1).dot(err_second.reshape(1, -1))
        reshaped_err_last = second_layer.reshape(-1, 1).dot(err_last.reshape(1, -1))

        err_biases = np.array([err_second, err_last])

        network.weights[0] = network.weights[0] - LEARNING_RATE * reshaped_err_second
        network.weights[1] = network.weights[1] - LEARNING_RATE * reshaped_err_last

        network.biases = network.biases - LEARNING_RATE * err_biases

        iterationResult = IterationResult(
            activation=np.array([image.picture, second_layer, last_layer]),
            layerErrors=np.array([reshaped_err_second, reshaped_err_last]),
            biasesErrors=err_biases,
            expectedOutcome=image.expected_output,
            actualOutcome=output
        )
        onIterationFinished(network, iterationResult)

    return output == image.expected_output


def train(data_set: list[TrainingData], network: Network, onIterationFinished, learn=True) -> float:
    success = 0

    for image in data_set:
        if iteration(image, network, onIterationFinished, learn=learn):
            success += 1

    return success / float(len(data_set))


def test(test_set, network: Network, onIterationFinished):
    return train(test_set, network, onIterationFinished, learn=False)


def print_network(network: Network):
    pass
    # with open("result.txt", "a") as fd:
    #     fd.write(network.weights)
    #     fd.write(network.biases)


def start(configuration, onEpochFinished, onIterationFinished):
    global LEARNING_RATE

    data_sets = get_data_sets()
    perceptrons = get_network()

    contionous_success = 0

    LEARNING_RATE = configuration.learningRate

    for epochIndex in range(configuration.epochCount):
        data_sets.train_set.sort(key=lambda x: random.random())

        result = train(data_sets.train_set, perceptrons, onIterationFinished)

        print(result)
        onEpochFinished(epochIndex)

        if result > configuration.expectedResult:
            if contionous_success < configuration.continousSuccess - 1:
                contionous_success += 1
            else:
                if test(data_sets.test_set, perceptrons, onIterationFinished) > configuration.expectedResult:
                    print("Neuronal network hit its target")
                    print_network(perceptrons)
                    exit(0)
                else:
                    contionous_success = 0
        else:
            contionous_success = 0

import numpy as np

import Network
import pygame

NEURON_COUNT = []
WINDOW_WIDTH = 0
WINDOW_HEIGHT = 0
LAYER_DISTANCES = 0
NEURON_DISTANCE_PER_LAYER = []
WINDOW_PADDING = {
    "N": 10,
    "E": 10,
    "S": 10,
    "W": 10
}
NEURON_POSITIONS = {}
NEURON_RADIUS_PER_LAYER = []
SCREEN: pygame.surface.Surface
AXON_ACTIVATION_COLOR = pygame.color.Color(255, 255, 102)
NEURON_ACTIVATION_COLOR = pygame.color.Color(173, 255, 47)
BACKGROUND_COLOR = pygame.Color(255, 255, 255)


class Position:
    x: int
    y: int

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def vector(self):
        return pygame.Vector2(self.x, self.y)


class Neuron:
    layer: int
    index: int

    def __init__(self, layer: int, index: int):
        self.layer = layer
        self.index = index


class Configuration:
    neurons: list
    learningRate: float
    epochCount: int
    expectedResult: float
    continousSuccess: int

    def __init__(self, neuronCounts: list, learningRate: float, epochCount: int, expectedResult: float, continousSuccess: int):
        self.neurons = neuronCounts
        self.learningRate = learningRate
        self.epochCount = epochCount
        self.expectedResult = expectedResult
        self.continousSuccess = continousSuccess


def activatedColor(color: pygame.Color, activated: float) -> pygame.color.Color:
    return pygame.color.Color(
        int(color.r * activated),
        int(color.g * activated),
        int(color.b * activated),
        int(color.a * activated)
    )


def calculateNeuronPosition(neuron: Neuron) -> Position:
    return Position(
        x=WINDOW_PADDING["W"] + LAYER_DISTANCES * neuron.layer - NEURON_RADIUS_PER_LAYER[neuron.layer],
        y=WINDOW_PADDING["N"] + neuron.index * NEURON_DISTANCE_PER_LAYER[neuron.layer] + NEURON_RADIUS_PER_LAYER[neuron.layer]
    )


def initNeuronPositions(neuronsInLayers: list):
    for layerIndex in range(len(neuronsInLayers)):
        NEURON_POSITIONS[layerIndex] = {}
        for neuronIndex in range(neuronsInLayers[layerIndex]):
            NEURON_POSITIONS[layerIndex][neuronIndex] = calculateNeuronPosition(Neuron(layerIndex, neuronIndex))


def neuronPosition(neuron: Neuron) -> Position:
    return NEURON_POSITIONS[neuron.layer][neuron.index]


def drawNeuron(neuron: Neuron, color: pygame.color.Color):
    position = neuronPosition(neuron)
    pygame.draw.circle(SCREEN, color, position.vector(), NEURON_RADIUS_PER_LAYER[neuron.layer])


def drawLine(n1: Neuron, n2: Neuron, color: pygame.color.Color):
    n1Position = neuronPosition(n1)
    n2Position = neuronPosition(n2)
    pygame.draw.line(SCREEN, color, n1Position.vector(), n2Position.vector())


def drawActivatedNeuron(neuron: Neuron, activation: float):
    color = activatedColor(NEURON_ACTIVATION_COLOR, activation)
    drawNeuron(neuron, color)


def minMaxNeuronActivation(activations: np.array):
    minim = 0
    maxim = 0
    for layerIndex in range(len(activations)):
        layer = activations[layerIndex]
        for neuronIndex in range(len(layer)):
            neuronActivation = layer[neuronIndex]
            if maxim < neuronActivation:
                maxim = neuronActivation
            if minim > neuronActivation:
                minim = neuronActivation

    return minim, maxim


def minMaxAxonActivation(activations: np.array):
    minim = 0
    maxim = 0
    for layerIndex in range(len(NEURON_COUNT) - 1):
        for currentLayerNeuronIndex in range(NEURON_COUNT[layerIndex]):
            for nextLayerNeuronIndex in range(NEURON_COUNT[layerIndex + 1]):
                axonActivation = activations[layerIndex][currentLayerNeuronIndex][nextLayerNeuronIndex]
                if maxim < axonActivation:
                    maxim = axonActivation
                if minim > axonActivation:
                    minim = axonActivation

    return minim, maxim


def normalizeActivation(activation: float, minim: float, maxim: float) -> float:
    if minim == activation:
        return 0
    return (minim - activation) / (minim - maxim)


def drawNeuronActivations(activation: np.array):
    minim, maxim = minMaxNeuronActivation(activation)
    for layerIndex in range(len(activation)):
        layer = activation[layerIndex]
        for neuronIndex in range(len(layer)):
            neuronActivation = layer[neuronIndex]
            normalizedNeuronActivation = normalizeActivation(neuronActivation, minim, maxim)

            drawActivatedNeuron(Neuron(layerIndex, neuronIndex), normalizedNeuronActivation)


def clearDisplay():
    SCREEN.fill(BACKGROUND_COLOR)


def showDisplay():
    pygame.display.update()


def drawActivatedAxon(n1: Neuron, n2: Neuron, activation: float):
    color = activatedColor(AXON_ACTIVATION_COLOR, activation)
    drawLine(n1, n2, color)


def drawAxonActivations(axonActivations: np.array):
    minim, maxim = minMaxAxonActivation(axonActivations)
    for layerIndex in range(len(NEURON_COUNT) - 1):
        for currentLayerNeuronIndex in range(NEURON_COUNT[layerIndex]):
            for nextLayerNeuronIndex in range(NEURON_COUNT[layerIndex + 1]):
                activation = axonActivations[layerIndex][currentLayerNeuronIndex][nextLayerNeuronIndex]
                normalizedActivation = normalizeActivation(activation, minim, maxim)

                n1 = Neuron(layerIndex, currentLayerNeuronIndex)
                n2 = Neuron(layerIndex + 1, nextLayerNeuronIndex)
                drawActivatedAxon(n1, n2, normalizedActivation)


epochFinished = True


def onIterationFinished(network: Network.Network, results: Network.IterationResult):
    global epochFinished

    for _ in pygame.event.get():
        pass

    if epochFinished:
        epochFinished = False
        print(f"Printing iteration\nExpected outcome:{results.expectedOutcome}\nActual outcome:{results.actualOutcome}")
        clearDisplay()
        drawNeuronActivations(results.activation)
        drawAxonActivations(network.weights)
        showDisplay()


def onEpochFinished(epochIndex: int):
    global epochFinished
    print(f"Epoch {epochIndex} finished")
    epochFinished = True


def initScreen():
    global WINDOW_HEIGHT, WINDOW_WIDTH, SCREEN

    screenInfo = pygame.display.Info()

    WINDOW_WIDTH = screenInfo.current_w - 100
    WINDOW_HEIGHT = screenInfo.current_h - 100

    pygame.display.set_caption("VISUAL NEURONAL")
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    WINDOW_HEIGHT = SCREEN.get_height()
    WINDOW_WIDTH = SCREEN.get_width()


def initDistances(config: Configuration):
    global LAYER_DISTANCES, NEURON_DISTANCE_PER_LAYER, NEURON_RADIUS_PER_LAYER
    numberOfLayers = len(config.neurons)

    LAYER_DISTANCES = (WINDOW_WIDTH - WINDOW_PADDING["E"] - WINDOW_PADDING["W"]) / (numberOfLayers - 1)

    innerHeight = WINDOW_HEIGHT - WINDOW_PADDING["N"] - WINDOW_PADDING["S"]

    for layerIndex in range(numberOfLayers):
        NEURON_DISTANCE_PER_LAYER.insert(layerIndex, innerHeight / config.neurons[layerIndex])

    for layerIndex in range(numberOfLayers):
        radius = innerHeight / config.neurons[layerIndex] / 2 - 1
        if radius <= 0:
            radius = 1
        NEURON_RADIUS_PER_LAYER.insert(layerIndex, radius)


def initNeurons(neurons):
    global NEURON_COUNT
    NEURON_COUNT = neurons
    initNeuronPositions(neurons)


def initPygame():
    pygame.init()


def init(config: Configuration):
    initPygame()
    initScreen()
    initDistances(config)
    initNeurons(config.neurons)


def typedInput(message: str, conversion):
    while True:
        try:
            f = input(message)
            return conversion(f)
        except ValueError:
            pass


def getConfiguration() -> Configuration:
    return Configuration(
        neuronCounts=[784, 100, 10],
        learningRate=typedInput("Enter learning rate: ", float),
        epochCount=typedInput("Enter epoch count: ", int),
        expectedResult=typedInput("Enter expected learning hit rate: ", float),
        continousSuccess=typedInput("Enter continous expected winning rate: ", int)
    )


if __name__ == "__main__":
    configuration = getConfiguration()

    init(configuration)

    Network.start(configuration, onEpochFinished, onIterationFinished)

from core import Flatten, FC, Conv, ExitLoss, Network, Layer, ProbaExit
from data import load_mnist_data, SaveHandler
from graphics import view_numbers, regression
import numpy as np
from numpy.typing import NDArray
import math
import random

def number_recognition() -> None:
    layers = [
        Conv(32, 3, 2),
        Conv(64, 3, 2),
        Conv(128, 3, 2),
        Flatten(),
        FC(10)
    ]
    network = Network(layers=layers, exit_loss=ProbaExit(), input_shape=(1, 28, 28), lr=0.005)

    save_handler = SaveHandler()
    name = "number_recognition"
    if save_handler.has(name):
        network = save_handler.load(name)

    data = load_mnist_data()
    network.train(data=data, batch=1)
    save_handler.save(network, name)

    view_numbers(network)


def learn_shape() -> None:
    def curve(x: float, y: float) -> float:
        circle = math.sqrt(0.5 * (x - 0.5) ** 2 + (y - 0.5) ** 2)
        donut = math.sin(7.5 * circle)
        return circle
    
    def get_data(dim: int) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        data = []
        for _ in range(dim):
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            data.append((np.array([[x], [y]]), np.array([[curve(x, y)]])))
        return data

    layers: list[Layer] = [
        FC(40),
        FC(40),
        FC(40),
        FC(40),
        FC(40),
        FC(40),
        FC(1)
    ]
    network = Network(layers=layers, exit_loss=ExitLoss(), input_shape=(2, 1), lr=0.0001)

    save_handler = SaveHandler()
    name = "reproduce_shape"
    if save_handler.has(name):
        network = save_handler.load(name)

    data = get_data(1000)
    network.train(data=data, batch=1000)
    save_handler.save(network, name)
    regression(network, curve)


def main() -> None:
    learn_shape()


if __name__ == "__main__":
    main()

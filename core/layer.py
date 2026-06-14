import numpy as np
from numpy.typing import NDArray


class Layer:
    def __init__(self: Layer) -> None:
        self.lr: float = 0.0
        self.input: NDArray[np.float64] | None = None
        self.input_shape: tuple = ()

    def set_lr(self: Layer, lr: float):
        self.lr = lr

    def set_input_shape(self: Layer, input_shape: tuple) -> tuple:
        self.input_shape = input_shape
        return input_shape

    def feed_forward(self: Layer, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry

    def compute(self: Layer, entry: NDArray[np.float64], memorize) -> NDArray[np.float64]:
        if entry.shape != self.input_shape:
            print(entry.shape)
            raise ValueError
        if memorize:
            self.input = entry
        return self.feed_forward(entry)

    def descend_gradient(self: Layer, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        return gradient

import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class Biais(Layer):
    def __init__(self: Biais) -> None:
        self.B = np.array([[]])

    def set_input_shape(self: Biais, input_shape: tuple[int, int]) -> tuple[int, int]:
        self.input_shape = input_shape
        self.B = np.random.normal(0, np.sqrt(2 / input_shape[0]), size=(input_shape[0], 1))  # He
        return (input_shape[0], 1)

    def feed_forward(self: Biais, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry + self.B

    def descend_gradient(self: Biais, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        self.B -= self.lr * gradient
        return gradient


class ConvBiais(Layer):
    def __init__(self: ConvBiais) -> None:
        self.B = np.array([[]])

    def set_input_shape(self: ConvBiais, input_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        self.input_shape = input_shape
        self.B = np.random.normal(0, np.sqrt(2 / input_shape[0]), size=(input_shape[0], 1))  # He
        return input_shape

    def feed_forward(self: ConvBiais, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry + np.expand_dims(self.B, axis=2)

    def descend_gradient(self: ConvBiais, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        for i in range(self.B.shape[0]):
            self.B[i, 0] -= self.lr * np.sum(gradient[i, :, :])
        return gradient

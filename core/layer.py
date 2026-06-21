import numpy as np
from numpy.typing import NDArray


class Layer:
    def __init__(self: Layer) -> None:
        self.lr: float = 0.0
        self.input: NDArray[np.float64] | None = None
        self.input_shape: tuple = ()

    def set_lr(self: Layer, lr: float) -> None:
        self.lr = lr

    def set_input_shape(self: Layer, input_shape: tuple) -> tuple:
        self.input_shape = input_shape
        return input_shape

    def feed_forward(self: Layer, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry

    def compute(self: Layer, entry: NDArray[np.float64], memorize) -> NDArray[np.float64]:
        if entry.shape != self.input_shape:
            print(entry.shape, self.input_shape)
            raise ValueError
        if memorize:
            self.input = entry
        return self.feed_forward(entry)

    def descend_gradient(self: Layer, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        return gradient

    def get_data(self: Layer) -> tuple[list[int], list[float]]:
        int_list = list(self.input_shape)
        float_list = [self.lr]
        return int_list, float_list

    def load_from_data(self: Layer, int_list: list[int], float_list: list[float]) -> None:
        self.input_shape = tuple(int_list)
        self.lr = float_list[0]

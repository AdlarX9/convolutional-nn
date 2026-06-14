import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class FC(Layer):
    def __init__(self: FC, n: int) -> None:
        self.n: int = n
        self.p: int = 0
        self.W = np.array([[]])

    def set_input_shape(self: FC, input_shape: tuple[int, int]) -> tuple[int, int]:
        self.input_shape = input_shape
        p, _ = input_shape
        self.p = p
        self.W = np.random.normal(0, np.sqrt(2 / self.n), size=(self.n, p))  # He
        return (self.n, 1)

    def feed_forward(self: FC, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.W @ entry

    def descend_gradient(self: FC, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        new_gradient = self.W.T @ gradient
        self.W -= self.lr * (gradient @ self.input.T)
        return new_gradient

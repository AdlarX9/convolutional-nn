import numpy as np
from numpy.typing import NDArray
from .layer import Layer
import math


class Pool(Layer):
    def __init__(self: Pool, dim: int) -> None:
        self.dim = dim

    def feed_forward(self: Pool, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        c, n, p = entry.shape
        out = np.zeros((c, math.ceil(n / self.dim), math.ceil(p / self.dim)))
        for i in range(c):
            for j in range(0, math.ceil(n / self.dim), self.dim):
                for k in range(0, math.ceil(p / self.dim), self.dim):
                    values = list(
                        entry[
                            i,
                            j : j + min(self.dim, n - self.dim * j),
                            k : k + min(self.dim, n - self.dim * k),
                        ].reshape(1, -1)
                    )
                    out[i, j // self.dim, k // self.dim] = max(values)  # max pool
        return out

    def descend_gradient(self: Pool, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        new_gradient = np.zeros_like(self.input)
        c, n, p = self.input.shape
        for i in range(c):
            for j in range(0, math.ceil(n / self.dim), self.dim):
                for k in range(0, math.ceil(p / self.dim), self.dim):
                    submatrix = self.input[
                        i,
                        j : j + min(self.dim, n - self.dim * j),
                        k : k + min(self.dim, n - self.dim * k),
                    ]
                    max_index = np.argmax(submatrix)
                    u, v = np.unravel_index(max_index, submatrix.shape)
                    new_gradient[i, j + u, k + v] = gradient[i, j // self.dim, k // self.dim]
        return new_gradient

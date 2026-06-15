import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class Conv(Layer):
    def __init__(self: Conv, N: int = 0, K: int = 0, S: int = 0):
        self.K = K  # Watching field dimension
        self.S = S  # Stride
        self.N = N
        self.kernels: list[NDArray[np.float64]] = []

    def set_input_shape(self: Conv, input_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        c, n, p = input_shape
        self.input_shape = input_shape
        self.kernels = [
            np.random.normal(0, np.sqrt(2 / self.K), size=(c, self.K, self.K)) for _ in range(self.N)
        ]  # He
        n_o = (n - self.K) // self.S + 1
        p_o = (p - self.K) // self.S + 1
        return (self.N, n_o, p_o)

    def feed_forward(self: Conv, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        _, n, p = entry.shape

        S = self.S
        K = self.K
        height = (n - K) // S + 1
        width = (p - K) // S + 1
        out = np.zeros((len(self.kernels), height, width))

        for i in range(height):
            for j in range(width):
                subtensor = entry[:, i * S : i * S + K, j * S : j * S + K]

                for k in range(len(self.kernels)):
                    crossed_correlation = np.sum(self.kernels[k] * subtensor)
                    out[k, i, j] = crossed_correlation

        return out

    def descend_gradient(self: Conv, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError

        new_gradient = np.zeros_like(self.input)
        C, n, p = self.input.shape
        S = self.S
        K = self.K
        for k in range(len(self.kernels)):
            kernel = self.kernels[k]

            height = (n - K) // S + 1
            width = (p - K) // S + 1
            for i in range(height):
                for j in range(width):
                    for u in range(K):
                        for v in range(K):
                            for c in range(C):
                                new_gradient[c, i * S + u, j * S + v] += gradient[k, i, j] * kernel[c, u, v]

            W_gradient = np.zeros_like(kernel)
            for i in range(height):
                for j in range(width):
                    for u in range(K):
                        for v in range(K):
                            for c in range(C):
                                W_gradient[c, u, v] += gradient[k, i, j] * self.input[c, i * S + u, j * S + v]
            self.kernels[k] -= self.lr * W_gradient

        return new_gradient

    def get_data(self: Conv) -> tuple[list[int], list[float]]:
        int_list = list(self.input_shape) + [self.K, self.S, self.N]
        float_list = [self.lr]
        for kernel in self.kernels:
            float_list += kernel.flatten().tolist()
        return int_list, float_list

    def load_from_data(self: Conv, int_list: list[int], float_list: list[float]) -> None:
        self.input_shape = tuple(int_list[:3])
        self.K = int_list[3]
        self.S = int_list[4]
        self.N = int_list[5]
        self.lr = float_list[0]
        del float_list[0]
        self.kernels = []
        for _ in range(self.N):
            length = self.input_shape[0] * self.K**2
            self.kernels += np.array(float_list[:length]).reshape(self.input_shape[0], self.K, self.K)
            del float_list[:length]

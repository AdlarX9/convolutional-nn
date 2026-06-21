import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class ExitLoss(Layer):
    def __init__(self: ExitLoss):
        super().__init__()

    def feed_forward(self: ExitLoss, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return entry  # no final activation

    def get_loss(self: ExitLoss, prediction: NDArray[np.float64], answer: NDArray[np.float64]) -> float:
        return np.sum(1 / 2 * (prediction - answer) ** 2)  # quadratic loss

    def get_gradient(
        self: ExitLoss, prediction: NDArray[np.float64], answer: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return prediction - answer


class ProbaExit(ExitLoss):
    def __init__(self: ExitLoss):
        super().__init__()

    def feed_forward(self: ExitLoss, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        max_entry = np.max(entry)
        exp_entry = np.exp(entry - max_entry)
        softmax = exp_entry / np.sum(exp_entry)
        return softmax

    def get_loss(self: ExitLoss, prediction: NDArray[np.float64], answer: NDArray[np.float64]) -> float:
        epsilon = 1e-10
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        loss = -np.sum(answer * np.log(prediction))  # cross-entropy
        return loss

    def get_gradient(
        self: ExitLoss, prediction: NDArray[np.float64], answer: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return prediction - answer  # cross-entropy + softmax

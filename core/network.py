import numpy as np
from numpy.typing import NDArray
from .layer import Layer
from .conv import Conv
from .biais import ConvBiais, Biais
from .activation import ReLU
from .fc import FC
from .exit import ExitLoss
from graphics import ConsoleVisualization


class Network:
    def __init__(
        self: Network,
        layers: list[Layer] = [],
        exit_loss: ExitLoss = ExitLoss(),
        input_shape: tuple[int, int, int] = (0, 0, 0),
        lr: float = 0.0001,
    ) -> None:
        self.lr = lr
        self.input_shape = input_shape
        self.exit_loss = exit_loss
        self.layers = self.build_layers(layers)

    def build_layers(self: Network, layers: list[Layer]) -> list[Layer]:
        if len(layers) == 0:
            return []
        new_layers: list[Layer] = []
        # Add unprecised layers
        for i in range(len(layers)):
            layer = layers[i]
            if type(layer) == Conv:
                new_layers.append(layer)
                new_layers.append(ConvBiais())
                new_layers.append(ReLU())
            elif type(layer) == FC:
                new_layers.append(layer)
                new_layers.append(Biais())
                new_layers.append(ReLU())
            else:
                new_layers.append(layer)
        # Set Learning Rate and Input Shape
        volume_shape = self.input_shape
        for layer in new_layers:
            layer.set_lr(self.lr)
            volume_shape = layer.set_input_shape(volume_shape)
        # Delete last ReLU layer
        if type(new_layers[-1]) == ReLU:
            del new_layers[-1]
        return new_layers

    def compute(self: Network, entry: NDArray[np.float64], memorize=False) -> NDArray[np.float64]:
        volume = entry
        for layer in self.layers:
            volume = layer.compute(volume, memorize)
        volume = self.exit_loss.feed_forward(volume)
        return volume

    def single_train(
        self: Network, entry: NDArray[np.float64], answer: NDArray[np.float64]
    ) -> tuple[float, bool]:
        prediction = self.compute(entry, memorize=True)
        loss = self.exit_loss.get_loss(prediction, answer)
        gradient = self.exit_loss.get_gradient(prediction, answer)
        correct = bool(np.argmax(prediction) == np.argmax(answer))
        for i in range(len(self.layers) - 1, -1, -1):
            gradient = self.layers[i].descend_gradient(gradient)
        return loss, correct

    def train(
        self: Network,
        data: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
        batch: int = 100,
        visualization: ConsoleVisualization | None = None,
    ) -> None:
        dashboard = visualization if visualization is not None else ConsoleVisualization(batch, len(data))
        try:
            seen_items = 0
            correct_items = 0
            for batch_index in range(1, batch + 1):
                for item_index, el in enumerate(data, start=1):
                    loss, is_correct = self.single_train(el[0], el[1])
                    seen_items += 1
                    if is_correct:
                        correct_items += 1
                    accuracy = correct_items / seen_items if seen_items else 0.0
                    dashboard.update(batch_index, item_index, loss, accuracy)
        finally:
            if visualization is None:
                dashboard.close()

from core import (
    Flatten,
    FC,
    Conv,
    ExitLoss,
    Network,
    Layer,
    ProbaExit,
    Biais,
    ReLU,
    ConvBiais,
    Tokenizer,
    Embedding,
)
from data import load_mnist_data, scrap_text, SaveHandler
from graphics import view_numbers, regression
import numpy as np
from numpy.typing import NDArray
import math
import random
import signal
import sys


def number_recognition() -> None:
    layers = [
        Conv(32, 3, 2),
        ConvBiais(),
        ReLU(),
        Conv(64, 3, 2),
        ConvBiais(),
        ReLU(),
        Conv(128, 3, 2),
        ConvBiais(),
        ReLU(),
        Flatten(),
        FC(10),
        Biais(),
    ]
    network = Network(layers=layers, exit_loss=ProbaExit(), input_shape=(1, 28, 28), lr=0.005)

    save_handler = SaveHandler()
    name = "number_recognition"
    if save_handler.has(name):
        network = save_handler.load(name)
        if not isinstance(network, Network):
            raise MemoryError

    data = load_mnist_data()
    network.train(data=data, batch=1)
    save_handler.save(network, name)

    view_numbers(network)


def learn_shape() -> None:
    def curve(x: float, y: float) -> float:
        circle = math.sqrt(0.5 * (x - 0.5) ** 2 + (y - 0.5) ** 2)
        donut = math.sin(7.5 * circle)
        return donut

    def get_data(dim: int) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        data = []
        for _ in range(dim):
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            data.append((np.array([[x], [y]]), np.array([[curve(x, y)]])))
        return data

    layers: list[Layer] = [
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(40),
        Biais(),
        ReLU(),
        FC(1),
        Biais(),
    ]
    network = Network(layers=layers, exit_loss=ExitLoss(), input_shape=(2, 1), lr=0.0001)

    save_handler = SaveHandler()
    name = "reproduce_shape"
    if save_handler.has(name):
        network = save_handler.load(name)
        if not isinstance(network, Network):
            raise MemoryError

    batch = 1000
    data = get_data(batch)
    network.train(data=data, batch=batch)
    save_handler.save(network, name)
    regression(network, curve)


def train_embedding():
    embedding_name = "embedding"

    def get_embedding() -> Embedding:
        tokenizer = Tokenizer()

        # Handle save
        save_handler = SaveHandler()
        if save_handler.has(embedding_name):
            embedding = save_handler.load(embedding_name)
            if not isinstance(embedding, Embedding):
                raise MemoryError
            return embedding
        else:
            # Build Tokenizer vocab
            text_for_vocab = scrap_text(1_000_000)
            tokenizer.build_vocab(text_for_vocab)

            # Build default Embedding
            embedding = Embedding()
            signal.signal(signal.SIGINT, save_handler.save(embedding, embedding_name))
            embedding.set_input_shape((tokenizer.length(), 1))
            save_handler.save(embedding, embedding_name)
            return embedding

    embedding = get_embedding()

    def see_familiarities(word: str):
        words, words2 = embedding.look_around(word)
        print("")
        for i in range(len(words)):
            print(words[i][0] + " " * (20 - len(words[i][0])) + " | " + words2[i][0])
        print("")

    def train():
        embedding.set_lr(0.1)
        text = scrap_text(length=10_000, offset=100_000)
        embedding.cbow_training(text, batch=1)
        SaveHandler().save(embedding, embedding_name)

    train()


def main() -> None:
    train_embedding()


if __name__ == "__main__":
    main()

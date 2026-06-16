from core import Flatten, FC, Conv, ExitLoss, Network
from data import load_mnist_data, SaveHandler
from graphics import visualize


def number_recognition() -> None:
    layers = [
        Conv(32, 3, 2),
        Conv(64, 3, 2),
        Conv(128, 3, 2),
        Flatten(),
        FC(10)
    ]
    network = Network(layers=layers, exit_loss=ExitLoss(), input_shape=(1, 28, 28), lr=0.005)

    save_handler = SaveHandler()
    name = "number_recognition"
    if save_handler.has(name):
        network = save_handler.load(name)

    data = load_mnist_data()
    network.train(data=data, batch=1)
    save_handler.save(network, name)

    visualize(network)


def main() -> None:
    number_recognition()


if __name__ == "__main__":
    main()

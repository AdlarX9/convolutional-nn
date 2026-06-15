from core import Flatten, FC, Conv, Network
from data import load_mnist_data, SaveHandler
from graphics import visualize


def main() -> None:
    layers = [
        Conv(32, 3, 2),
        Conv(64, 3, 2),
        Conv(128, 3, 2),
        Flatten(),
        FC(10),
    ]

    data = load_mnist_data()
    network = Network(layers=layers, lr=0.005, input_shape=(1, 28, 28))
    network.train(data=data, batch=1)
    SaveHandler().save(network, 'conv')
    visualize(network)



if __name__ == "__main__":
    main()

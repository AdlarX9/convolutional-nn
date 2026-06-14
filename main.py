from core import Flatten, FC, Network
from data import load_mnist_data

def main() -> None:
    layers = [
        Flatten(),
        FC(1000),
        FC(100),
        FC(10),
    ]

    data = load_mnist_data()
    network = Network(layers=layers, lr=0.001, input_shape=(1, 28, 28))

    network.train(data=data, batch=1)


if __name__ == "__main__":
    main()

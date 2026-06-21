from core import Network, Layer, Conv, FC, Biais, ConvBiais, ReLU, Flatten, ExitLoss, ProbaExit
import os
import struct
from io import BufferedWriter, BufferedReader
from typing import Any
from pathlib import Path

layer_types = {
    "Layer": Layer,
    "Conv": Conv,
    "FC": FC,
    "Biais": Biais,
    "ConvBiais": ConvBiais,
    "ReLU": ReLU,
    "Flatten": Flatten,
    "ExitLoss": ExitLoss,
    "ProbaExit": ProbaExit,
}


class SaveHandler:
    _instance: SaveHandler | None = None

    def __new__(cls) -> SaveHandler:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self: SaveHandler) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

    def get_path(self: SaveHandler, name: str) -> str:
        return os.path.join("data", "models", name + ".nn")

    def has(self: SaveHandler, name: str) -> bool:
        path = self.get_path(name)
        path = Path(path)
        return path.exists()

    def write_number(self: SaveHandler, f: BufferedWriter, val: float | int, type: str) -> None:
        f.write(struct.pack(type, val))

    def read_number(self: SaveHandler, f: BufferedReader, type: str) -> float | int:
        val = struct.unpack(type, f.read(4))[0]
        return val

    def write_list(self: SaveHandler, f: BufferedWriter, values: tuple | list, type: str) -> None:
        self.write_number(f, len(values), "i")
        f.write(struct.pack(f"{len(values)}{type}", *values))

    def read_list(self: SaveHandler, f: BufferedReader, type: str) -> list[Any]:
        length = struct.unpack("i", f.read(4))[0]
        values = list(struct.unpack(f"{length}{type}", f.read(4 * length)))
        return values

    def write_string(self: SaveHandler, f: BufferedWriter, string: str) -> None:
        self.write_number(f, len(string), "i")
        f.write(string.encode("utf-8"))

    def read_string(self: SaveHandler, f: BufferedReader) -> str:
        length = struct.unpack("i", f.read(4))[0]
        s = f.read(length).decode("utf-8")
        return s

    def write_layer(self: SaveHandler, f: BufferedWriter, layer: Layer) -> None:
        self.write_string(f, layer.__class__.__name__)
        int_list, float_list = layer.get_data()
        self.write_list(f, int_list, "i")
        self.write_list(f, float_list, "f")

    def read_layer(self: SaveHandler, f: BufferedReader) -> Layer:
        layer_type = self.read_string(f)
        int_list = self.read_list(f, "i")
        float_list = self.read_list(f, "f")
        new_layer: Layer = layer_types[layer_type]()
        new_layer.load_from_data(int_list, float_list)
        return new_layer

    def save(self: SaveHandler, network: Network, name: str) -> None:
        path = self.get_path(name)
        with open(path, "wb") as f:
            self.write_number(f, len(network.layers), "i")
            self.write_number(f, network.lr, "f")
            self.write_list(f, network.input_shape, "i")
            self.write_list(f, network.output_shape, "i")
            self.write_layer(f, network.exit_loss)
            for layer in network.layers:
                self.write_layer(f, layer)

    def load(self: SaveHandler, name: str) -> Network:
        network = Network()
        path = self.get_path(name)
        with open(path, "rb") as f:
            len_layers = int(self.read_number(f, "i"))
            network.lr = self.read_number(f, "f")
            network.input_shape = tuple(self.read_list(f, "i"))
            network.output_shape = tuple(self.read_list(f, "i"))
            exit_loss = self.read_layer(f)
            if isinstance(exit_loss, ExitLoss):
                network.exit_loss = exit_loss
            else:
                raise MemoryError("Expected ExitLoss, got:", type(exit_loss))
            for _ in range(len_layers):
                new_layer = self.read_layer(f)
                network.layers.append(new_layer)
        return network

    def test(self: SaveHandler):
        name = "test"
        path = self.get_path(name)
        string = "hey moi c'est Alexis !"
        number = 189
        other_number = 25.3
        int_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        float_list = [0.1, 0.2, 0.3]
        with open(path, "wb") as f:
            self.write_string(f, string)
            self.write_number(f, number, "i")
            self.write_number(f, other_number, "f")
            self.write_list(f, int_list, "i")
            self.write_list(f, float_list, "f")
        with open(path, "rb") as f:
            decoded_string = self.read_string(f)
            print(string, "=>", decoded_string)
            decoded_number = self.read_number(f, "i")
            print(number, "=>", decoded_number)
            decoded_other_number = self.read_number(f, "f")
            print(other_number, "=>", decoded_other_number)
            decoded_int_list = self.read_list(f, "i")
            print(int_list, "=>", decoded_int_list)
            decoded_float_list = self.read_list(f, "f")
            print(float_list, "=>", decoded_float_list)

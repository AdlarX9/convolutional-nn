from core import Network, Layer, Conv, FC, Biais, ConvBiais, ReLU, Flatten
import os
import struct

layer_types = [Layer, Conv, FC, Biais, ConvBiais, ReLU, Flatten]


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

    def save(self: SaveHandler, network: Network, name: str) -> None:
        path = os.path.join("data", "models", name + ".cnn")
        with open(path, "wb") as f:
            f.write(struct.pack("i", len(network.layers)))
            f.write(struct.pack("f", network.lr))  # learning rate
            f.write(struct.pack(f"{len(network.input_shape)}f", *network.input_shape))  # input shape
            for layer in network.layers:
                int_list, float_list = layer.get_data()
                f.write(struct.pack("i", layer_types.index(type(layer))))
                f.write(struct.pack("i", len(int_list)))
                f.write(struct.pack(f"{len(int_list)}i", *int_list))
                f.write(struct.pack("i", len(float_list)))
                f.write(struct.pack(f"{len(float_list)}f", *float_list))

    def load(self: SaveHandler, name: str) -> Network:
        network = Network()
        path = os.path.join("data", "models", name + ".cnn")
        with open(path, "rb") as f:
            len_layers = struct.unpack("i", f.read(4))[0]
            network.lr = struct.unpack("f", f.read(4))[0]
            network.input_shape = struct.unpack("3i", f.read(4 * 3))
            for _ in range(len_layers):
                layer_type = struct.unpack("i", f.read(4))[0]
                len_int = struct.unpack("i", f.read(4))[0]
                int_list = list(struct.unpack(f"{len_int}i", f.read(4 * len_int)))
                len_float = struct.unpack("i", f.read(4))[0]
                float_list = list(struct.unpack(f"{len_float}f", f.read(4 * len_float)))
                new_layer: Layer = layer_types[layer_type]()
                new_layer.load_from_data(int_list, float_list)
                network.layers.append(new_layer)
        return network

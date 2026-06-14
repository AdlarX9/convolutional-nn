from core import Network


class SaveHandler:
    _instance: SaveHandler | None = None

    def __new__(cls) -> SaveHandler:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

    def save(self: SaveHandler, network: Network, name: str) -> None:
        pass

    def load(self: SaveHandler, path: str) -> Network:
        pass

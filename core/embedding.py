from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .layer import Layer
from .tokenizer import Tokenizer
from .exit import softmax
from graphics import ConsoleVisualization
import math


class Embedding(Layer):
    def __init__(self: Embedding, dim: int = 100) -> None:
        super().__init__()
        self.dim = dim
        self.W = np.array([[]])
        self.W_prime = np.array([[]])

    def set_input_shape(self: Embedding, input_shape: tuple[int, int]) -> tuple[int, int]:
        if len(input_shape) != 2 and input_shape[0] != Tokenizer().length():
            raise ValueError(
                "Expected dimension does not fit Tokenizer requiremenents:",
                (Tokenizer().length(), 1),
                "!=",
                input_shape,
            )
        self.input_shape = input_shape
        self.W = np.random.uniform(-1, 1, (self.dim, Tokenizer().length()))
        self.W_prime = np.random.uniform(-1, 1, (Tokenizer().length(), self.dim))
        self.output_shape = (self.dim, self.input_shape[1])
        return self.output_shape

    def feed_forward(self: Embedding, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.W @ entry

    def descend_gradient(self: Embedding, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.input is None:
            raise MemoryError
        new_gradient = self.W.T @ gradient
        self.W -= self.lr * gradient @ self.input.T
        return new_gradient

    def cbow_training(self: Embedding, text: str, window: int = 7, batch: int = 10) -> None:
        tokenizer = Tokenizer()
        words = tokenizer.split_text(text)
        dashboard = ConsoleVisualization(batch, len(words) - 2 * window)
        max_length = 500
        corrects = []
        losses = []
        for batch_index in range(1, batch + 1):
            for i in range(window, len(words) - window):
                target = words[i]
                target_index = tokenizer.get_index(target)
                answer = tokenizer.get_one_hot(target)

                # Context vector
                contexts = words[i - window : i] + words[i + 1 : i + window + 1]
                one_hots = [tokenizer.get_one_hot(word) for word in contexts]
                embedded_contexts = [self.feed_forward(one_hot) for one_hot in one_hots]
                embedded_context = sum(embedded_contexts) / len(embedded_contexts)

                # Compute score
                score = self.W_prime @ embedded_context
                probability = softmax(score)

                # Learn
                gradient = probability - answer
                new_gradient = self.W_prime.T @ gradient
                self.W_prime -= self.lr * gradient @ embedded_context.T  # type: ignore
                self.W -= self.lr * new_gradient @ sum(one_hots).T / (2 * window)  # type: ignore

                # Dashboard
                correct = bool(np.argmax(probability) == np.argmax(answer))
                corrects.append(correct)
                if len(corrects) > max_length:
                    corrects.pop(0)
                loss = -math.log(probability[target_index, 0])
                losses.append(loss)
                if len(losses) > max_length:
                    losses.pop(0)
                dashboard.update(
                    batch_index,
                    i + 1 - window,
                    sum(losses) / len(losses),
                    corrects.count(True) / len(corrects),
                )

    def cosine_similarity(self: Embedding, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def distance(self: Embedding, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.sum((b - a) ** 2))

    def estimate(self: Embedding, formula: str) -> str:
        split = formula.split(" ")
        if len(split) % 2 == 0:
            raise ValueError("Invalid Formula")
        words = split[::2]
        symbols = split[1::2]
        vectors = [self.W[:, Tokenizer().get_index(word)].copy() for word in words]
        output = vectors[0].copy()
        index = 1
        for symbol in symbols:
            if symbol == "+":
                output += vectors[index]
            elif symbol == "-":
                output -= vectors[index]
            index += 1
        best_word = ""
        best_similarity = -math.inf
        for word in Tokenizer().V:
            embedded_word = self.W[:, Tokenizer().get_index(word)]
            similarity = self.cosine_similarity(output, embedded_word)
            if similarity > best_similarity:
                print(similarity, word)
                best_similarity = similarity
                best_word = word
        return best_word

    def look_around(self: Embedding, word: str) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        embedded_word = self.W[:, Tokenizer().get_index(word)]
        words = [
            (
                random_word,
                self.cosine_similarity(embedded_word, self.W[:, Tokenizer().get_index(random_word)]),
            )
            for random_word in Tokenizer().V.keys()
        ]
        words2 = [
            (
                random_word,
                self.distance(embedded_word, self.W[:, Tokenizer().get_index(random_word)]),
            )
            for random_word in Tokenizer().V.keys()
        ]
        words = sorted(words, key=lambda x: x[1], reverse=True)
        words2 = sorted(words2, key=lambda x: x[1])
        return words[:10], words2[:10]

    def get_data(self: Embedding) -> tuple[list[int], list[float], list[str]]:
        int_list = (
            list(self.input_shape)
            + list(self.output_shape)
            + [self.dim, Tokenizer().length()]
            + list(Tokenizer().V.values())
        )
        float_list = [self.lr] + self.W.flatten().tolist() + self.W_prime.flatten().tolist()
        string_list = list(Tokenizer().V.keys())
        return int_list, float_list, string_list

    def load_from_data(
        self: Embedding, int_list: list[int], float_list: list[float], string_list: list[str]
    ) -> None:
        self.input_shape = tuple(int_list[:2])
        del int_list[:2]
        self.output_shape = tuple(int_list[:2])
        del int_list[:2]
        self.dim = int_list.pop(0)
        tokenizer_length = int_list.pop(0)
        self.lr = float_list.pop(0)
        self.W = np.array(float_list[: tokenizer_length * self.dim]).reshape(self.dim, tokenizer_length)
        del float_list[: tokenizer_length * self.dim]
        self.W_prime = np.array(float_list).reshape(tokenizer_length, self.dim)
        if Tokenizer().length() != tokenizer_length:
            Tokenizer().V = {}
            for i in range(tokenizer_length):
                Tokenizer().V[string_list[i]] = int_list[i]

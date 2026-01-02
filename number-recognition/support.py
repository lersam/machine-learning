import pickle
from pathlib import Path


def load_mnist_pickle(file_path: str) -> tuple:
    _path = Path(file_path)
    if not _path.is_file():
        raise FileNotFoundError(f"No such file: '{file_path}'")

    """Load data from a local file."""
    with _path.open(mode='rb') as file:
        first = pickle.load(file)
        second = pickle.load(file)
        return first, second


def load_mnist_data(mnist_path: str) -> tuple:
    """Load MNIST data from the default pickle file."""

    first, second = load_mnist_pickle(mnist_path)

    # Heuristics: prefer 2D array as X (samples, features) and 1D as y (labels)
    if hasattr(first, 'shape') and hasattr(second, 'shape'):
        if getattr(second, 'ndim', None) == 2 and getattr(first, 'ndim', None) == 1:
            first, second = second, first

    return first, second

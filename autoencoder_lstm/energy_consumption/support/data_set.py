import torch
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
from typing import Optional, Tuple


class EnergyConsumption(torch.utils.data.Dataset):
    """
    Sequence dataset for Individual Household Electric Power Consumption.

    Usage patterns:
    - Initialize with a local CSV/Parquet path via load_local() to avoid repeated network calls.
    - Or call save_ucirepo_to_local() once to fetch via ucimlrepo and persist locally.
    """

    def __init__(self, history_samples, horizon_samples, scale=True, train_fraction=0.9, scaler=None):
        super().__init__()
        self.history_samples = history_samples
        self.horizon_samples = horizon_samples
        self.train_fraction = train_fraction
        self.data_features = None
        self.data_headers = None
        self._load_dataset()
        if self.data_features is None:
            raise ValueError("Failed to load dataset.")


    def _load_dataset(self):
        data_features_path = Path(Path(__file__).parent,"../local_data/data_features.feather")
        data_headers_path = Path(Path(__file__).parent, "../local_data/data_headers.npy")

        if data_features_path.exists() and data_headers_path.exists():
            self.data_features = pd.read_feather(data_features_path.absolute())

            values = np.load(data_headers_path.absolute(), allow_pickle=True)
            self.data_headers = pd.Index(values)
        else:
            # read from network
            power_consumption = fetch_ucirepo(id=235)
            self.data_features = power_consumption.data.features

            self.data_headers = power_consumption.data.headers

            # clearing
            self.data_features.replace('?', np.nan, inplace=True)

            # saving locally
            self.data_features.to_feather(data_features_path.absolute())

            np.save(data_headers_path.absolute(), self.data_headers.values)






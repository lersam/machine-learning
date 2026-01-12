import torch
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from pathlib import Path


class EnergyConsumptionDataset(torch.utils.data.Dataset):
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

        self.number_of_channels = self.data_features.shape[1]

    def __len__(self):
        return len(self.data_features) - self.history_samples - self.horizon_samples + 1

    def __getitem__(self, idx):
        start_hist = idx
        end_hist = idx + self.history_samples
        start_horizon = end_hist
        end_horizon = start_horizon + self.horizon_samples

        history = self.data_features.iloc[start_hist:end_hist].values
        horizon = self.data_features.iloc[start_horizon:end_horizon].values

        return torch.tensor(history, dtype=torch.float32), torch.tensor(horizon, dtype=torch.float32)

    def _load_dataset(self):
        data_features_path = Path(Path(__file__).parent, "../local_data/data_features.csv")
        data_headers_path = Path(Path(__file__).parent, "../local_data/data_headers.npy")

        if data_features_path.exists() and data_headers_path.exists():
            self.data_features = pd.read_csv(data_features_path.absolute())

            values = np.load(data_headers_path.absolute(), allow_pickle=True)
            self.data_headers = pd.Index(values)
        else:
            # read from network
            power_consumption = fetch_ucirepo(id=235)
            self.data_features = power_consumption.data.features

            self.data_headers = power_consumption.data.headers

            # clearing
            self.data_features.replace('?', np.nan, inplace=True)

            data_features_path.parent.mkdir(parents=True, exist_ok=True)
            # saving locally
            self.data_features.to_csv(data_features_path.absolute(), index=False)

            np.save(data_headers_path.absolute(), self.data_headers.values)

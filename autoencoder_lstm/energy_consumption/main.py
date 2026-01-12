from support.data_set import EnergyConsumptionDataset

if __name__ == '__main__':
    energy_data = EnergyConsumptionDataset(history_samples=24 * 14, horizon_samples=24 * 3, scale=True, train_fraction=0.9)

    print(energy_data.data_features)
    print(energy_data.data_headers)
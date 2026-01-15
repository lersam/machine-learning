import os

import torch

from support import build_loaders, EnergyConsumptionModule, demo_model_shapes, \
    plot_convergence, plot_predictions_grid, plot_mse_horizon, train_and_evaluate
from support.data_set import EnergyConsumptionDataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    data_folder = "local_data/cache"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)

    energy_data = EnergyConsumptionDataset(history_samples=24 * 14, horizon_samples=24 * 3, scale=True,
                                           train_fraction=0.9)

    train_dataset, test_dataset, train_loader, test_loader = build_loaders(energy_data, train_fraction=0.9,
                                                                           batch_size_train=200, batch_size_eval=2000)

    model = EnergyConsumptionModule(input_dim=energy_data.number_of_channels, output_dim=energy_data.horizon_samples)
    demo_model_shapes(model, train_loader)


    checkpoint_file_name = os.path.join(data_folder, "ER Admission - CNN-LSTM Fusion.checkpoint.pt")
    model, progress_log, test_accuracy, test_loader = train_and_evaluate(model, train_dataset, test_dataset,
                                                                         device=device, n_epochs=10, batch_size=200,
                                                                         lr=1e-3,
                                                                         checkpoint_file_name=checkpoint_file_name)
    plot_convergence(progress_log, test_accuracy)
    plot_predictions_grid(model, test_loader, device=device, n=6)
    plot_mse_horizon(model, test_loader, energy_data.horizon_samples, device=device)

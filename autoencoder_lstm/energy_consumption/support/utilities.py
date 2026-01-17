import random
import os
import time
import numpy as np
import torch
from matplotlib import pyplot as plt


def split_indices(n_windows: int, train_fraction: float = 0.9):
    split = int(n_windows * train_fraction)
    return list(range(split)), list(range(split, n_windows))


def build_loaders(dataset, *, train_fraction=0.9, batch_size_train=200, batch_size_eval=2000, shuffle=True):
    train_idx, test_idx = split_indices(len(dataset), train_fraction)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=shuffle,
                                               drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False, drop_last=False)
    return train_dataset, test_dataset, train_loader, test_loader


def demo_model_shapes(model, train_loader, device=None):
    x, y = next(iter(train_loader))
    if device is not None:
        x = x.to(device)
    with torch.no_grad():
        y_pred = model(x)

    print(f"x: {tuple(x.shape)}, model(x): {tuple(y_pred.shape)}, y: {tuple(y.shape)}")


def model_accuracy(model, data_loader):
    if data_loader is None:
        return -1

    # allow passing a Dataset as well as a DataLoader
    if isinstance(data_loader, torch.utils.data.Dataset):
        data_loader = torch.utils.data.DataLoader(data_loader, batch_size=2000, shuffle=False, drop_last=False)

    device_model = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        current_mse = 0.0
        samples_counts = 0
        loss_fn = torch.nn.MSELoss(reduction='sum')
        for X, y in data_loader:
            X = X.to(device=device_model)
            y = y.to(device=device_model)
            # if targets are (batch, channels, horizon) reduce to primary channel
            if y.ndim == 3:
                y = y[:, 0, :]
            y_pred = model(X)
            current_mse += loss_fn(y_pred, y).item()
            samples_counts += y.shape[0]
        return current_mse / samples_counts


def plot_convergence(progress_log, test_accuracy):
    import pandas as pd
    log_df = pd.DataFrame(progress_log)
    plt.figure(figsize=(10, 6))
    if 'loss' in log_df:
        plt.plot(log_df.epoch, log_df.loss, label='train loss', marker='o', alpha=0.7)
    plt.plot(log_df.epoch, test_accuracy * np.ones_like(log_df.epoch), label=f'test MSE: {test_accuracy:0.4f}',
             linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)


def plot_predictions_grid(model, test_loader, device=None, n=6):
    model.eval()
    with torch.no_grad():
        X_all, y_all = next(iter(test_loader))
        if device is not None:
            X_all = X_all.to(device)
        y_pred_all = model(X_all).detach().cpu()
    y_all = y_all.cpu()
    idxs = random.sample(range(min(len(y_all), 1000)), k=min(n, len(y_all)))
    rows, cols = 2, 3
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i + 1)
        plt.plot(y_all[idx], label='Actual', marker='o', markersize=3, linewidth=2)
        plt.plot(y_pred_all[idx], label='Predicted', marker='x', markersize=3, linewidth=2, alpha=0.8)
        plt.xlabel('Hours')
        plt.ylabel('Normalized')
        plt.title(f'Sample {idx}')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()
    plt.tight_layout()


def plot_mse_horizon(model, test_loader, horizon, device=None):
    model.eval()
    mse_t = torch.zeros(horizon)
    with torch.no_grad():
        for X, y in test_loader:
            if device is not None:
                X, y = X.to(device), y.to(device)
            # reduce targets to primary channel if necessary: (batch, channels, horizon) -> (batch, horizon)
            if y.ndim == 3:
                y = y[:, 0, :]
            # ensure predictions match target shape (reduce channels if model outputs channels dimension)
            y_pred = model(X)
            if y_pred.ndim == 3:
                y_pred = y_pred[:, 0, :]
            mse_t += (y_pred - y).square().sum(dim=0).to('cpu')
    mse_t = mse_t / len(test_loader.dataset)
    plt.figure(figsize=(12, 6))
    plt.plot(mse_t, '*:', linewidth=2, markersize=8)
    plt.xlabel('Time (hours)')
    plt.ylabel('Mean MSE')
    plt.title('MSE over Prediction Horizon')
    plt.grid(True, alpha=0.3)


def training_loop(model, optimizer, criterion, device, train_dataset, validation_dataset, n_epochs=50, batch_size=20,
                  checkpoint_file_name=None):

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                                    drop_last=False) if validation_dataset is not None else None

    if (checkpoint_file_name is not None) and (os.path.isfile(checkpoint_file_name)):  # checkpoint found!
        checkpoint_data = torch.load(checkpoint_file_name)

        first_epoch = checkpoint_data['epoch'] + 1  # +1 continue from the following epoch
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        progress_log = checkpoint_data['progress_log']
        # print progress log so far:
        for row in progress_log:
            epoch, epoch_run_time, epoch_loss, train_accuracy, validate_accuracy = row['epoch'], row[
                'epoch_start_time'], row['loss'], row['train_accuracy'], row['validate_accuracy']
            print(f"{epoch + 1} of {n_epochs}. time={epoch_run_time:0.2f}sec. Loss: {epoch_loss:0.4f}."
                  f"Train accuracy: {train_accuracy:0.4f}. Validate accuracy: {validate_accuracy:0.4f}")
    else:
        first_epoch = 0
        progress_log = []

    model.to(device)

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(first_epoch, n_epochs):
        epoch_start_time = time.time()
        model.train()  # set model into train mode.

        current_loss = 0
        total_samples = len(train_loader)
        for index, (train_data, train_data_labels) in enumerate(train_loader):
            train_data = train_data.to(device=device)
            train_data_labels = train_data_labels.to(device=device)  # move train data to device.

            # reduce targets to primary channel if necessary: (batch, channels, horizon) -> (batch, horizon)
            if train_data_labels.ndim == 3:
                train_data_labels = train_data_labels[:, 0, :]

            optimizer.zero_grad()
            output = model(train_data)
            loss = criterion(output, train_data_labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item() * train_data.shape[0]
            if index % 1_000 == 0:
                print(f"{index :,} of {total_samples:,}. batch loss: {loss.item():0.4f}.",)

        # evaluate performance of the batch: torch.no_grad() is not required since it's called in the model_accuracy function.
        train_accuracy = model_accuracy(model, train_loader)
        validate_accuracy = model_accuracy(model, validation_loader) if validation_loader is not None else -1
        print(f"{epoch + 1} of {n_epochs}. time={time.time() - epoch_start_time:0.2f}sec."
              f" Loss: {current_loss / len(train_loader.dataset):0.4f}. Train accuracy: {train_accuracy:0.4f}."
              f" Validate accuracy: {validate_accuracy:0.4f}")
        progress_log.append({'epoch': epoch + 1, 'epoch_start_time': time.time() - epoch_start_time,
                             'loss': current_loss / len(train_loader.dataset), 'train_accuracy': train_accuracy,
                             'validate_accuracy': validate_accuracy})

        # save checkpoint:
        if checkpoint_file_name is not None:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'progress_log': progress_log
            }
            torch.save(checkpoint_data, checkpoint_file_name)
    model.eval()
    return model, progress_log


def train_and_evaluate(model, train_dataset, test_dataset, *, device, n_epochs=10, batch_size=200, lr=1e-3,
                       checkpoint_file_name=None):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, progress_log = training_loop(model, optimizer, criterion, device, train_dataset, validation_dataset=None,
                                        n_epochs=n_epochs, batch_size=batch_size,
                                        checkpoint_file_name=checkpoint_file_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2000, shuffle=False, drop_last=False)
    test_accuracy = model_accuracy(model, test_loader)
    return model, progress_log, test_accuracy, test_loader

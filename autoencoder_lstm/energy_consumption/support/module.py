import torch


class EnergyConsumptionModule(torch.nn.Module):
    """
    Autoencoder LSTM module for energy consumption data.
    """

    def __init__(self, window_size, input_dim, output_dim, lstm_hidden_size=50, lstm_num_layers=2):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        # CNN branch using Sequential
        self.cnn_branch = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.input_dim, out_channels=64, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Dropout(p=0.3),

            torch.nn.Conv1d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Dropout(p=0.3)
        )
        self.cnn_output_size = 32

        # LSTM branch on raw channels
        self.lstm = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True, dropout=0.2 if self.lstm_num_layers>1 else 0)
        fusion_size = self.cnn_output_size + self.lstm_hidden_size

        # FC head using Sequential
        self.fc_head = torch.nn.Sequential(
            torch.nn.Linear(fusion_size, 150),
            torch.nn.BatchNorm1d(150),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(150, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(100, self.output_dim)
        )
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        # CNN branch
        cnn_out = self.cnn_branch(x)
        cnn_encoding = cnn_out.mean(dim=2)

        # LSTM branch (batch, time, channels)
        lstm_in = x.transpose(1,2)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_encoding = lstm_out[:, -1, :]
        fused = torch.cat([cnn_encoding, lstm_encoding], dim=1)
        return self.fc_head(fused)
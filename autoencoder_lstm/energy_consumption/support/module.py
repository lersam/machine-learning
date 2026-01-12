import torch


class EnergyConsumptionModule(torch.nn.Module):
    """
    Autoencoder LSTM module for energy consumption data.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.encoder = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                     num_layers=num_layers, dropout=dropout, batch_first=True)
        self.decoder = torch.nn.LSTM(input_size=hidden_size, hidden_size=input_size,
                                     num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        # Encode
        _, (hidden, _) = self.encoder(x)
        # Repeat hidden state for each time step in the output sequence
        repeated_hidden = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        # Decode
        decoded_output, _ = self.decoder(repeated_hidden)
        return decoded_output
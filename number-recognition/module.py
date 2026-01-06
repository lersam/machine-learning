import torch
import os


class NumberRecognition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_head = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(64, 10),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc_head(x)

# -----------------------------------------------------------
# Save the entire model
def save_entire_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model, path)


# Load it back
def load_entire_model(path) -> torch.nn.Module:
    return torch.load(path).eval()


# -----------------------------------------------------------
# Save only the model parameters (recommended approach)
def save_model_parameters(model, path):
    """Save the model's state dictionary to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


# Load it back
def load_model_parameters(model, path):
    model.load_state_dict(torch.load(path))
    return model.eval()

# -----------------------------------------------------------
# Save model with optimizer state
def save_model_with_optimizer(model, optimizer, path, epoch, epoch_loss, accuracy):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'accuracy': accuracy
    }, path)

# Load it back
def load_model_with_optimizer(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    return model.eval(), optimizer, epoch, loss, accuracy

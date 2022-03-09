import numpy as np
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd

if not torch.cuda.is_available():
    warnings.warn('CUDA is not available.')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module) :
    def __init__(self):
        super(MLP, self).__init__()
        self.input_size = None

        self.network = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def train_epoch(model, data_loader, optimizer: torch.optim, learning_rate=0.1, loss_func=nn.MSELoss()):

    model.train()
    train_loss = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        x, y = batch.values()
        x = x.to(device)
        y = y.to(device)

        output = model.forward(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad:
            train_loss += loss.sum().cpu().numpy()

    print("Train loss :", train_loss)


def valid_epoch(model, data_loader, loss_func=nn.MSELoss()):

    model.eval()
    valid_loss = 0
    with torch.no_grad:
        for batch in tqdm(data_loader):
            x, y = batch.values()
            x = x.to(device)
            y = y.to(device)

            output = model.forward(x)
            loss = loss_func(output, y)
            valid_loss += loss.sum().cpu().numpy()

    print("Validation loss :", valid_loss)


def train(train, valid, epochs, lr, batch_size=1, loss_func=nn.MSELoss()):
    x = torch.tensor(train)


def make_dataset(file_name='variable_length_dataset.csv') :
    dataset = pd.read_csv(file_name, header=None, delimiter=',', dtype={0: str, 1: float})
    # np_data = dataset.to_numpy()
    x_array = dataset.values[:, 0].astype(str)
    y_array = dataset.values[:, 1].astype(float)

    x_tensor = torch.tensor(x_array)
    y_tensor = torch.tensor(y_array)

    return TensorDataset(x_tensor, y_tensor)



dataset = make_dataset()
n_examples = len(dataset)
train, valid, test = random_split(dataset, [n_examples*0.7, n_examples*0.2, n_examples*0.1], generator=torch.Generator().manual_seed(42))

print()

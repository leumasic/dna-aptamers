import numpy as np
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
import pandas as pd


if not torch.cuda.is_available():
    warnings.warn("CUDA is not available.")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DNA_ONE_HOT = {
    "A": [0, 0, 0, 1],
    "C": [0, 0, 1, 0],
    "G": [0, 1, 0, 0],
    "T": [1, 0, 0, 0],
}
MAX_LEN = 20


def make_dataset(file_name="variable_length_dataset.csv", targets=DNA_ONE_HOT):
    dataset = pd.read_csv(
        file_name, header=None, delimiter=",", dtype={0: str, 1: float}
    )
    # np_data = dataset.to_numpy()
    x_array = dataset.values[:, 0].astype(str)
    y_array = dataset.values[:, 1].astype(float)

    # *** with max length of aptamer sequence 60 ***
    x_one_hot_array = np.zeros((len(x_array), MAX_LEN, 4))

    for i, sequence in enumerate(x_array):
        one_hot = np.array([targets[letter] for letter in sequence])
        padded_one_hot = np.pad(
            one_hot,
            ((0, MAX_LEN - one_hot.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        x_one_hot_array[i] = padded_one_hot

    x_tensor = torch.tensor(x_one_hot_array, dtype=torch.float32)
    y_tensor = torch.tensor(y_array, dtype=torch.float32)

    return TensorDataset(x_tensor, y_tensor)


def split_dataset(dataset, seed=42, train_split=0.8):
    n_examples = len(dataset)
    nb_train = int(n_examples * train_split)
    train, test = random_split(
        dataset,
        [nb_train, n_examples - nb_train],
        generator=torch.Generator().manual_seed(seed),
    )

    return train, test


num_epochs = 50
batch_size = 1
learning_rate = 0.001


class ConvNet(nn.Module):
    def __init__(self, max_len):
        super(ConvNet, self).__init__()

        self.max_len = max_len

        self.network = nn.Sequential(
            # nn.Flatten(),
            nn.Conv1d(4, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(192, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # print(x.shape)
        # x = nn.Flatten()(x)
        # print(x.shape)
        x = x.view(batch_size, 4, MAX_LEN)
        # x = x.permute(1, 0, 2)
        # print(x.shape)
        return self.network(x)

    def predict(self, x):
        self.eval()
        return self.forward(x)


def train_epoch(
    model, data_loader, optimizer: torch.optim, loss_func=nn.MSELoss(), device=DEVICE
):

    model.train()
    train_loss = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        x = batch[0]
        y = batch[1]
        x = x.to(device)
        y = y.to(device)

        # TODO fix float type at source
        output = model.forward(x)
        loss = torch.sqrt(loss_func(output, y))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += loss.item()

    print("Train loss :", train_loss / len(data_loader))


def valid_epoch(model, data_loader, loss_func=nn.MSELoss(), device=DEVICE):

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0]
            y = batch[1]
            x = x.to(device)
            y = y.to(device)

            # TODO fix float type at source
            output = model.forward(x)
            loss = torch.sqrt(loss_func(output, y))
            valid_loss += loss.item()

    print("Validation loss :", valid_loss / len(data_loader))


def train(
    model,
    train_dataset,
    valid_dataset,
    epochs,
    learning_rate=0.1,
    batch_size=1,
    loss_func=nn.MSELoss(),
    device=DEVICE,
):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    for i in range(epochs):
        train_epoch(model, train_dataloader, optimizer, loss_func, device)
        valid_epoch(model, valid_dataloader, loss_func, device)


dataset = make_dataset(
    "/c/Users/arazaghi/OneDrive - Ubisoft/Documents/School/IFT6759/dna-aptamers/src/variable_length_dataset.csv"
)
train_set, validation_set = split_dataset(dataset)
cnn_model = ConvNet(60)

train(
    cnn_model,
    train_set,
    validation_set,
    epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    loss_func=nn.MSELoss(),
    device=DEVICE,
)

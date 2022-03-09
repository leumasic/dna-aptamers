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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DNA_ONE_HOT = {'A': [0, 0, 0, 1],
               'C': [0, 0, 1, 0],
               'G': [0, 1, 0, 0],
               'T': [1, 0, 0, 0]
               }
MAX_LEN = 60

class MLP(nn.Module) :
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, 164),
            nn.ReLU(),
            nn.Linear(164, 164),
            nn.ReLU(),
            nn.Linear(164, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        self.eval()
        return self.forward(x)


def train_epoch(model, data_loader, optimizer: torch.optim, loss_func=nn.MSELoss(), device=DEVICE):

    model.train()
    train_loss = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        x = batch[0]
        y = batch[1]
        x = x.to(device)
        y = y.to(device)

        # TODO fix float type at source
        output = model.forward(x.float())
        loss = loss_func(output.float(), y.float())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += loss.sum().cpu().numpy()

    print("Train loss :", train_loss)


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
            output = model.forward(x.float())
            loss = loss_func(output, y)
            valid_loss += loss.sum().cpu().numpy()

    print("Validation loss :", valid_loss)


def train(model, train_dataset, valid_dataset, epochs, learning_rate=0.1, batch_size=1, loss_func=nn.MSELoss(), device=DEVICE):

    model.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    for i in range(epochs):
        train_epoch(model, train_dataloader, optimizer, loss_func, device)
        valid_epoch(model, valid_dataloader, loss_func, device)



def make_dataset(file_name='variable_length_dataset.csv', targets=DNA_ONE_HOT):
    dataset = pd.read_csv(file_name, header=None, delimiter=',', dtype={0: str, 1: float})
    # np_data = dataset.to_numpy()
    x_array = dataset.values[:, 0].astype(str)
    y_array = dataset.values[:, 1].astype(float)

    # *** with max length of aptamer sequence 60 ***
    x_one_hot_array = np.zeros((len(x_array), MAX_LEN, 4))

    for i, sequence in enumerate(x_array):
        one_hot = np.array([targets[letter] for letter in sequence])
        padded_one_hot = np.pad(one_hot, ((0, MAX_LEN-one_hot.shape[0]), (0, 0)), mode='constant', constant_values=0)
        x_one_hot_array[i] = padded_one_hot

    x_tensor = torch.tensor(x_one_hot_array)
    y_tensor = torch.tensor(y_array)

    return TensorDataset(x_tensor, y_tensor)


# ***
# split dataset in 2 with a proportion
# Usage :
# -use once for train,valid
# -use twice for train,valid, test (recall function on second split)
# ***
def split_dataset(dataset, seed=42, train_split=0.8):
    n_examples = len(dataset)
    nb_train = int(n_examples * train_split)
    train, test = random_split(dataset, [nb_train, n_examples-nb_train], generator=torch.Generator().manual_seed(seed))

    return train, test


dataset = make_dataset()
train_set, validation_set = split_dataset(dataset)
mlp_model = MLP(240, 1)

train(mlp_model, train_set, validation_set, epochs=5, learning_rate=0.1, batch_size=5, loss_func=nn.MSELoss(), device=DEVICE)

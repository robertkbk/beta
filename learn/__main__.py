from pathlib import Path
from sklearn.preprocessing import StandardScaler
import tap
import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
from tqdm import tqdm


class Arguments(tap.Tap):
    dataset: Path = Path("dataset/beta")
    window: int = 60
    target: int = 5


class BetaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Path, window: int, target: int) -> None:
        self.window = window
        self.target = target
        self.data = [
            item for filename in dataset.iterdir() for item in self.from_file(filename)
        ]

    def __getitem__(self, index):
        data = self.data[index]
        return (
            torch.Tensor(data[: self.window]),
            torch.Tensor(data[self.window :]),
        )

    def __len__(self):
        return len(self.data)

    @staticmethod
    def sequences(series: np.ndarray, window: int, step: int):
        return (
            series[i : i + window] for i in range(0, len(series) - window - step, step)
        )

    def from_file(self, filename: Path):
        data = pd.read_csv(filename)
        array = data["Beta"].to_numpy()

        if len(array) == 0:
            return []

        normalized = StandardScaler().fit_transform(array.reshape(-1, 1))
        return self.sequences(normalized, self.window + self.target, 7)


class Beta(torch.nn.Module):
    def __init__(self, window: int, target: int):
        super().__init__()

        self.network = torch.nn.Linear(window, target)
        # torch.nn.Sequential(
        #     torch.nn.Linear(window, 32),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(32, 16),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(16, target),
        # )

    def forward(self, x: torch.Tensor):
        return self.network.forward(x)


def main(dataset: Path, window: int, target: int):
    # BATCH_SIZE = 8
    SPLIT = 0.8
    DEVICE = "cpu"
    LR = 4e-4
    EPOCHS = 10

    print("Loading dataset...")
    dataset = BetaDataset(dataset, window, target)
    model = Beta(window, target)

    print("Initializing dataset...")
    train_set, test_set = torch.utils.data.random_split(dataset, (SPLIT, 1 - SPLIT))

    trainloader = torch.utils.data.DataLoader(train_set, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_set, shuffle=True, drop_last=True)

    criterion = torch.nn.MSELoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Learning...")
    for epoch in range(EPOCHS):
        train_loss, valid_loss = 0.0, 0.0

        model.train()

        for x, y in tqdm(trainloader):
            optimizer.zero_grad()

            x = x.squeeze().to(DEVICE)
            y = y.squeeze().to(DEVICE)

            preds = model(x)
            # .squeeze()

            loss = criterion(preds, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            epoch_loss = train_loss / len(trainloader)
            # t_losses.append(epoch_loss)

        model.eval()

        for x, y in tqdm(testloader):
            with torch.no_grad():
                x, y = x.squeeze().to(DEVICE), y.squeeze().to(DEVICE)
                preds = model(x).squeeze()
                error = criterion(preds, y)

            valid_loss += error.item()
            valid_loss = valid_loss / len(testloader)
            # v_losses.append(valid_loss)

        print(f"{epoch} - train: {epoch_loss}, valid: {valid_loss}")
        # plot_losses(t_losses, v_losses)

    torch.save(model, "beta.pt")


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args.dataset, args.window, args.target)

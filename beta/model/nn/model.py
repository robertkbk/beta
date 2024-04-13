from torch import Tensor, nn

from beta.model.nn.dishTS import DishTS


class BetaModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        lookback: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dishts = DishTS(input_size=input_size, lookback=lookback)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bias=True,
            batch_first=True,
            bidirectional=False,
            proj_size=0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dishts.forward(x)
        out, hidden = self.lstm.forward(x)
        out = self.head.forward(out)
        out = self.dishts.forward(out, inverse=True)
        return out

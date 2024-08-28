import torch
from torch import nn


class DishTS(nn.Module):
    def __init__(self, input_size: int, lookback: int, activate: bool = False) -> None:
        super().__init__()

        self._activate = activate
        self._phil: torch.Tensor
        self._phih: torch.Tensor

        self._reduce_layer = nn.Parameter(
            torch.randn(input_size, lookback, 2) / lookback
        )

        self._a = nn.Parameter(torch.ones(input_size))
        self._b = nn.Parameter(torch.zeros(input_size))

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if inverse:
            return self._inverse(x)

        else:
            self._precalc(x)
            return self._normalize(x)

    def _precalc(self, x: torch.Tensor) -> None:
        xT = x.permute(2, 0, 1)
        ts = x.shape[1] - 1
        theta = xT.bmm(self._reduce_layer).permute(1, 2, 0)

        if self._activate:
            theta = nn.functional.gelu(theta)

        self._phil, self._phih = theta.split(1, dim=1)
        self._xil = torch.pow(x - self._phil, 2).sum(axis=1, keepdim=True) / ts
        self._xih = torch.pow(x - self._phih, 2).sum(axis=1, keepdim=True) / ts

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._phil) / torch.sqrt(self._xil + 1e-8)
        x = x.mul(self._a) + self._b
        return x

    def _inverse(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._b) / self._a * torch.sqrt(self._xih + 1e-8) + self._phih
        return x

from decimal import Decimal
from os import listdir
from pathlib import Path
from typing import Iterable
import statistics
import numpy as np
import tap
from tqdm import tqdm
import pandas as pd
import pandas as pd


class Arguments(tap.Tap):
    dataset: Path = Path("dataset/extracted")
    savedir: Path = Path("dataset/beta")
    market: Path = Path("WIG-rates.csv")


def beta(cr: list[Decimal], mr: list[Decimal]) -> Decimal:
    return statistics.covariance(cr, mr) / statistics.variance(mr)


def walking(iterable: Iterable, window: int = 60):
    it = iter(iterable)

    initial = [e for _, e in zip(range(window), it)]
    yield initial

    try:
        while True:
            n = next(it)
            initial = initial[1:] + [n]
            yield initial

    except StopIteration:
        pass


def main(dataset: Path, savedir: Path, market: pd.DataFrame):
    savedir.mkdir(exist_ok=True)

    for filename in tqdm(listdir(dataset)):
        csv = pd.read_csv(dataset / filename, header=None)
        rates = np.array(csv[1])

        csv.rolling(60, step=7)

        for cr in walking(rates, 60):
            pass


if __name__ == "__main__":
    args = Arguments().parse_args()
    market = pd.read_csv(args.market)
    main(args.dataset, args.savedir, market)

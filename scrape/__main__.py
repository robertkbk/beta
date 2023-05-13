import requests
import tap
from tqdm import tqdm
from pathlib import Path


# https://stooq.pl/q/t
BASE_URL = "https://stooq.pl/q/d/l/"
INVALID_DATA = "Brak danych"


class Arguments(tap.Tap):
    dataset: Path = Path("dataset")
    symbols: Path = Path("scrape/symbols.txt")


def scrape(symbol: str) -> str:
    params = {"s": symbol.strip(), "i": "d"}
    res = requests.get(BASE_URL, params=params)

    if res.text.strip() == INVALID_DATA:
        return None

    return res.text


def save(dataset: Path, symbol: str, data: str):
    with open(dataset / f"{symbol}.csv", "w") as file:
        file.writelines(line.strip() for line in data.splitlines())


def main(dataset: Path, symbols: Path):
    dataset.mkdir(exist_ok=True)

    with open(symbols, "r") as file:
        for symbol in tqdm(file.readlines()):
            data = scrape(symbol.strip())

            if data is not None:
                save(dataset, symbol.strip(), data)


if __name__ == "__main__":
    args = Arguments()
    main(args.dataset, args.symbols)

import pandas as pd
from pathlib import Path
import tap
from tqdm import tqdm

COLUMNS = ["Data", "Otwarcie", "Najwyzszy", "Najnizszy", "Zamkniecie", "Wolumen"]


class Arguments(tap.Tap):
    dataset: Path = Path("dataset/stripped")
    savedir: Path = Path("dataset/beta")
    marketpath: Path = Path("WIG-rates.csv")


def rates(frame: pd.DataFrame):
    return pd.DataFrame.from_records(
        zip(
            frame["Data"],
            (frame["Zamkniecie"] - frame["Otwarcie"]) / frame["Otwarcie"] * 100,
        ),
        columns=["Date", "Rate"],
    )


def main(dataset: Path, savedir: Path, marketpath: Path):
    savedir.mkdir(exist_ok=True)
    market = pd.read_csv(marketpath, parse_dates=["Date"])

    for filename in tqdm([*dataset.iterdir()]):
        data = rates(pd.read_csv(filename, parse_dates=["Data"]))
        joined = data.merge(market, "inner", on="Date", suffixes=("", "Market"))

        cov = (
            joined[["Rate", "RateMarket"]]
            .rolling(60)
            .cov()
            .unstack()["Rate"]["RateMarket"]
        )

        var = joined["RateMarket"].to_frame().rolling(60).var()
        out = data["Date"].to_frame()
        out["Beta"] = (cov / var.T).T
        out.dropna(inplace=True)
        out.to_csv(savedir / filename.name, index=False)


if __name__ == "__main__":
    args = Arguments()
    main(args.dataset, args.savedir, args.marketpath)

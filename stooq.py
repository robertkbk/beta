import requests

"""
Params:
    s - stock code
    i - interval (d, w, m, q, y)
    d1/d2 - start/end date in YYYYMMDD format
    o - options; additional filters in binary mask format (all disabled by default)
"""
params = {
    "s": "CDR",
    "i": "d"
}

"""
Data format:    CSV

Columns:
    Date (YYYY-MM-DD)
    Open
    Min
    Max
    Close
    Volume

Exceptions:
    When there's no data or invalid parameter "Brak danych" string is returned
"""
res = requests.get("https://stooq.pl/q/d/l", params=params)

lines = res.text.splitlines()[:5]
text = "\n".join(lines)
print(text)

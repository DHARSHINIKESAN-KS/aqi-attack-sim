import pandas as pd, numpy as np

def load_clean(path="data/AirQuality.csv"):
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.loc[:, df.columns.str.contains(r"[A-Za-z]") & ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(how="all")
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"].str.replace(".", ":"),
                                     format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").drop(columns=["Date", "Time"])
    df = df.drop(columns=["NMHC(GT)"])                 # ~90% missing
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace(-200, np.nan)                      # sentinel -> NaN
    df = df.dropna(subset=["CO(GT)"])                  # keep real CO readings only
    df = df.interpolate(limit_direction="both").ffill().bfill()
    return df                                          # raw units, no NaNs

def standardize(df):
    return (df - df.mean()) / df.std()

if __name__ == "__main__":
    d = load_clean()
    z = standardize(d)
    print(d.shape, "NaNs:", int(d.isna().sum().sum()))
    print("mean~0:", bool(np.allclose(z.mean(), 0)), "std~1:", bool(np.allclose(z.std(), 1)))
    d.to_csv("data/clean.csv")
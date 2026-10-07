import glob, os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from load_data import load_clean

def runs(s, R=5):                                   # naturally occurring stuck segments (>=R identical readings)
    s = s.dropna(); g = (s != s.shift()).cumsum(); sz = s.groupby(g).transform("size")
    return int(s[sz >= R].groupby(g[sz >= R]).ngroups), round(float((sz >= R).mean()), 4)
print("REAL stuck-sensor segments (no injection), runs >=5 identical readings:")
raw = pd.read_csv("data/AirQuality.csv", sep=";", decimal=",")["CO(GT)"].replace(-200, np.nan)
print(" UCI CO(GT):", runs(raw), "(segments, fraction of data)")
for f in sorted(glob.glob("data/cities/*_Air_Quality.csv")):
    print(f" {os.path.basename(f).split('_Air')[0]} CO:", runs(pd.read_csv(f)["CO"]))

d = load_clean(); k = int(.4*len(d)); others = [c for c in d.columns if c != "CO(GT)"]
reg = HistGradientBoostingRegressor(random_state=0).fit(d.iloc[:k][others], d.iloc[:k]["CO(GT)"])
r = pd.Series(d["CO(GT)"].to_numpy() - reg.predict(d[others]), index=d.index)
m = r.groupby(r.index.to_period("M")).agg(median="median", spread=lambda x: 1.4826*np.median(np.abs(x-np.median(x)))).round(3)
print("\nREAL sensor aging: monthly error of CO predicted from other sensors (model trained on first 40% of data)")
print(m.to_string())
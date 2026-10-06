import os, joblib, numpy as np, pandas as pd
from load_data import load_clean
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)

files = ["data/AirQuality.csv", "results/models.pkl", "results/results.csv",
         "results/fig_series.png", "results/fig_f1.png", "results/fig_alarm.png"]
for f in files: ok(os.path.exists(f), f"file exists: {f}")

d = load_clean()
ok(d.shape[1] == 12 and d.isna().sum().sum() == 0, f"clean data {d.shape}, no NaNs")
ok(not (d == -200).any().any(), "no -200 sentinels left")

M = joblib.load("results/models.pkl"); X = M["scaler"].transform(d)
ia = (M["iso"].predict(X) == -1).mean(); ae = (((M["ae"].predict(X) - X)**2).mean(axis=1) > M["thr"]).mean()
ok(abs(ia - .03) < .01, f"IsoForest clean alarm {ia:.3f} (~0.03)")
ok(abs(ae - .05) < .01, f"Autoencoder clean alarm {ae:.3f} (~0.05)")

R = pd.read_csv("results/results.csv")
ok(len(R) == 60, f"results rows {len(R)} (expect 60)")
ok(R[["precision", "recall", "f1", "alarm_rate"]].apply(lambda c: c.between(0, 1)).all().all(), "all metrics in [0,1]")
g = lambda a, i, m: R[(R.attack == a) & (R.intensity == i) & (R.method == m)].f1.iloc[0]
ok(g("spike", "10%", "IsoForest+Defense") > g("spike", "10%", "IsolationForest"), "defense improves IsoForest (spike 10%)")
ok(g("freeze", "10%", "Defense only") > g("freeze", "10%", "Autoencoder"), "defense beats Autoencoder on freeze 10%")
import glob, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import precision_recall_fscore_support as prf
from defense import defend

FEATS = ["CO", "NO2", "SO2", "O3", "PM2.5", "PM10"]
def inject(te, atk, frac, seed, sd):
    rng = np.random.default_rng(seed); n = len(te)
    d = te.copy(); x = d["CO"].to_numpy(copy=True); y = np.zeros(n, bool)
    if atk == "spike":
        i = rng.choice(n, int(frac*n), replace=False)
        x[i] += rng.choice([-1, 1], len(i))*rng.uniform(4, 8, len(i))*sd; y[i] = True
    else:
        while y.mean() < frac:
            L = int(rng.integers(24, 72)); s = int(rng.integers(0, n-L))
            if atk == "drift": x[s:s+L] += np.linspace(0, 3*sd, L)
            else: x[s:s+L] = x[s]
            y[s:s+L] = True
    d["CO"] = x; return d, y

rows = []
for f in sorted(glob.glob("data/cities/*_Air_Quality.csv")):
    city = os.path.basename(f).replace("_Air_Quality.csv", "")
    df = pd.read_csv(f)[FEATS].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")
    k = int(.6*len(df)); tr, te = df.iloc[:k], df.iloc[k:]; sd = tr["CO"].std()
    sc = StandardScaler().fit(tr); Xtr = sc.transform(tr)
    iso = IsolationForest(n_estimators=200, contamination=.03, random_state=42).fit(Xtr)
    ae = MLPRegressor(hidden_layer_sizes=(4, 2, 4), max_iter=300, random_state=42).fit(Xtr, Xtr)
    thr = np.percentile(((ae.predict(Xtr)-Xtr)**2).mean(1), 95)
    for atk in ["spike", "drift", "freeze"]:
        for seed in range(3):
            d, y = inject(te, atk, .10, seed, sd)
            fl = defend(d["CO"].reset_index(drop=True))[0].to_numpy(); X = sc.transform(d)
            F = {"IsolationForest": iso.predict(X) == -1, "Autoencoder": ((ae.predict(X)-X)**2).mean(1) > thr, "Defense (ours)": fl}
            for m, p in F.items():
                rows.append(dict(city=city, attack=atk, method=m, f1=prf(y, p, average="binary", zero_division=0)[2]))
R = pd.DataFrame(rows)
T = R.groupby(["city", "attack", "method"]).f1.mean().unstack("method").round(3)
T.to_csv("results/multicity.csv"); print(T.to_string())
print("\nMean F1 across all cities and attacks:"); print(R.groupby("method").f1.mean().round(3))
print("\nBy attack:"); print(R.groupby(["attack", "method"]).f1.mean().unstack().round(3))
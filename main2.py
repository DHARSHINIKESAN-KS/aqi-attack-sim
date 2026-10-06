import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import precision_recall_fscore_support as prf
from load_data import load_clean
from defense import defend

df0 = load_clean(); k = int(.6*len(df0)); k0 = int(.4*len(df0))
tr, va, te = df0.iloc[:k0], df0.iloc[k0:k], df0.iloc[k:]   # chronological: fit / validation / test
trv = df0.iloc[:k]
sd = trv["CO(GT)"].std(); others = [c for c in df0.columns if c != "CO(GT)"]
sc = StandardScaler().fit(trv); Xtr = sc.transform(trv)
iso = IsolationForest(n_estimators=200, contamination=.03, random_state=42).fit(Xtr)
ae = MLPRegressor(hidden_layer_sizes=(8, 3, 8), max_iter=300, random_state=42).fit(Xtr, Xtr)
thr_ae = np.percentile(((ae.predict(Xtr)-Xtr)**2).mean(1), 95)
reg = HistGradientBoostingRegressor(random_state=0).fit(tr[others], tr["CO(GT)"])   # cross-sensor model
def resid(d):
    r = pd.Series(d["CO(GT)"].to_numpy() - reg.predict(d[others]))
    return (r - r.rolling(168, min_periods=24).median().shift(1)).fillna(0).to_numpy()   # adaptive to sensor aging
rtr = resid(va)   # thresholds calibrated on held-out validation residuals
mad = 1.4826*np.median(np.abs(rtr-np.median(rtr)))
rm = lambda r: pd.Series(r).rolling(12, min_periods=1).median().to_numpy()
thr_pt = np.percentile(np.abs(rtr)/mad, 99.5); thr_rm = np.percentile(np.abs(rm(rtr)), 99)

def cross(d):
    r = resid(d)
    return (np.abs(r)/mad > thr_pt) | (np.abs(rm(r)) > thr_rm)
def cross_drift(d): return np.abs(rm(resid(d))) > thr_rm   # slow-bias check only

def inject(atk, frac, seed):
    rng = np.random.default_rng(seed); n = len(te)
    d = te.copy(); x = d["CO(GT)"].to_numpy(copy=True); y = np.zeros(n, bool)
    if atk == "spike":
        i = rng.choice(n, int(frac*n), replace=False)
        x[i] += rng.choice([-1, 1], len(i))*rng.uniform(4, 8, len(i))*sd; y[i] = True
    else:
        while y.mean() < frac:
            L = int(rng.integers(24, 72)); s = int(rng.integers(0, n-L))
            if atk == "drift": x[s:s+L] += np.linspace(0, 3*sd, L)
            else: x[s:s+L] = x[s]
            y[s:s+L] = True
    d["CO(GT)"] = x; return d, y

rows = []
for atk in ["spike", "drift", "freeze"]:
    for frac in [.05, .10, .20]:
        for seed in range(5):
            d, y = inject(atk, frac, seed)
            fl, rep = defend(d["CO(GT)"].reset_index(drop=True)); fl = fl.to_numpy()
            fl0 = defend(d["CO(GT)"].reset_index(drop=True), drift_k=1e9)[0].to_numpy()
            X = sc.transform(d); cr = cross(d)
            F = {"IsolationForest": iso.predict(X) == -1,
                 "Autoencoder": ((ae.predict(X)-X)**2).mean(1) > thr_ae,
                 "Defense (stat)": fl, "CrossSensor": cr, "Proposed (Stat+CrossDrift)": fl0 | cross_drift(d)}
            for m, f in F.items():
                p, r, f1, _ = prf(y, f, average="binary", zero_division=0)
                rows.append(dict(attack=atk, intensity=f"{int(frac*100)}%", method=m, p=p, r=r, f1=f1))
R = pd.DataFrame(rows)
S = R.groupby(["attack", "intensity", "method"]).agg(P=("p", "mean"), R=("r", "mean"), F1=("f1", "mean"), F1_std=("f1", "std")).round(3).reset_index()
S.to_csv("results/results_v2.csv", index=False)
print(S[S.intensity == "10%"].to_string(index=False))
print("\nMean F1 over all settings:"); print(R.groupby("method").f1.mean().round(3).sort_values())
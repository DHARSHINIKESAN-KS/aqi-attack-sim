import time, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as prf
from load_data import load_clean
from defense import defend

M = joblib.load("results/models.pkl")
scaler, iso, ae, thr = M["scaler"], M["iso"], M["ae"], M["thr"]
clean = load_clean(); n = len(clean); sd = clean["CO(GT)"].std()

def inject(atk, frac, seed=0):
    rng = np.random.default_rng(seed)
    df = clean.copy(); x = df["CO(GT)"].to_numpy(copy=True); y = np.zeros(n, bool)
    if atk == "spike":
        i = rng.choice(n, int(frac*n), replace=False)
        x[i] += rng.choice([-1, 1], len(i)) * rng.uniform(4, 8, len(i)) * sd; y[i] = True
    else:
        while y.mean() < frac:
            L = int(rng.integers(24, 72)); s = int(rng.integers(0, n-L))
            if atk == "drift": x[s:s+L] += np.linspace(0, 3*sd, L)
            else: x[s:s+L] = x[s]
            y[s:s+L] = True
    df["CO(GT)"] = x
    return df, y

def det_iso(df): return iso.predict(scaler.transform(df)) == -1
def det_ae(df):
    X = scaler.transform(df); return ((ae.predict(X) - X)**2).mean(axis=1) > thr

rows, ts = [], {}
def add(atk, frac, name, flags, y):
    p, r, f, _ = prf(y, flags, average="binary", zero_division=0)
    rows.append(dict(attack=atk, intensity=f"{int(frac*100)}%", method=name,
                     precision=round(p,3), recall=round(r,3), f1=round(f,3), alarm_rate=round(flags.mean(),3)))

for atk in ["spike", "drift", "freeze"]:
    for frac in [.05, .10, .20, .30]:
        df, y = inject(atk, frac)
        fl, rep = defend(df["CO(GT)"].reset_index(drop=True))
        fl = fl.to_numpy(); dfr = df.copy(); dfr["CO(GT)"] = rep.to_numpy()
        add(atk, frac, "IsolationForest", det_iso(df), y)
        add(atk, frac, "Autoencoder", det_ae(df), y)
        add(atk, frac, "Defense only", fl, y)
        add(atk, frac, "IsoForest+Defense", fl | det_iso(dfr), y)
        add(atk, frac, "Autoencoder+Defense", fl | det_ae(dfr), y)
        if frac == .10: ts[atk] = (df["CO(GT)"].to_numpy(), rep.to_numpy(), y)

R = pd.DataFrame(rows); R.to_csv("results/results.csv", index=False)
print(R.to_string(index=False))

# per-sample time
df, _ = inject("spike", .1)
for name, fn in [("IsolationForest", lambda: det_iso(df)), ("Autoencoder", lambda: det_ae(df)),
                 ("Defense", lambda: defend(df["CO(GT)"].reset_index(drop=True)))]:
    t = time.perf_counter(); fn(); print(f"{name}: {(time.perf_counter()-t)/n*1000:.4f} ms/sample")

# Plot 1: clean vs attacked vs repaired
fig, ax = plt.subplots(3, 1, figsize=(10, 8))
c = clean["CO(GT)"].to_numpy(); s = slice(500, 700)
for a, k in zip(ax, ts):
    att, rep, _ = ts[k]
    a.plot(c[s], label="clean", color="green"); a.plot(att[s], label="attacked", color="red", alpha=.6)
    a.plot(rep[s], label="repaired", color="blue", ls="--"); a.set_title(f"{k} (10%)"); a.set_ylabel("CO(GT)")
ax[0].legend(); plt.tight_layout(); plt.savefig("results/fig_series.png", dpi=120); plt.close()

# Plot 2: F1 at 10%
P = R[R.intensity == "10%"].pivot(index="attack", columns="method", values="f1")
P.plot.bar(figsize=(10, 5), rot=0, title="F1 at 10% attack: before vs after defense")
plt.ylabel("F1"); plt.tight_layout(); plt.savefig("results/fig_f1.png", dpi=120); plt.close()

# Plot 3: alarm rate vs intensity (spike)
Q = R[R.attack == "spike"].pivot(index="intensity", columns="method", values="alarm_rate").loc[["5%","10%","20%","30%"]]
Q.plot(marker="o", figsize=(8, 5), title="Alarm rate vs attack intensity (spike)")
plt.ylabel("alarm rate"); plt.tight_layout(); plt.savefig("results/fig_alarm.png", dpi=120); plt.close()
print("Saved results/results.csv + 3 figures in results/")
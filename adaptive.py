import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from load_data import load_clean
from defense import defend

df0 = load_clean(); k = int(.6*len(df0)); k0 = int(.4*len(df0))
tr, va, te = df0.iloc[:k0], df0.iloc[k0:k], df0.iloc[k:]
sd = df0.iloc[:k]["CO(GT)"].std(); others = [c for c in df0.columns if c != "CO(GT)"]
reg = HistGradientBoostingRegressor(random_state=0).fit(tr[others], tr["CO(GT)"])
def resid(d):
    r = pd.Series(d["CO(GT)"].to_numpy() - reg.predict(d[others]))
    return (r - r.rolling(168, min_periods=24).median().shift(1)).fillna(0).to_numpy()
rm = lambda r: pd.Series(r).rolling(12, min_periods=1).median().to_numpy()
rv = resid(va); thr_rm = np.percentile(np.abs(rm(rv)), 99)
cross = lambda d: np.abs(rm(resid(d))) > thr_rm      # cross-sensor consistency check
stat = lambda d: defend(d["CO(GT)"].reset_index(drop=True))[0].to_numpy()

def stealth_attack(M, seed, frac=.10):
    """Adaptive low-and-slow attacker: constant offset of M*sd (small M = stealthier)."""
    rng = np.random.default_rng(seed); n = len(te)
    d = te.copy(); x = d["CO(GT)"].to_numpy(copy=True); y = np.zeros(n, bool)
    while y.mean() < frac:
        L = int(rng.integers(24, 72)); s = int(rng.integers(0, n-L)); x[s:s+L] += M*sd; y[s:s+L] = True
    d["CO(GT)"] = x; return d, y

clean_fpr = {"Statistical": stat(te).mean(), "Cross-sensor": cross(te).mean(), "Combined": (stat(te) | cross(te)).mean()}
print("False-alarm rate on clean test data:", {k_: round(float(v), 3) for k_, v in clean_fpr.items()})
rows = []
for M in [0.25, 0.5, 0.75, 1, 1.5, 2, 3]:
    for seed in range(5):
        d, y = stealth_attack(M, seed); s_, c_ = stat(d), cross(d)
        for name, f in [("Statistical", s_), ("Cross-sensor", c_), ("Combined", s_ | c_)]:
            rec = (f & y).sum()/y.sum()
            rows.append(dict(bias_sd=M, method=name, recall=rec, undetected_impact_sd=(1-rec)*M))
R = pd.DataFrame(rows).groupby(["bias_sd", "method"]).mean().round(3).reset_index()
R.to_csv("results/adaptive.csv", index=False)
P = R.pivot(index="bias_sd", columns="method", values="recall"); print("\nRecall vs attacker bias:"); print(P)
print("\nUndetected impact (bias still reaching the system, in std units):")
print(R.pivot(index="bias_sd", columns="method", values="undetected_impact_sd"))
P.plot(marker="o", figsize=(8, 5), title="Stealthy attacker: detection vs injected bias"); plt.xlabel("Injected bias (x sensor std)"); plt.ylabel("Recall")
plt.tight_layout(); plt.savefig("results/fig_adaptive.png", dpi=120)
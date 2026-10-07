import numpy as np, pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score
exec(open("adaptive.py").read().split("clean_fpr =")[0])   # reuse data, models, stealth attacker

# ---- naive baselines (fit on training data only) ----
x_tr = tr["CO(GT)"]; med0 = x_tr.median(); mad0 = 1.4826*np.median(np.abs(x_tr-med0))
thresh = lambda d: np.abs(d["CO(GT)"].to_numpy()-med0) > 3*mad0                 # fixed 3-sigma threshold
ma_res = lambda s: s - s.rolling(24, min_periods=3).mean().shift(1)
ma_sd = ma_res(x_tr).std()
movavg = lambda d: (ma_res(d["CO(GT)"]).abs() > 3*ma_sd).fillna(False).to_numpy()  # moving-average filter
DET = {"Fixed threshold": thresh, "Moving average": movavg, "Ours (statistical)": stat, "Cross-sensor": cross}

def attack(atk, frac, seed):
    rng = np.random.default_rng(seed); n = len(te)
    d = te.copy(); x = d["CO(GT)"].to_numpy(copy=True); y = np.zeros(n, bool)
    if atk == "spike":
        i = rng.choice(n, int(frac*n), replace=False); x[i] += rng.choice([-1, 1], len(i))*rng.uniform(4, 8, len(i))*sd; y[i] = True
    else:
        while y.mean() < frac:
            L = int(rng.integers(24, 72)); s = int(rng.integers(0, n-L))
            if atk == "drift": x[s:s+L] += np.linspace(0, 3*sd, L)
            else: x[s:s+L] = x[s]
            y[s:s+L] = True
    d["CO(GT)"] = x; return d, y

# ---- 1+2: baselines and paired t-test over 10 seeds (10% attacks) ----
F = {(a, m): [] for a in ["spike", "drift", "freeze"] for m in DET}
for a in ["spike", "drift", "freeze"]:
    for seed in range(10):
        d, y = attack(a, .10, seed)
        for m, f in DET.items(): F[(a, m)].append(f1_score(y, f(d), zero_division=0))
rows = []
for a in ["spike", "drift", "freeze"]:
    for m in DET:
        if m == "Ours (statistical)": continue
        t, p = ttest_rel(F[(a, "Ours (statistical)")], F[(a, m)])
        rows.append(dict(attack=a, vs=m, ours_F1=round(np.mean(F[(a, "Ours (statistical)")]), 3), other_F1=round(np.mean(F[(a, m)]), 3),
                         p_value=float(f"{p:.2g}"), ours_better_significant=bool(p < .05 and np.mean(F[(a, 'Ours (statistical)')]) > np.mean(F[(a, m)]))))
T = pd.DataFrame(rows); T.to_csv("results/significance.csv", index=False); print(T.to_string(index=False))

# ---- 3: threshold-aware attacker: finds the bias with maximum undetected impact for each detector ----
print("\nThreshold-aware attacker: best bias it can slip through (std units), clean false-alarm rate")
for m in ["Ours (statistical)", "Cross-sensor"]:
    best = (0, 0)
    for M in np.arange(.1, 3.01, .1):
        imp = np.mean([(1-(((f:=DET[m](d)) & y).sum()/y.sum()))*M for d, y in [stealth_attack(M, s) for s in range(3)]])
        if imp > best[0]: best = (imp, M)
    print(f"{m}: max undetected impact {best[0]:.2f} sd (attacker's best bias = {best[1]:.1f} sd), clean false alarms {DET[m](te).mean():.3f}")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("adaptive.py").read().split("clean_fpr =")[0])   # reuse data, models, attacker

def persist(flag, P):                       # keep flags only if on for >= P consecutive hours
    s = pd.Series(flag.astype(int)); g = (s != s.shift()).cumsum()
    return (s.groupby(g).transform("size").to_numpy() >= P) & flag

rows = []
for q in [95, 97, 98, 99, 99.5, 99.9]:
    thr = np.percentile(np.abs(rm(rv)), q)
    for P in [1, 3, 6, 12]:
        det = lambda d: persist(np.abs(rm(resid(d))) > thr, P)
        fpr = det(te).mean(); rec = []
        for seed in range(5):
            d, y = stealth_attack(1.0, seed); rec.append((det(d) & y).sum()/y.sum())
        rows.append(dict(percentile=q, persist_hours=P, false_alarm=round(fpr, 3), recall_at_1sd=round(np.mean(rec), 3)))
T = pd.DataFrame(rows); T.to_csv("results/tradeoff.csv", index=False)
print(T.to_string(index=False))
ok = T[T.false_alarm <= 0.03].sort_values("recall_at_1sd", ascending=False)
print("\nBest setting with false alarms <= 3%:"); print(ok.head(3).to_string(index=False))
for P, g in T.groupby("persist_hours"): plt.plot(g.false_alarm, g.recall_at_1sd, marker="o", label=f"persist {P}h")
plt.xlabel("False-alarm rate (clean data)"); plt.ylabel("Recall at 1σ stealth bias"); plt.legend(); plt.title("Detection vs false-alarm trade-off")
plt.tight_layout(); plt.savefig("results/fig_tradeoff.png", dpi=120)
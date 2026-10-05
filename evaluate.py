import numpy as np, pandas as pd, joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from load_data import load_clean
from attacks import inject

m = joblib.load("results/models.pkl")
clean = load_clean()
rows = []
for kind in ["spike", "drift", "freeze"]:
    for frac in [0.05, 0.10, 0.20, 0.30]:
        att, y = inject(clean, kind, frac)
        X = m["scaler"].transform(att)
        p_iso = (m["iso"].predict(X) == -1).astype(int)
        p_ae = (((m["ae"].predict(X) - X) ** 2).mean(axis=1) > m["thr"]).astype(int)
        for name, p in [("IsolationForest", p_iso), ("Autoencoder", p_ae)]:
            rows.append(dict(attack=kind, intensity=f"{int(frac*100)}%", detector=name,
                precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p, zero_division=0),
                f1=f1_score(y, p, zero_division=0), alarm_rate=p.mean()))
r = pd.DataFrame(rows).round(3)
r.to_csv("results/degradation.csv", index=False)
print(r.to_string(index=False))
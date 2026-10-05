import numpy as np, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from load_data import load_clean

df = load_clean()
scaler = StandardScaler().fit(df)
X = scaler.transform(df)

iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42).fit(X)

# Autoencoder 12->8->3->8->12 (hidden layers 8,3,8)
ae = MLPRegressor(hidden_layer_sizes=(8, 3, 8), activation="relu",
                  max_iter=300, random_state=42).fit(X, X)
err = ((ae.predict(X) - X) ** 2).mean(axis=1)
thr = np.percentile(err, 95)          # ~5% clean alarm rate

print("IsoForest clean alarm rate:", (iso.predict(X) == -1).mean().round(3))
print("Autoencoder clean alarm rate:", (err > thr).mean().round(3))
joblib.dump({"scaler": scaler, "iso": iso, "ae": ae, "thr": thr}, "results/models.pkl")
print("saved results/models.pkl")
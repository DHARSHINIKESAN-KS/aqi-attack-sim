import numpy as np

def inject(df, kind, frac, col="CO(GT)", seed=42):
    rng = np.random.default_rng(seed)
    d = df.copy(); n = len(d); k = int(n * frac)
    y = np.zeros(n, dtype=int); v = d[col].values.copy(); sd = v.std()
    if kind == "spike":
        idx = rng.choice(n, k, replace=False)
        v[idx] += rng.choice([-1, 1], k) * rng.uniform(4, 8, k) * sd
        y[idx] = 1
    else:  # contiguous window (drift / freeze)
        s = rng.integers(0, n - k); e = s + k
        if kind == "drift":
            v[s:e] += np.linspace(0, 4 * sd, k)       # slow ramp to +4 sigma
        else:
            v[s:e] = v[s]                              # freeze
        y[s:e] = 1
    d[col] = v
    return d, y
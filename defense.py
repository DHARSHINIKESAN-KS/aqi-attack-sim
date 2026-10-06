import numpy as np, pandas as pd

def defend(x, w_s=24, w_l=168, z_thr=4.0, roc_k=6.0, run_len=5, drift_k=2.0):
    """x: pd.Series (sensor stream). Returns (flags: bool Series, repaired: Series)."""
    x = pd.Series(np.asarray(x, float), index=getattr(x, "index", None))
    mad = lambda s: 1.4826 * np.median(np.abs(s - np.median(s)))
    # 1) robust Z-score vs rolling median (past window)
    med = x.rolling(w_s, min_periods=3).median().shift(1)
    res = x - med
    z = res.abs() / (mad(res.dropna()) + 1e-9)
    f_z = z > z_thr
    # 2) rate of change
    d = x.diff().abs()
    f_roc = (d > roc_k * (mad(x.diff().dropna()) + 1e-9)) & (z > z_thr / 2)
    # 3) stuck-value run
    grp = (x != x.shift()).cumsum()
    f_run = x.groupby(grp).transform("size") >= run_len
    # 4) short-vs-long baseline divergence (drift), on hour-of-day-adjusted residual
    r = x - x.groupby(np.arange(len(x)) % 24).transform("median")
    lng = r.rolling(w_l, min_periods=24).median()
    div = (r.rolling(12, min_periods=6).mean() - lng).abs()
    f_dr = div > drift_k * (mad((r - lng).dropna()) + 1e-9)
    flags = (f_z | f_roc | f_run | f_dr).fillna(False)
    # repair: drop flagged points, interpolate from clean neighbours
    rep = x.mask(flags).interpolate(limit_direction="both")
    return flags, rep

if __name__ == "__main__":   # quick self-test
    from load_data import load_clean
    rng = np.random.default_rng(0)
    clean = load_clean()["CO(GT)"]
    n, sd = len(clean), clean.std()
    f, _ = defend(clean); print("clean false-flag rate:", round(f.mean(), 3))
    for atk in ["spike", "drift", "freeze"]:
        x, y = clean.to_numpy(copy=True), np.zeros(n, bool)
        if atk == "spike":
            i = rng.choice(n, int(.1*n), replace=False)
            x[i] += rng.choice([-1, 1], len(i)) * rng.uniform(4, 8, len(i)) * sd; y[i] = True
        else:
            tot = 0
            while tot < .1*n:
                L = int(rng.integers(24, 72)); s = int(rng.integers(0, n-L))
                if atk == "drift": x[s:s+L] += np.linspace(0, 3*sd, L)
                else: x[s:s+L] = x[s]
                y[s:s+L] = True; tot += L
        fl, rep = defend(pd.Series(x))
        rec = (fl & y).sum()/y.sum(); prec = (fl & y).sum()/max(fl.sum(), 1)
        err_b = np.abs(x-clean.values)[y].mean(); err_a = np.abs(rep.values-clean.values)[y].mean()
        print(f"{atk}: recall {rec:.2f} precision {prec:.2f} | error on attacked pts {err_b:.2f} -> {err_a:.2f}")
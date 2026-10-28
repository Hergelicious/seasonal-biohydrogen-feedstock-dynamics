"""Mechanistic test: can temperature alone reproduce the seasonal BOD-load deficit at Ismailia?

Observed weekly log load  y = year effect + Ramadan effect + log(1 - L) + e
In-sewer loss of BOD      L = f * (1 - exp(-D))
Removal number            D = D20 * theta^(T_x - 20)

M0 (kinetics only):       T_x = sewage temperature now (reaction rates respond at once)
M1 (biomass memory):      T_x = sewage temperature smoothed with time constant tau
                          (active biofilm/sediment biomass builds and decays slowly)
Sewage temperature comes from ERA5 air temperature through a transfer fitted on the
plant record in its reliable years (2000-2003). Benchmarks: no season (E0) and an
empirical two-harmonic season (E1).
"""
import json, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
OUT = {}

e = pd.read_csv("data/era5_openmeteo.csv", skiprows=3); e.columns = ["date", "Ta"]
e["date"] = pd.to_datetime(e.date); e = e.set_index("date").asfreq("D")
# 1-year ERA5 spin-up is not available before 2000; start state uses the first 120 days (handled by burn-in)
d = pd.read_csv("data/ismailia.csv"); d["date"] = pd.to_datetime(d.SampleDate, format="%d-%m-%y")
u = d[d.Stage == 1].set_index("date").sort_index()[["BOD5_inf", "Flow", "Temp"]].asfreq("D").join(e)
u["load"] = u.BOD5_inf * u.Flow / 1000
RAM = {2000: ("2000-11-27", "2000-12-26"), 2001: ("2001-11-16", "2001-12-15"), 2002: ("2002-11-06", "2002-12-05"),
       2003: ("2003-10-27", "2003-11-24"), 2004: ("2004-10-15", "2004-11-13"), 2005: ("2005-10-04", "2005-11-02"),
       2006: ("2006-09-24", "2006-10-23"), 2007: ("2007-09-13", "2007-10-12"), 2008: ("2008-09-01", "2008-09-30")}
u["ram"] = 0.0
for a, b in RAM.values(): u.loc[a:b, "ram"] = 1.0

# ---- 1. Sewage temperature from air: choose the air-smoothing window that best fits the plant in 2000-2003
best = None; win_fit = []
for W in (1, 7, 14, 21, 30, 45, 60, 90):
    s = u.Ta.ewm(span=W, adjust=False).mean() if W > 1 else u.Ta
    g = pd.DataFrame({"T": u.Temp, "A": s}).loc["2000":"2003"].dropna()
    m = smf.ols("T ~ A", data=g).fit(); win_fit.append({"span_d": W, "r2": m.rsquared, "a": m.params["Intercept"], "b": m.params["A"]})
    if best is None or m.rsquared > best[1]: best = (W, m.rsquared, m.params["Intercept"], m.params["A"])
OUT["sewage_temp_transfer"] = {"fits": win_fit, "chosen_span_d": best[0], "r2": best[1], "a": best[2], "b": best[3]}
W, _, a0, b0 = best
u["Ts"] = a0 + b0 * u.Ta.ewm(span=W, adjust=False).mean()
OUT["Ts_range"] = [float(u.Ts.min()), float(u.Ts.max())]

def ema(x, tau):
    """first-order lag (biomass memory), daily step; tau in days"""
    if tau <= 0.5: return x.copy()
    a = 1.0 / tau
    y, _ = lfilter([a], [1, -(1 - a)], x, zi=[(1 - a) * x[:120].mean()])
    return y
Ts = u.Ts.values

# ---- 2. Weekly data (reduces day-to-day autocorrelation); burn-in: drop first 120 days of 2000
wk_idx = u.index.to_period("W")
def weekly(series):
    return pd.Series(series, index=u.index).groupby(wk_idx).mean()
obs = u.load.notna().values
W_load = u.load.groupby(wk_idx).mean(); W_n = u.load.groupby(wk_idx).count()
W_ram = u.ram.groupby(wk_idx).mean()
keep = (W_n >= 4) & (W_load.index.start_time >= pd.Timestamp("2000-05-01"))
y = np.log(W_load[keep].values); ram = W_ram[keep].values
yr = W_load[keep].index.start_time.year.values
wdates = W_load[keep].index.start_time
doy = wdates.dayofyear.values
YR = pd.get_dummies(yr).values.astype(float)
Xbase = np.column_stack([YR, ram])
H = np.column_stack([np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25), np.sin(4*np.pi*doy/365.25), np.cos(4*np.pi*doy/365.25)])
n = len(y); OUT["n_weeks"] = int(n)
codes, uniq = pd.factorize(wk_idx, sort=True); gid = codes; G = len(uniq); cnt = np.bincount(gid, minlength=G).astype(float)
pos = {p: i for i, p in enumerate(uniq)}; kidx = np.array([pos[p] for p in W_load[keep].index])

def ols_sse(yv, X):
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None); r = yv - X @ beta; return float(r @ r), beta, r
sse_E0, _, _ = ols_sse(y, Xbase)
sse_E1, bE1, rE1 = ols_sse(y, np.column_stack([Xbase, H]))
def aic(sse, k): return n * np.log(sse / n) + 2 * k
k0 = Xbase.shape[1]

def loss_series(D20, theta, tau, f):
    Tx = ema(Ts, tau)
    D = D20 * theta ** (Tx - 20.0)
    L = f * (1 - np.exp(-D))
    z = np.log(1 - L)
    zw = (np.bincount(gid, weights=z, minlength=G) / cnt)[kidx]
    Lw = (np.bincount(gid, weights=L, minlength=G) / cnt)[kidx]
    return zw, Lw

def fit_model(tau_fixed=None, f=0.6, memory=True):
    # params: log D20, theta, log tau
    def obj(p):
        D20 = np.exp(p[0]); th = p[1]
        tau = tau_fixed if tau_fixed is not None else (np.exp(p[2]) if memory else 0.0)
        if not (1.0 < th < 1.25): return 1e9
        z, _ = loss_series(D20, th, tau, f)
        sse, _, _ = ols_sse(y - z, Xbase); return sse
    best = None
    taus0 = [np.log(40.0)] if (memory and tau_fixed is None) else [0.0]
    for D0 in (0.2, 0.6, 1.5):
        for th0 in (1.04, 1.08, 1.12):
            for t0 in taus0:
                p0 = [np.log(D0), th0] + ([t0] if (memory and tau_fixed is None) else [])
                r = minimize(obj, p0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 3000})
                if best is None or r.fun < best.fun: best = r
    p = best.x; D20 = float(np.exp(p[0])); th = float(p[1])
    tau = float(tau_fixed if tau_fixed is not None else (np.exp(p[2]) if memory else 0.0))
    z, Lw = loss_series(D20, th, tau, f)
    return {"D20": D20, "theta": th, "tau": tau, "f": f, "sse": float(best.fun), "meanL": float(Lw.mean()),
            "L_range": [float(Lw.min()), float(Lw.max())]}

res = {}
for f in (0.4, 0.6, 0.8):
    m0 = fit_model(f=f, memory=False); m1 = fit_model(f=f, memory=True)
    for nm, m, k in (("M0", m0, 2), ("M1", m1, 3)):
        m["AIC"] = float(aic(m["sse"], k0 + k)); m["season_explained"] = float((sse_E0 - m["sse"]) / (sse_E0 - sse_E1))
        res[f"{nm}_f{f}"] = m
OUT["benchmarks"] = {"sse_E0": sse_E0, "sse_E1": sse_E1, "AIC_E0": float(aic(sse_E0, k0)), "AIC_E1": float(aic(sse_E1, k0 + 4))}
OUT["fits"] = res

# ---- 3. Profile over tau (f = 0.6), with D20 and theta re-fitted; and with theta held inside 1.03-1.10
prof = []
for tau in (0, 5, 10, 20, 30, 45, 60, 75, 90, 120, 150, 200):
    m = fit_model(tau_fixed=float(tau), f=0.6, memory=True) if tau > 0 else fit_model(f=0.6, memory=False)
    prof.append({"tau": tau, "theta": m["theta"], "D20": m["D20"], "sse": m["sse"], "meanL": m["meanL"],
                 "season_explained": float((sse_E0 - m["sse"]) / (sse_E0 - sse_E1))})
prof = pd.DataFrame(prof); prof["dAIC"] = n * np.log(prof.sse / prof.sse.min())
prof.to_csv("out/model_tau_profile.csv", index=False)
OUT["tau_profile"] = prof.round(4).to_dict(orient="list")

# Literature-constrained check: theta fixed at 1.05 and 1.07, tau free
lit = {}
for th in (1.05, 1.07):
    def obj(p, th=th):
        D20 = np.exp(p[0]); tau = np.exp(p[1]); z, _ = loss_series(D20, th, tau, 0.6)
        return ols_sse(y - z, Xbase)[0]
    best = None
    for D0 in (0.3, 1.0, 3.0):
        for t0 in (20.0, 60.0):
            r = minimize(obj, [np.log(D0), np.log(t0)], method="Nelder-Mead", options={"maxiter": 2000})
            if best is None or r.fun < best.fun: best = r
    D20, tau = np.exp(best.x)
    z, Lw = loss_series(D20, th, tau, 0.6)
    lit[str(th)] = {"D20": float(D20), "tau": float(tau), "sse": float(best.fun), "meanL": float(Lw.mean()),
                    "L_range": [float(Lw.min()), float(Lw.max())], "season_explained": float((sse_E0 - best.fun) / (sse_E0 - sse_E1))}
OUT["literature_theta"] = lit

# ---- 4. Out-of-sample: fit on 2000-2003, predict the seasonal shape of 2005 and 2008 (full years)
tr = yr <= 2003; te = np.isin(yr, [2005, 2008])
def fit_sub(mask, memory):
    def obj(p):
        D20 = np.exp(p[0]); th = p[1]; tau = np.exp(p[2]) if memory else 0.0
        if not (1.0 < th < 1.25): return 1e9
        z, _ = loss_series(D20, th, tau, 0.6)
        return ols_sse((y - z)[mask], Xbase[mask][:, Xbase[mask].std(0) > 0])[0]
    best = None
    for D0 in (0.3, 1.0):
        for th0 in (1.05, 1.10):
            p0 = [np.log(D0), th0] + ([np.log(40.0)] if memory else [])
            r = minimize(obj, p0, method="Nelder-Mead", options={"maxiter": 3000})
            if best is None or r.fun < best.fun: best = r
    p = best.x; return np.exp(p[0]), p[1], (np.exp(p[2]) if memory else 0.0)
oos = {}
for nm, memory in (("M0", False), ("M1", True)):
    D20, th, tau = fit_sub(tr, memory)
    z, _ = loss_series(D20, th, tau, 0.6)
    # test years: remove year means from both observed and predicted, compare seasonal anomalies
    pred, obsv = [], []
    for Y in (2005, 2008):
        mk = (yr == Y) & (ram < 0.5)
        pred.append(z[mk] - z[mk].mean()); obsv.append(y[mk] - y[mk].mean())
    pred = np.concatenate(pred); obsv = np.concatenate(obsv)
    oos[nm] = {"D20": float(D20), "theta": float(th), "tau": float(tau), "r": float(np.corrcoef(pred, obsv)[0, 1]),
               "rmse": float(np.sqrt(np.mean((obsv - pred) ** 2))), "rmse_null": float(np.sqrt(np.mean(obsv ** 2)))}
# harmonic benchmark trained on 2000-2003
Xh = np.column_stack([Xbase, H]); mk_tr = tr
cols = Xh[mk_tr].std(0) > 0
beta, *_ = np.linalg.lstsq(Xh[mk_tr][:, cols], y[mk_tr], rcond=None)
bh = beta[-4:]
pred, obsv = [], []
for Y in (2005, 2008):
    mk = (yr == Y) & (ram < 0.5); hp = H[mk] @ bh
    pred.append(hp - hp.mean()); obsv.append(y[mk] - y[mk].mean())
pred = np.concatenate(pred); obsv = np.concatenate(obsv)
oos["E1_harmonic"] = {"r": float(np.corrcoef(pred, obsv)[0, 1]), "rmse": float(np.sqrt(np.mean((obsv - pred) ** 2))),
                      "rmse_null": float(np.sqrt(np.mean(obsv ** 2)))}
OUT["out_of_sample_2005_2008"] = oos

# ---- 5. Seasonal shape for plotting: monthly climatology of observed (year- and Ramadan-adjusted) vs models
best0 = res["M0_f0.6"]; best1 = res["M1_f0.6"]
z0, L0 = loss_series(best0["D20"], best0["theta"], 0.0, 0.6)
z1, L1 = loss_series(best1["D20"], best1["theta"], best1["tau"], 0.6)
_, b_base, _ = ols_sse(y - z1, Xbase)
yadj = y - Xbase[:, :-1] @ b_base[:-1] - Xbase[:, -1] * b_base[-1]   # remove year + Ramadan effects (M1 fit)
mo = pd.DataFrame({"month": wdates.month, "obs": yadj, "M0": z0, "M1": z1, "E1": H @ bE1[-4:],
                   "Ts": pd.Series(Ts, index=u.index).groupby(wk_idx).mean()[keep].values})
clim = mo.groupby("month").mean()
for c in ("obs", "M0", "M1", "E1"): clim[c] = 100 * (np.exp(clim[c] - clim[c].mean()) - 1)
clim["obs_se"] = 100 * mo.groupby("month").obs.sem()
clim.to_csv("out/model_monthly.csv")
# timing of minima
OUT["trough_month"] = {c: int(clim[c].idxmin()) for c in ("obs", "M0", "M1", "E1")}
OUT["peak_month"] = {c: int(clim[c].idxmax()) for c in ("obs", "M0", "M1", "E1")}
OUT["amplitude_pct"] = {c: float(clim[c].max() - clim[c].min()) for c in ("obs", "M0", "M1", "E1")}
# hysteresis metric: Nov+Dec vs Mar+Apr at similar sewage temperature
OUT["hysteresis_NovDec_minus_MarApr_pct"] = {c: float(clim.loc[[11, 12], c].mean() - clim.loc[[3, 4], c].mean()) for c in ("obs", "M0", "M1", "E1")}
OUT["Ts_MarApr_vs_NovDec"] = [float(clim.loc[[3, 4], "Ts"].mean()), float(clim.loc[[11, 12], "Ts"].mean())]

json.dump(OUT, open("out/model_results.json", "w"), indent=1, default=float)
print(json.dumps({k: OUT[k] for k in OUT if k not in ("tau_profile",)}, indent=1, default=float))
print(prof.round(3).to_string())
print(clim.round(2).to_string())

# ---- 6. Trade-off: how much of the season can temperature explain if mean in-sewer BOD loss is capped?
#         theta held to the literature range 1.03-1.10; tau, f, D20 free
def sig(x, lo, hi): return lo + (hi - lo) / (1 + np.exp(-x))
trade = []
for target in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
    def obj(p):
        D20 = np.exp(p[0]); th = sig(p[1], 1.03, 1.10); tau = sig(p[2], 1.0, 200.0); f = sig(p[3], 0.05, 1.0)
        z, Lw = loss_series(D20, th, tau, f)
        sse = ols_sse(y - z, Xbase)[0]
        return sse + 2000.0 * (Lw.mean() - target) ** 2
    best = None
    for D0 in (0.1, 0.5, 2.0):
        for t0 in (-1.0, 0.0, 1.0):
            for f0 in (-1.0, 1.5):
                r = minimize(obj, [np.log(D0), 1.0, t0, f0], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-8})
                if best is None or r.fun < best.fun: best = r
    p = best.x; D20 = np.exp(p[0]); th = sig(p[1], 1.03, 1.10); tau = sig(p[2], 1.0, 200.0); f = sig(p[3], 0.05, 1.0)
    z, Lw = loss_series(D20, th, tau, f); sse = ols_sse(y - z, Xbase)[0]
    trade.append({"target_meanL": target, "meanL": float(Lw.mean()), "Lmin": float(Lw.min()), "Lmax": float(Lw.max()),
                  "theta": float(th), "tau": float(tau), "f": float(f), "D20": float(D20),
                  "season_explained": float((sse_E0 - sse) / (sse_E0 - sse_E1))})
trade = pd.DataFrame(trade); trade.to_csv("out/model_loss_tradeoff.csv", index=False)
OUT["loss_tradeoff"] = trade.round(4).to_dict(orient="list")

# ---- 7. Per-capita check (plant assumed to serve Ismailia city; census 1996 = 255,134; 2006 = 293,184)
pop = lambda yy: 255134 * (293184 / 255134) ** ((yy - 1996) / 10)
pc = {}
for Y in (2000, 2001, 2002, 2003, 2005, 2008):
    Lt = u.loc[str(Y), "load"].mean()
    pc[Y] = {"load_t_d": float(Lt / 1000), "pop": float(pop(Y)), "g_cap_d": float(Lt * 1000 / pop(Y))}
OUT["per_capita"] = pc
json.dump(OUT, open("out/model_results.json", "w"), indent=1, default=float)
print(trade.round(3).to_string()); print(json.dumps(pc, indent=0))

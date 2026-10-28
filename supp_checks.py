"""Supplementary checks reported in the manuscript (run after sewer_model.py):
(1) influent DO vs ERA5-derived sewage temperature and vs plant temperature without 2004 (Section 3.4);
(2) timing and hysteresis of the biomass-memory model with theta fixed at 1.05 and 1.07 (Section 3.5)."""
import json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import lfilter

M = json.load(open("out/model_results.json"))
tr = M["sewage_temp_transfer"]
e = pd.read_csv("data/era5_openmeteo.csv", skiprows=3); e.columns = ["date", "Ta"]
e["date"] = pd.to_datetime(e.date); e = e.set_index("date").asfreq("D")
d = pd.read_csv("data/ismailia.csv"); d["date"] = pd.to_datetime(d.SampleDate, format="%d-%m-%y")
u = d[d.Stage == 1].set_index("date").sort_index()[["BOD5_inf", "Flow", "Temp", "DO_inf"]].asfreq("D").join(e)
u["Ts"] = tr["a"] + tr["b"] * u.Ta.ewm(span=tr["chosen_span_d"], adjust=False).mean()
u["load"] = u.BOD5_inf * u.Flow / 1000
OUT = {}

# (1) DO with year-month fixed effects
u["ym"] = u.index.to_period("M").astype(str)
g = u.dropna(subset=["DO_inf", "Ts", "Flow"])
m = smf.ols("DO_inf ~ Ts + np.log(Flow) + C(ym)", data=g).fit(cov_type="HAC", cov_kwds={"maxlags": 14})
OUT["DO_vs_ERA5_sewage_T"] = {"b": m.params["Ts"], "se": m.bse["Ts"], "n": int(m.nobs)}
g2 = g[g.index.year != 2004]
m2 = smf.ols("DO_inf ~ Temp + np.log(Flow) + C(ym)", data=g2).fit(cov_type="HAC", cov_kwds={"maxlags": 14})
OUT["DO_vs_plant_T_no2004"] = {"b": m2.params["Temp"], "se": m2.bse["Temp"], "n": int(m2.nobs)}

# (2) fixed-theta model timing (same weekly aggregation as sewer_model.py)
def ema(x, tau):
    a = 1.0 / tau
    y, _ = lfilter([a], [1, -(1 - a)], x, zi=[(1 - a) * x[:120].mean()]); return y
wk = u.index.to_period("W")
W_n = u.load.groupby(wk).count(); W_load = u.load.groupby(wk).mean()
keep = (W_n >= 4) & (W_load.index.start_time >= pd.Timestamp("2000-05-01"))
months = W_load[keep].index.start_time.month
for th, p in M["literature_theta"].items():
    Tx = ema(u.Ts.values, p["tau"])
    z = np.log(1 - 0.6 * (1 - np.exp(-p["D20"] * float(th) ** (Tx - 20))))
    zw = pd.Series(z, index=u.index).groupby(wk).mean()[keep].values
    mo = pd.Series(zw).groupby(np.asarray(months)).mean()
    pct = 100 * (np.exp(mo - mo.mean()) - 1)
    OUT[f"theta_{th}"] = {"tau": p["tau"], "peak_month": int(pct.idxmax()), "trough_month": int(pct.idxmin()),
                          "hysteresis_NovDec_minus_MarApr_pct": float(pct.loc[[11, 12]].mean() - pct.loc[[3, 4]].mean()),
                          "amplitude_pct": float(pct.max() - pct.min())}
json.dump(OUT, open("out/supp_checks.json", "w"), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))

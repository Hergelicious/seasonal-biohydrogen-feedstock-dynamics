"""Feasibility analysis: temperature-driven loss of dark-fermentation feedstock in sewers.
Data: Ismailia STP daily monitoring 2000-2008 (Farghaly et al. 2026, J Environ Manage 413:130281, CC BY 4.0)
      Hannovil (Alagamy) monthly tables 2008-2009.
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
R = {}

# ---------------- Ismailia: daily influent ----------------
d = pd.read_csv("data/ismailia.csv")
d["date"] = pd.to_datetime(d.SampleDate, format="%d-%m-%y")
u = d[d.Stage == 1].set_index("date").sort_index()
u = u[["BOD5_inf", "TSS_inf", "VSS_inf", "Temp", "Flow", "pH_inf", "DO_inf"]].asfreq("D")
u["load"] = u.BOD5_inf * u.Flow / 1000          # kg BOD5/d
u["vss_load"] = u.VSS_inf * u.Flow / 1000
u["tss_load"] = u.TSS_inf * u.Flow / 1000
u["bod_vss"] = u.BOD5_inf / u.VSS_inf
u["yr"] = u.index.year.astype(str); u["wd"] = u.index.dayofweek.astype(str)
u["moy"] = u.index.month.astype(str); u["ym"] = u.index.to_period("M").astype(str)
doy = u.index.dayofyear.values
for k in (1, 2):
    u[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25); u[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

RAM = {2000: ("2000-11-27", "2000-12-26"), 2001: ("2001-11-16", "2001-12-15"), 2002: ("2002-11-06", "2002-12-05"),
       2003: ("2003-10-27", "2003-11-24"), 2004: ("2004-10-15", "2004-11-13"), 2005: ("2005-10-04", "2005-11-02"),
       2006: ("2006-09-24", "2006-10-23"), 2007: ("2007-09-13", "2007-10-12"), 2008: ("2008-09-01", "2008-09-30")}
u["ram"] = 0
for a, b in RAM.values(): u.loc[a:b, "ram"] = 1

obs = u.load.notna()
R["coverage"] = {"days_total": int(len(u)), "days_with_data": int(obs.sum()),
                 "by_year": u.load.notna().groupby(u.index.year).sum().astype(int).to_dict()}
R["means"] = {"load_kg_d": float(u.load.mean()), "flow_m3_d": float(u.Flow.mean()), "BOD_mg_L": float(u.BOD5_inf.mean()),
              "temp_C": float(u.Temp.mean()), "temp_min": float(u.Temp.min()), "temp_max": float(u.Temp.max()),
              "load_2007_08": float(u.loc["2007":"2008", "load"].mean()), "flow_2000_01": float(u.loc["2000":"2001", "Flow"].mean()),
              "flow_2007_08": float(u.loc["2007":"2008", "Flow"].mean())}

# trailing temperature windows
WINS = [1, 7, 14, 30, 45, 60, 75, 90, 105, 120, 150, 180]
for W in WINS:
    u[f"T{W}"] = u.Temp.rolling(W, min_periods=max(1, int(W * 0.5))).mean()
# 30-day temperature windows ending k days earlier (timing scan)
LAGS = list(range(0, 181, 10))
for k in LAGS:
    u[f"L{k}"] = u.Temp.shift(k).rolling(30, min_periods=15).mean()

def fit(f, data, lags=30):
    return smf.ols(f, data=data).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

v = u.dropna(subset=["load", "T180", "L180"]).copy()
R["lag_sample_n"] = int(len(v))

# ---- Monthly climatology & harmonic phase
clim = u.groupby(u.index.month).agg(Temp=("Temp", "mean"), load=("load", "mean"), Flow=("Flow", "mean"),
                                    BOD=("BOD5_inf", "mean"), DO=("DO_inf", "mean"), vss_load=("vss_load", "mean"))
mon = u.resample("MS").agg({"load": ["mean", "count"], "Temp": "mean", "Flow": "mean", "DO_inf": "mean",
                            "vss_load": "mean", "tss_load": "mean", "BOD5_inf": "mean", "T90": "mean", "ram": "mean"})
mon.columns = ["load", "n", "Temp", "Flow", "DO", "vss_load", "tss_load", "BOD", "T90", "ram"]
mon = mon[mon.n >= 15].copy()
mon["month"] = mon.index.month; mon["year"] = mon.index.year
# climatology SE across years from monthly means
cl_se = mon.groupby("month")[["load", "Temp", "Flow", "DO", "vss_load"]].agg(["mean", "sem"])
cl_se.to_csv("out/climatology_monthly.csv")
R["climatology"] = clim.round(2).reset_index().to_dict(orient="list")
apr, aug = clim.load.loc[4], clim.load.loc[8]
R["seasonal_drop_apr_to_aug_pct"] = float(100 * (aug / apr - 1))
R["flow_range_pct"] = float(100 * (clim.Flow.max() / clim.Flow.min() - 1))
ph = {}
for y in ["np.log(load)", "Temp", "DO_inf", "np.log(Flow)", "np.log(BOD5_inf)", "np.log(vss_load)"]:
    m = smf.ols(f"{y} ~ s1 + c1 + ram + C(yr)", data=u.dropna(subset=["load"])).fit()
    a, b = m.params["s1"], m.params["c1"]
    ph[y] = {"amp": float(np.hypot(a, b)), "peak_doy": float((np.degrees(np.arctan2(a, b)) % 360) / 360 * 365.25)}
R["harmonic"] = ph
R["phase_lag_load_trough_after_T_peak_days"] = float(((ph["np.log(load)"]["peak_doy"] + 182.6) - ph["Temp"]["peak_doy"]) % 365.25)

# ---- Lag scan (trailing windows) and timing scan
base = "ram + C(wd) + C(yr)"
scan = []
for W in WINS:
    m = fit(f"np.log(load) ~ T{W} + {base}", v)
    scan.append({"W": W, "b": m.params[f"T{W}"], "se": m.bse[f"T{W}"], "aic": m.aic})
scan = pd.DataFrame(scan); scan["dAIC"] = scan.aic - scan.aic.min()
scan.to_csv("out/lag_scan_windows.csv", index=False)
R["lag_scan"] = scan.round(5).to_dict(orient="list")
tim = []
for k in LAGS:
    m = fit(f"np.log(load) ~ L{k} + {base}", v)
    tim.append({"k": k, "b": m.params[f"L{k}"], "se": m.bse[f"L{k}"], "aic": m.aic})
tim = pd.DataFrame(tim); tim["dAIC"] = tim.aic - tim.aic.min()
tim.to_csv("out/lag_scan_timing.csv", index=False)
R["timing_scan"] = tim.round(5).to_dict(orient="list")
# same-day, within-year-month (day-to-day) effect
m = fit("np.log(load) ~ Temp + ram + C(wd) + C(ym)", u.dropna(subset=["load"]), 14)
R["daily_within_month"] = {"b": m.params["Temp"], "se": m.bse["Temp"]}

# ---- Headline daily model
m90 = fit(f"np.log(load) ~ T90 + {base}", v)
R["main_T90"] = {"b": m90.params["T90"], "se": m90.bse["T90"], "ram": m90.params["ram"], "ram_se": m90.bse["ram"],
                 "pct_per_C": 100 * (np.exp(m90.params["T90"]) - 1)}

# ---- Interannual test: monthly, calendar-month FE + year FE
mm = mon.dropna(subset=["T90"]).copy()
mm["lload"] = np.log(mm.load)
mi = smf.ols("lload ~ T90 + ram + C(month) + C(year)", data=mm).fit(cov_type="cluster", cov_kwds={"groups": mm.year})
mi_hac = smf.ols("lload ~ T90 + ram + C(month) + C(year)", data=mm).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
R["interannual"] = {"n_months": int(len(mm)), "b": mi.params["T90"], "se_cluster": mi.bse["T90"], "se_hac": mi_hac.bse["T90"],
                    "pct_per_C": 100 * (np.exp(mi.params["T90"]) - 1)}
# partial residuals for plotting (FWL)
rx = smf.ols("T90 ~ ram + C(month) + C(year)", data=mm).fit().resid
ry = smf.ols("lload ~ ram + C(month) + C(year)", data=mm).fit().resid
pd.DataFrame({"T90_anom": rx, "lload_anom": ry, "month": mm.month, "year": mm.year}).to_csv("out/interannual_partial.csv")
# permutation: shuffle T90 anomalies across years within calendar month
obs_b = mi.params["T90"]; null = []
Tanom = rx.values; months = mm.month.values
for _ in range(5000):
    perm = Tanom.copy()
    for mo in np.unique(months):
        idx = np.where(months == mo)[0]; perm[idx] = rng.permutation(perm[idx])
    null.append(np.polyfit(perm, ry.values, 1)[0])
null = np.array(null)
np.save("out/perm_null.npy", null)
R["permutation"] = {"obs_b_fwl": float(np.polyfit(Tanom, ry.values, 1)[0]), "p_one_sided": float((null <= obs_b).mean()),
                    "p_two_sided": float((np.abs(null) >= abs(obs_b)).mean())}
# leave-one-year-out
loyo = {}
for y in sorted(mm.year.unique()):
    s = mm[mm.year != y]
    loyo[int(y)] = float(smf.ols("lload ~ T90 + ram + C(month) + C(year)", data=s).fit().params["T90"])
R["loyo"] = loyo

# ---- DO septicity (within year-month)
mdo = fit("DO_inf ~ Temp + np.log(Flow) + C(ym)", u.dropna(subset=["DO_inf"]), 14)
R["DO"] = {"b": mdo.params["Temp"], "se": mdo.bse["Temp"]}
R["DO_binned"] = u.groupby(pd.cut(u.Temp, [12, 16, 20, 24, 28, 35]), observed=True).DO_inf.agg(["mean", "sem", "count"]).reset_index().astype(str).to_dict(orient="list")
u.groupby(pd.cut(u.Temp, np.arange(12, 36, 2)), observed=True).DO_inf.agg(["mean", "sem", "count"]).to_csv("out/do_binned.csv")

# ---- Particulate vs total: VSS load, TSS load, BOD/VSS
part = {}
for y in ["np.log(vss_load)", "np.log(tss_load)", "np.log(bod_vss)", "np.log(Flow)", "np.log(BOD5_inf)"]:
    m = fit(f"{y} ~ T90 + {base}", v)
    part[y] = {"b": m.params["T90"], "se": m.bse["T90"]}
for y, nm in [("np.log(vss_load)", "vss"), ("np.log(tss_load)", "tss")]:
    s = mm.copy(); s["y"] = np.log(s[nm + "_load"])
    m = smf.ols("y ~ T90 + ram + C(month) + C(year)", data=s).fit(cov_type="cluster", cov_kwds={"groups": s.year})
    part[nm + "_interannual"] = {"b": m.params["T90"], "se": m.bse["T90"]}
R["particulate"] = part

# ---- Ramadan event study: residual log load after year FE, weekday, 2 harmonics, T90
w = u.dropna(subset=["load", "T90"]).copy()
mres = smf.ols("np.log(load) ~ T90 + s1 + c1 + s2 + c2 + C(wd) + C(yr)", data=w).fit()
w["res"] = mres.resid
ev = []
for y, (a, b) in RAM.items():
    st = pd.Timestamp(a)
    for off in range(-30, 61):
        dd = st + pd.Timedelta(days=off)
        if dd in w.index: ev.append({"year": y, "off": off, "res": w.at[dd, "res"]})
ev = pd.DataFrame(ev)
evw = ev.assign(wk=(ev.off // 7)).groupby("wk").res.agg(["mean", "sem", "count"]).reset_index()
evw["day"] = evw.wk * 7 + 3
evw.to_csv("out/ramadan_event.csv", index=False)
R["ramadan_years_with_data"] = sorted(ev.year.unique().tolist())
inside = ev[(ev.off >= 0) & (ev.off < 30)].res.mean(); before = ev[(ev.off >= -30) & (ev.off < 0)].res.mean()
R["ramadan_effect_pct"] = float(100 * (np.exp(inside - before) - 1))

# ---- Robustness of headline coefficient
rob = []
def add(name, f, data, lags=30, key="T90"):
    m = fit(f, data, lags); rob.append({"spec": name, "b": m.params[key], "se": m.bse[key], "n": int(m.nobs)})
add("Baseline: daily load, 90-d mean T, year FE, weekday, Ramadan", f"np.log(load) ~ T90 + {base}", v)
add("Drop sparse years 2006-2007", f"np.log(load) ~ T90 + {base}", v[~v.yr.isin(["2006", "2007"])])
add("Drop 2004 (suspect temperature record)", f"np.log(load) ~ T90 + {base}", v[v.yr != "2004"])
add("Exclude Ramadan +/- 7 d", f"np.log(load) ~ T90 + C(wd) + C(yr)", v[v.ram.rolling(15, center=True, min_periods=1).max() == 0])
add("BOD concentration, flow as covariate", f"np.log(BOD5_inf) ~ T90 + np.log(Flow) + {base}", v)
add("Linear trend instead of year FE", "np.log(load) ~ T90 + t + ram + C(wd)", v.assign(t=np.arange(len(v))))
add("Season fully controlled (2 harmonics) + year FE", f"np.log(load) ~ T90 + s1 + c1 + s2 + c2 + {base}", v)
add("60-d window", f"np.log(load) ~ T60 + {base}", v, key="T60")
add("120-d window", f"np.log(load) ~ T120 + {base}", v, key="T120")
wk = v.resample("W").mean(numeric_only=True).dropna(subset=["load"]); wk["yr"] = wk.index.year.astype(str)
m = smf.ols("np.log(load) ~ T90 + ram + C(yr)", data=wk).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
rob.append({"spec": "Weekly means", "b": m.params["T90"], "se": m.bse["T90"], "n": int(m.nobs)})
rob.append({"spec": "Monthly, calendar-month + year FE (interannual)", "b": mi.params["T90"], "se": mi_hac.bse["T90"], "n": int(mi.nobs)})
mm4 = mm[mm.year != 2004]
m4 = smf.ols("lload ~ T90 + ram + C(month) + C(year)", data=mm4).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
rob.append({"spec": "Season removed, drop 2004", "b": m4.params["T90"], "se": m4.bse["T90"], "n": int(m4.nobs)})
rob = pd.DataFrame(rob); rob["pct_per_C"] = 100 * (np.exp(rob.b) - 1)
rob["lo"] = 100 * (np.exp(rob.b - 1.96 * rob.se) - 1); rob["hi"] = 100 * (np.exp(rob.b + 1.96 * rob.se) - 1)
rob.to_csv("out/robustness.csv", index=False)
R["robustness"] = rob.round(4).to_dict(orient="records")

# ---------------- Hannovil ----------------
h = pd.read_excel("data/hannovil.xlsx", header=None)
rows = h.iloc[3:20, :17].copy()
rows.columns = ["Month", "pH_i", "pH_e", "T_i", "T_e", "GO_i", "GO_e", "H2S_i", "H2S_e", "DO_i", "DO_e",
                "TSS_i", "TSS_e", "BOD_i", "BOD_e", "COD_i", "COD_e"]
rows["date"] = pd.date_range("2008-08-01", periods=17, freq="MS")
for c in rows.columns[1:-1]: rows[c] = pd.to_numeric(rows[c])
rows["COD_BOD"] = rows.COD_i / rows.BOD_i
rows["T_lag2"] = rows.T_i.shift(2)
rows.to_csv("out/hannovil_monthly.csv", index=False)
from scipy.stats import spearmanr
def sp(a, b):
    s = rows[[a, b]].dropna(); r, p = spearmanr(s[a], s[b]); return {"rho": float(r), "p": float(p), "n": int(len(s))}
pre = rows[rows.date < "2009-07-01"]
R["hannovil"] = {"n_months": int(len(rows)), "COD_BOD_mean": float(rows.COD_BOD.mean()), "COD_BOD_min": float(rows.COD_BOD.min()),
                 "COD_BOD_max": float(rows.COD_BOD.max()), "COD_BOD_pre_jul09": float(pre.COD_BOD.mean()),
                 "COD_BOD_post_jul09": float(rows[rows.date >= "2009-07-01"].COD_BOD.mean()),
                 "COD_pre": float(pre.COD_i.mean()), "COD_post": float(rows[rows.date >= "2009-07-01"].COD_i.mean()),
                 "T_range": [float(rows.T_i.min()), float(rows.T_i.max())],
                 "rho_COD_T": sp("COD_i", "T_i"), "rho_COD_Tlag2": sp("COD_i", "T_lag2"), "rho_BOD_T": sp("BOD_i", "T_i"),
                 "rho_BOD_Tlag2": sp("BOD_i", "T_lag2"), "H2S_mean": float(rows.H2S_i.mean()),
                 "DO_zero_months": int((rows.DO_i == 0).sum())}
# within pre-step only (Aug08-Jun09, 11 months)
p2 = pre.copy(); p2["T_lag2"] = p2.T_i.shift(2)
r_, p_ = spearmanr(p2.dropna(subset=["T_lag2"]).COD_i, p2.dropna(subset=["T_lag2"]).T_lag2)
R["hannovil"]["rho_COD_Tlag2_pre_step"] = {"rho": float(r_), "p": float(p_), "n": int(p2.T_lag2.notna().sum())}

# ---------------- DF hydrogen translation (Monte Carlo) ----------------
N = 20000
beta_d = rng.normal(R["interannual"]["b"], R["interannual"]["se_cluster"], N)   # per degC, log scale
r_cb = rng.uniform(1.6, 2.4, N)        # COD:BOD5 (Hannovil 1.7; raw sewage literature ~2-2.4)
f_fer = rng.uniform(0.10, 0.30, N)     # readily fermentable (carbohydrate-type) fraction of COD
Y = rng.uniform(0.8, 2.0, N)           # mol H2 / mol hexose, mixed-culture DF
load = R["means"]["load_2007_08"]      # kg BOD5/d (2007-2008 mean)
H2_mol = load * 1000 * r_cb * f_fer / 192.0 * Y      # mol H2/d  (192 g COD per mol hexose)
H2_kg = H2_mol * 2.016 / 1000
H2_GJ = H2_mol * 0.242 / 1000                        # LHV 242 kJ/mol
R["H2_baseline"] = {"kg_d": [float(np.percentile(H2_kg, q)) for q in (5, 50, 95)],
                    "GJ_d": [float(np.percentile(H2_GJ, q)) for q in (5, 50, 95)]}
loss_per_C = 1 - np.exp(beta_d)
R["H2_loss_per_C"] = {"pct": [float(100 * np.percentile(loss_per_C, q)) for q in (5, 50, 95)],
                      "kg_d": [float(np.percentile(H2_kg * loss_per_C, q)) for q in (5, 50, 95)]}
scen = []
for dT in (1.5, 2.0, 3.0, 4.0):
    l = 1 - np.exp(beta_d * dT)
    scen.append({"dT": dT, "loss_pct_p5": 100 * np.percentile(l, 5), "loss_pct_p50": 100 * np.percentile(l, 50),
                 "loss_pct_p95": 100 * np.percentile(l, 95), "kg_d_p50": np.percentile(H2_kg * l, 50),
                 "kg_d_p5": np.percentile(H2_kg * l, 5), "kg_d_p95": np.percentile(H2_kg * l, 95)})
scen = pd.DataFrame(scen); scen.to_csv("out/warming_scenarios.csv", index=False)
R["warming"] = scen.round(3).to_dict(orient="records")
# observed seasonal swing in H2 potential (Apr vs Aug climatology)
R["H2_seasonal_swing_kg_d_p50"] = float(np.percentile((apr - aug) * 1000 * r_cb * f_fer / 192 * Y * 2.016 / 1000, 50))
R["T90_range_obs"] = [float(v.T90.min()), float(v.T90.max())]

R["summer_temp_JulAug_by_year"] = u[u.index.month.isin([7, 8])].groupby(u[u.index.month.isin([7, 8])].index.year).Temp.mean().round(1).to_dict()
# monthly H2 feedstock potential climatology
mc = []
for mo in range(1, 13):
    L = clim.load.loc[mo]
    h2 = L * 1000 * r_cb * f_fer / 192.0 * Y * 2.016 / 1000
    mc.append({"month": mo, "p5": np.percentile(h2, 5), "p50": np.percentile(h2, 50), "p95": np.percentile(h2, 95)})
pd.DataFrame(mc).to_csv("out/h2_monthly.csv", index=False)
R["h2_monthly"] = pd.DataFrame(mc).round(1).to_dict(orient="records")
json.dump(R, open("out/results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float)[:12000])

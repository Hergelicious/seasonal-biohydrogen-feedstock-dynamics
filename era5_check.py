"""ERA5 (via Open-Meteo, grid 30.50N 32.25E) check of the Ismailia temperature record and re-run of the
season-removed test with an independent temperature series."""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
rng = np.random.default_rng(11)
OUT = {}

e = pd.read_csv("data/era5_openmeteo.csv", skiprows=3); e.columns = ["date", "Ta"]
e["date"] = pd.to_datetime(e.date); e = e.set_index("date").asfreq("D")
d = pd.read_csv("data/ismailia.csv"); d["date"] = pd.to_datetime(d.SampleDate, format="%d-%m-%y")
u = d[d.Stage == 1].set_index("date").sort_index()[["BOD5_inf", "Flow", "Temp", "VSS_inf", "DO_inf"]].asfreq("D")
u = u.join(e)
u["load"] = u.BOD5_inf * u.Flow / 1000
RAM = {2000: ("2000-11-27", "2000-12-26"), 2001: ("2001-11-16", "2001-12-15"), 2002: ("2002-11-06", "2002-12-05"),
       2003: ("2003-10-27", "2003-11-24"), 2004: ("2004-10-15", "2004-11-13"), 2005: ("2005-10-04", "2005-11-02"),
       2006: ("2006-09-24", "2006-10-23"), 2007: ("2007-09-13", "2007-10-12"), 2008: ("2008-09-01", "2008-09-30")}
u["ram"] = 0
for a, b in RAM.values(): u.loc[a:b, "ram"] = 1
u["yr"] = u.index.year.astype(str); u["wd"] = u.index.dayofweek.astype(str)
doy = u.index.dayofyear.values
for k in (1, 2):
    u[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25); u[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
# ERA5 is complete, so its trailing means are exact on every day
for W in (1, 7, 14, 30, 45, 60, 75, 90, 105, 120, 150, 180):
    u[f"A{W}"] = u.Ta.rolling(W).mean()
    u[f"T{W}"] = u.Temp.rolling(W, min_periods=max(1, int(W * 0.5))).mean()
for k in range(0, 181, 10):
    u[f"AL{k}"] = u.Ta.shift(k).rolling(30).mean()
OUT["grid"] = {"lat": 30.5, "lon": 32.25, "elev_m": 7, "plant": [30.486, 32.247], "distance_km": 1.6}

# ---- 1. Plant record vs ERA5
u["A30"] = u.Ta.rolling(30).mean()
ja = u[u.index.month.isin([7, 8])]
summ = pd.DataFrame({"plant": ja.groupby(ja.index.year).Temp.mean(), "era5": ja.groupby(ja.index.year).Ta.mean(),
                     "n_plant": ja.groupby(ja.index.year).Temp.count()})
summ["diff"] = summ.plant - summ.era5
summ.to_csv("out/era5_summer_compare.csv")
OUT["summer_JulAug"] = summ.round(2).reset_index().to_dict(orient="list")
win = u[u.index.month.isin([1, 2])]
wint = pd.DataFrame({"plant": win.groupby(win.index.year).Temp.mean(), "era5": win.groupby(win.index.year).Ta.mean()})
wint["diff"] = wint.plant - wint.era5
OUT["winter_JanFeb"] = wint.round(2).reset_index().to_dict(orient="list")
# per-year calibration: plant ~ a + b * ERA5 30-d trailing mean
cal = []
for y, g in u.dropna(subset=["Temp", "A30"]).groupby(u.dropna(subset=["Temp", "A30"]).index.year):
    m = smf.ols("Temp ~ A30", data=g).fit()
    cal.append({"year": y, "slope": m.params["A30"], "intercept": m.params["Intercept"], "r2": m.rsquared, "n": int(m.nobs),
                "resid_sd": float(np.std(m.resid))})
cal = pd.DataFrame(cal); cal.to_csv("out/era5_calibration_by_year.csv", index=False)
OUT["calibration_by_year"] = cal.round(3).to_dict(orient="list")
mo = u.resample("MS").mean(numeric_only=True)
mo["offset"] = mo.Temp - mo.A30
mo[["Temp", "Ta", "A30", "offset"]].to_csv("out/era5_monthly_offset.csv")

# ---- 2. Seasonal (daily) models with ERA5
v = u.dropna(subset=["load", "A180", "AL180"]).copy()
base = "ram + C(wd) + C(yr)"
def fit(f, data, lags=30):
    return smf.ols(f, data=data).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
scan = []
for W in (1, 7, 14, 30, 45, 60, 75, 90, 105, 120, 150, 180):
    m = fit(f"np.log(load) ~ A{W} + {base}", v); scan.append({"W": W, "b": m.params[f"A{W}"], "se": m.bse[f"A{W}"], "aic": m.aic})
scan = pd.DataFrame(scan); scan["dAIC"] = scan.aic - scan.aic.min(); scan.to_csv("out/era5_lag_windows.csv", index=False)
tim = []
for k in range(0, 181, 10):
    m = fit(f"np.log(load) ~ AL{k} + {base}", v); tim.append({"k": k, "b": m.params[f"AL{k}"], "se": m.bse[f"AL{k}"], "aic": m.aic})
tim = pd.DataFrame(tim); tim["dAIC"] = tim.aic - tim.aic.min(); tim.to_csv("out/era5_lag_timing.csv", index=False)
OUT["era5_lag_windows"] = scan.round(5).to_dict(orient="list")
OUT["era5_timing"] = tim.round(5).to_dict(orient="list")
m = fit(f"np.log(load) ~ A90 + {base}", v)
OUT["era5_seasonal_A90"] = {"b": m.params["A90"], "se": m.bse["A90"], "n": int(m.nobs)}
m = fit(f"np.log(load) ~ A90 + s1 + c1 + s2 + c2 + {base}", v)
OUT["era5_harmonics_A90"] = {"b": m.params["A90"], "se": m.bse["A90"]}

# ---- 3. Season-removed (same month across years) with ERA5
mon = u.resample("MS").agg({"load": ["mean", "count"], "A90": "mean", "A60": "mean", "A120": "mean", "T90": "mean", "ram": "mean"})
mon.columns = ["load", "n", "A90", "A60", "A120", "T90", "ram"]
mon = mon[mon.n >= 15].dropna(subset=["A90"]).copy()
mon["lload"] = np.log(mon.load); mon["month"] = mon.index.month; mon["year"] = mon.index.year
res = {}
def season_removed(var, data, tag):
    data = data.dropna(subset=[var]).copy()
    mm = smf.ols(f"lload ~ {var} + ram + C(month) + C(year)", data=data).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    rx = smf.ols(f"{var} ~ ram + C(month) + C(year)", data=data).fit().resid
    ry = smf.ols("lload ~ ram + C(month) + C(year)", data=data).fit().resid
    b = np.polyfit(rx, ry, 1)[0]; null = []
    mos = data.month.values
    for _ in range(5000):
        p = rx.values.copy()
        for mth in np.unique(mos):
            i = np.where(mos == mth)[0]; p[i] = rng.permutation(p[i])
        null.append(np.polyfit(p, ry.values, 1)[0])
    null = np.array(null)
    loyo = {int(y): float(smf.ols(f"lload ~ {var} + ram + C(month) + C(year)", data=data[data.year != y]).fit().params[var])
            for y in sorted(data.year.unique())}
    res[tag] = {"b": float(mm.params[var]), "se": float(mm.bse[var]), "n": int(mm.nobs),
                "pct": float(100 * (np.exp(mm.params[var]) - 1)),
                "lo": float(100 * (np.exp(mm.params[var] - 1.96 * mm.bse[var]) - 1)),
                "hi": float(100 * (np.exp(mm.params[var] + 1.96 * mm.bse[var]) - 1)),
                "p_perm_two_sided": float((np.abs(null) >= abs(b)).mean()), "anom_sd_C": float(np.std(rx)),
                "loyo_pct": {k: round(100 * (np.exp(v_) - 1), 2) for k, v_ in loyo.items()}}
    if tag == "ERA5 90-d":
        pd.DataFrame({"T_anom": rx, "lload_anom": ry, "month": data.month, "year": data.year}).to_csv("out/era5_partial.csv")
season_removed("A90", mon, "ERA5 90-d")
season_removed("A60", mon, "ERA5 60-d")
season_removed("A120", mon, "ERA5 120-d")
season_removed("A90", mon[mon.year != 2004], "ERA5 90-d, drop 2004")
mp = mon.dropna(subset=["T90"])
season_removed("T90", mp, "Plant 90-d (for comparison)")
OUT["season_removed"] = res
# Minimum detectable effect at 80% power, two-sided 5%: (1.96+0.84)*SE
OUT["MDE_pct_per_C_era5"] = float(100 * (np.exp(2.8 * res["ERA5 90-d"]["se"]) - 1))

# ---- 4. Corrected plant temperature: ERA5-calibrated sewage temperature (2000-2003 calibration)
c0 = u.loc["2000":"2003"].dropna(subset=["Temp", "A30"])
mc = smf.ols("Temp ~ A30", data=c0).fit()
OUT["calib_2000_2003"] = {"intercept": mc.params["Intercept"], "slope": mc.params["A30"], "r2": mc.rsquared, "resid_sd": float(np.std(mc.resid))}

json.dump(OUT, open("out/era5_results.json", "w"), indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ["summer_JulAug", "winter_JanFeb", "calibration_by_year", "era5_seasonal_A90",
                                     "era5_harmonics_A90", "season_removed", "MDE_pct_per_C_era5", "calib_2000_2003"]}, indent=1, default=float))
print(scan[["W", "b", "dAIC"]].round(4).to_string()); print(tim[["k", "b", "se", "dAIC"]].round(4).to_string())

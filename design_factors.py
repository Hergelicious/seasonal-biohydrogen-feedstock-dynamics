"""Engineering summary (Table 1): monthly feedstock design factors with uncertainty across years.
Design factor = monthly mean BOD5 load / annual mean load of the same year (removes growth trend);
95% CI by bootstrap over years (complete years only). Also the April->August change with its CI."""
import json, numpy as np, pandas as pd
rng = np.random.default_rng(2026)
d = pd.read_csv("data/ismailia.csv"); d["date"] = pd.to_datetime(d.SampleDate, format="%d-%m-%y")
u = d[d.Stage == 1].set_index("date").sort_index()
u["load"] = u.BOD5_inf * u.Flow / 1000
mon = u.load.resample("MS").agg(["mean", "count"])
mon = mon[mon["count"] >= 15].copy(); mon["year"] = mon.index.year; mon["month"] = mon.index.month
full = mon.groupby("year").month.nunique(); years = full[full == 12].index.values      # complete years
m = mon[mon.year.isin(years)].copy()
m["factor"] = m["mean"] / m.groupby("year")["mean"].transform("mean")
F = m.pivot(index="year", columns="month", values="factor")
B = 10000
boot = np.array([F.loc[rng.choice(F.index, len(F.index), replace=True)].mean().values for _ in range(B)])
H2 = pd.read_csv("out/h2_monthly.csv").set_index("month")
tab = pd.DataFrame({"Month": range(1, 13), "Design factor": F.mean().values,
                    "95% CI low": np.percentile(boot, 2.5, axis=0), "95% CI high": np.percentile(boot, 97.5, axis=0),
                    "Years below 0.90": (F < 0.90).sum().values, "Median DF H2 potential (kg/d)": H2.p50.values})
ratio = F[8] / F[4]
rb = np.array([ratio.loc[rng.choice(ratio.index, len(ratio), replace=True)].mean() for _ in range(B)])
out = {"complete_years": [int(y) for y in years], "n_years": int(len(years)),
       "apr_to_aug_change_pct": float(100 * (ratio.mean() - 1)),
       "apr_to_aug_CI_pct": [float(100 * (np.percentile(rb, 2.5) - 1)), float(100 * (np.percentile(rb, 97.5) - 1))],
       "years_with_aug_below_apr": int((ratio < 1).sum()),
       "min_month": int(tab.loc[tab["Design factor"].idxmin(), "Month"]), "min_factor": float(tab["Design factor"].min()),
       "max_month": int(tab.loc[tab["Design factor"].idxmax(), "Month"]), "max_factor": float(tab["Design factor"].max()),
       "months_below_0.95": tab.loc[tab["Design factor"] < 0.95, "Month"].tolist()}
tab.round(3).to_csv("out/table1_design_factors.csv", index=False)
json.dump(out, open("out/design_factors.json", "w"), indent=1)
print(tab.round(3).to_string(index=False)); print(json.dumps(out, indent=1))

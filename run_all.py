"""Reproduces every number, table and figure in the manuscript and supplement.
Usage (from this folder):  python run_all.py
Order matters: later scripts read the JSON/CSV outputs of earlier ones."""
import os, shutil, subprocess, sys, time
os.makedirs("out", exist_ok=True); os.makedirs("fig", exist_ok=True); os.makedirs("manuscript_figs", exist_ok=True)
STEPS = ["analysis.py",        # Sections 3.1-3.4, 3.6: seasonal cycle, lag scans, season-removed test (plant T), DO, Ramadan, Hannovil, H2 Monte Carlo
         "era5_check.py",      # Section 3.3: ERA5 vs plant temperature, season-removed test with ERA5
         "sewer_model.py",     # Section 3.5: lumped sewer process model, tau profile, loss trade-off, per-capita check
         "supp_checks.py",     # Sections 3.4-3.5: DO vs ERA5-derived sewage T; fixed-theta timing/hysteresis
         "design_factors.py",  # Table 1 and Section 3.1: monthly design factors and April-August change with bootstrap CI
         "figures.py",         # Figures 1, 2, 4 (mechanism checks) and 6 (H2) in their draft numbering
         "fig5_era5.py",       # Figure 3 (ERA5 check)
         "fig6_model.py",      # Figure 5 (process model)
         "graphical_abstract.py",
         "supplementary.py",   # Tables S1-S5 (Excel) and Figs S1-S2
         "supp_docx.py"]       # Supplementary_Material.docx
for s in STEPS:
    t = time.time(); print(f"-> {s}", flush=True)
    subprocess.run([sys.executable, s], check=True, stdout=subprocess.DEVNULL)
    print(f"   done in {time.time() - t:.0f} s")
ORDER = {"Fig1_seasonal": "Figure1", "Fig2_lag_robustness": "Figure2", "Fig5_era5_check": "Figure3",
         "Fig3_mechanism_checks": "Figure4", "Fig6_sewer_model": "Figure5", "Fig4_h2_monthly": "Figure6"}
for src, dst in ORDER.items():
    for ext in ("png", "pdf"):
        shutil.copy(f"fig/{src}.{ext}", f"manuscript_figs/{dst}.{ext}")
print("All outputs in out/, fig/ and manuscript_figs/.")

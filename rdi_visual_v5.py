# -*- coding: utf-8 -*-
"""
rdi_visual_v5.py

Vizuelna provera RDI_v5 rezultata:
- Histogram RDI_v5_score_0_100
- Scatter RDI_v5_score_0_100 vs viability_pct (+ linearni fit i R^2)
- RDI_v5_score_0_100 kroz vreme (time_h) ili po koncentraciji (concentration_uM)

Upotreba (primer):
python "E:\OneDrive\Desktop\Redox B2\rdi_visual_v5.py" ^
  --csv "E:\OneDrive\Desktop\Redox B2\redox_data_with_RDI_v5.csv"
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_r2(x, y):
    # R^2 iz linearne regresije (numpy polyfit 1. stepena)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, (np.nan, np.nan)
    x_ = x[mask]
    y_ = y[mask]
    coef = np.polyfit(x_, y_, 1)
    y_pred = np.polyval(coef, x_)
    ss_res = np.sum((y_ - y_pred) ** 2)
    ss_tot = np.sum((y_ - np.mean(y_)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r2, tuple(coef)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Putanja do redox_data_with_RDI_v5.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = os.path.dirname(os.path.abspath(args.csv))

    # --- 1) Histogram RDI_v5_score_0_100
    if "RDI_v5_score_0_100" not in df.columns:
        raise SystemExit("Nema kolone 'RDI_v5_score_0_100' u CSV-u.")

    plt.figure()
    df["RDI_v5_score_0_100"].plot(kind="hist", bins=20, edgecolor="black")
    plt.xlabel("RDI_v5_score_0_100 (0 = najzdravije, 100 = najtoksičnije)")
    plt.ylabel("Broj uzoraka")
    plt.title("Histogram RDI_v5_score_0_100")
    hist_path = os.path.join(out_dir, "RDI_v5_score_histogram.png")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    # plt.show()  # opciono

    # --- 2) Scatter RDI vs Viability (+ R^2)
    if "viability_pct" in df.columns:
        x = df["RDI_v5_score_0_100"].values.astype(float)
        y = df["viability_pct"].values.astype(float)
        r2, (a, b) = compute_r2(x, y)

        plt.figure()
        plt.scatter(x, y, alpha=0.8)
        # lin. fit
        if np.isfinite(r2):
            xgrid = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ygrid = a * xgrid + b
            plt.plot(xgrid, ygrid, linewidth=2)
            plt.title(f"RDI score vs Viability (R² = {r2:.3f})")
        else:
            plt.title("RDI score vs Viability (R² = n/a)")
        plt.xlabel("RDI_v5_score_0_100 (veće = toksičnije)")
        plt.ylabel("Viability (%)")
        scat_path = os.path.join(out_dir, "RDI_v5_score_vs_viability.png")
        plt.tight_layout()
        plt.savefig(scat_path, dpi=200)

        # Pearson korelacija (informativno)
        mask = np.isfinite(x) & np.isfinite(y)
        pearson = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() >= 3 else np.nan

        print("=== METRIKE ===")
        print(f"R²(RDI_score -> Viability) = {r2:.4f}" if np.isfinite(r2) else "R² = n/a")
        print(f"Pearson corr = {pearson:.4f}" if np.isfinite(pearson) else "Pearson = n/a")
        print(f"Sačuvano: {scat_path}")
    else:
        print("Upozorenje: Nema kolone 'viability_pct' – preskačem scatter i R².")

    # --- 3) RDI vs vreme ili koncentracija
    plt.figure()
    if "time_h" in df.columns:
        df_sorted = df.sort_values("time_h")
        plt.plot(df_sorted["time_h"].values, df_sorted["RDI_v5_score_0_100"].values, marker="o")
        plt.xlabel("time_h")
        plt.title("RDI_v5_score_0_100 kroz vreme")
        time_path = os.path.join(out_dir, "RDI_v5_score_vs_time.png")
        plt.tight_layout()
        plt.savefig(time_path, dpi=200)
        print(f"Sačuvano: {time_path}")
    elif "concentration_uM" in df.columns:
        df_sorted = df.sort_values("concentration_uM")
        plt.plot(df_sorted["concentration_uM"].values, df_sorted["RDI_v5_score_0_100"].values, marker="o")
        plt.xlabel("concentration_uM")
        plt.title("RDI_v5_score_0_100 po koncentraciji")
        conc_path = os.path.join(out_dir, "RDI_v5_score_vs_concentration.png")
        plt.tight_layout()
        plt.savefig(conc_path, dpi=200)
        print(f"Sačuvano: {conc_path}")
    else:
        plt.close()
        print("Info: Nema 'time_h' ni 'concentration_uM' – preskačem treći graf.")

    print(f"Sačuvano: {hist_path}")

if __name__ == "__main__":
    # Ako želiš da pokrećeš direktno iz IDLE-a bez argumenata, otkomentariši sledeći blok:
    """
    import sys
    sys.argv = [
        "rdi_visual_v5.py",
        "--csv",
        r"E:\OneDrive\Desktop\Redox B2\redox_data_with_RDI_v5.csv"
    ]
    """
    main()

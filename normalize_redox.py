# normalize_redox.py
# Korak 1: normalizacija 0–1 za redox parametre po ćeliji (bez ikakvog učenja/pondera).
# Ulaz:  E:\OneDrive\Desktop\Redox B2\redox_data.csv
# Izlaz: E:\OneDrive\Desktop\Redox B2\redox_data_normalized.csv

import os
import sys
import math
import pandas as pd
import numpy as np

# === PODESI PUTANJU (po potrebi izmeni samo ove dve linije) ===
FOLDER = r"E:\OneDrive\Desktop\Redox B2"
INPUT_CSV = "redox_data.csv"
OUTPUT_CSV = "redox_data_normalized.csv"

INPUT_PATH = os.path.join(FOLDER, INPUT_CSV)
OUTPUT_PATH = os.path.join(FOLDER, OUTPUT_CSV)

# Kolone za normalizaciju (moraju postojati u CSV)
FEATURE_COLS = [
    "superoxide_uM_per_viable_cell",
    "nitrites_uM_per_viable_cell",
    "h2o2_uM_per_viable_cell",
    "gsh_uM_per_viable_cell",
    "gssg_uM_per_viable_cell",
]

# Dodatne kolone koje prenosimo "kroz" fajl bez promene
PASS_THROUGH_COLS = [
    "time_h",
    "concentration_uM",
    "viability_pct"  # ostaje netaknuto; služiće kasnije kao ishod (target)
]

# Opcija: želiš li log10 transform za GSH/GSSG (zbog velikog opsega)?
# Ako True, radi se log10(x+epsilon) pre min-max skaliranja; rezultat i dalje 0-1.
USE_LOG_FOR = {
    "gsh_uM_per_viable_cell": False,
    "gssg_uM_per_viable_cell": False,
}

EPS = 1e-12  # mali broj da se izbegne log(0)

def min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max 0–1 normalizacija; ako je konstanta, vrati 0.0."""
    s = series.astype(float)
    s_min, s_max = s.min(), s.max()
    if pd.isna(s_min) or pd.isna(s_max):
        return pd.Series(np.nan, index=s.index)
    if s_max == s_min:
        return pd.Series(0.0, index=s.index)
    return (s - s_min) / (s_max - s_min)

def safe_log10(series: pd.Series) -> pd.Series:
    """log10(x + EPS) samo za nenegativne vrednosti; ako ima negativnih, baci grešku."""
    if (series < 0).any():
        raise ValueError(
            "Nađena negativna vrednost u koloni koja je označena za log-transform. "
            "Proveri podatke."
        )
    return np.log10(series + EPS)

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"[GREŠKA] Ne postoji ulazni fajl: {INPUT_PATH}")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)

    # Provera kolona
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[GREŠKA] Nedostaju kolone u CSV: {missing}")
        sys.exit(1)

    # Ako neka PASS_THROUGH kolona ne postoji, samo je preskoči (nije kritično)
    pass_cols = [c for c in PASS_THROUGH_COLS if c in df.columns]

    # Kopija izlaza: prvo prenesi pass-through kolone (ako postoje)
    out = pd.DataFrame()
    for c in pass_cols:
        out[c] = df[c]

    # Normalizuj svaku feature kolonu
    for col in FEATURE_COLS:
        col_data = df[col].astype(float)

        # Osnovna validacija
        if col_data.isna().all():
            print(f"[UPOZORENJE] Kolona '{col}' je cela NaN; biće normirana u NaN.")
        if (col_data < 0).any():
            # Negativne vrednosti obično nisu očekivane za koncentracije po ćeliji
            print(f"[UPOZORENJE] Kolona '{col}' sadrži negativne vrednosti.")

        # (Opcioni) log10 pre min-max
        if USE_LOG_FOR.get(col, False):
            col_data = safe_log10(col_data)

        out["norm_" + col] = min_max_normalize(col_data)

    # Sačuvaj rezultat
    out.to_csv(OUTPUT_PATH, index=False)

    # Kratak rezime u konzoli
    print("[OK] Normalizacija završena.")
    print(f"Ulaz:  {INPUT_PATH}")
    print(f"Izlaz: {OUTPUT_PATH}")
    print("\nPrvih 5 redova normalizovanog izlaza:")
    with pd.option_context("display.max_columns", None):
        print(out.head())

if __name__ == "__main__":
    main()

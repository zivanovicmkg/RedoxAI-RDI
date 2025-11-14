# -*- coding: utf-8 -*-
"""
compute_rdi_v5.py

Robustno računanje RDI_v5 iz normalizovanog CSV + AI-pondera, sa:
- Quick-start modom (bez argumenata) -> koristi fajlove u istom folderu
- Automatskim smerom skora (veće = toksičnije; niža viability)
- Robustnim skaliranjem na 0–100 (percentili 5–95)
- Podrškom za interakcione termine (A*B) u weights CSV

Primer (Windows, sa navodnicima):
python compute_rdi_v5.py ^
  --input-csv "E:\OneDrive\Desktop\Redox B2\redox_data_normalized.csv" ^
  --weights-csv "E:\OneDrive\Desktop\Redox B2\ai_learned_weights_v5.csv" ^
  --output-csv "E:\OneDrive\Desktop\Redox B2\redox_data_with_RDI_v5.csv" ^
  --train-csv "E:\OneDrive\Desktop\Redox B2\redox_data_normalized.csv"

Autor: RedoxAI (RDI_v5)
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

def _echo(msg: str):
    print(f"[RDI] {msg}", flush=True)

def load_weights(weights_path: str) -> pd.DataFrame:
    """
    Učita CSV sa kolonama:
      - (preference) term, weight
      - (alternativa) feature, mean
    Term može biti single feature (npr. norm_superoxide...) ili interakcija 'A*B'.
    """
    w = pd.read_csv(weights_path)
    # Normalizuj kolone
    cols = [c.lower() for c in w.columns]
    w.columns = cols

    # Pronađi naziv kolone sa imenima termina
    term_col = None
    for cand in ["term", "feature", "name"]:
        if cand in w.columns:
            term_col = cand
            break
    if term_col is None:
        raise SystemExit("Weights CSV mora imati kolonu 'term' ili 'feature'.")

    # Pronađi vrednosti pondera
    weight_col = None
    for cand in ["weight", "coef", "mean", "value"]:
        if cand in w.columns:
            weight_col = cand
            break
    if weight_col is None:
        raise SystemExit("Weights CSV mora imati kolonu 'weight' ili 'mean' (ili 'coef'/'value').")

    w = w[[term_col, weight_col]].copy()
    w.columns = ["term", "weight"]
    # Filtriraj NaN/inf
    w = w.replace([np.inf, -np.inf], np.nan).dropna(subset=["term", "weight"])
    return w

def compute_linear_score(df: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """
    Izračunava linearnu kombinaciju:
      score = Σ w_i * X_i + Σ w_ij * (X_i * X_j)
    gde su termini u weights.term (single: 'feat', interakcija: 'featA*featB').
    Ako neka kolona ne postoji u df, termin se preskače uz upozorenje.
    """
    score = np.zeros(len(df), dtype=float)
    missing_terms = []

    for _, row in weights.iterrows():
        term = str(row["term"])
        w = float(row["weight"])

        if "*" in term:
            a, b = term.split("*", 1)
            a = a.strip()
            b = b.strip()
            if a in df.columns and b in df.columns:
                score += w * (df[a].astype(float).values * df[b].astype(float).values)
            else:
                missing_terms.append(term)
        else:
            feat = term.strip()
            if feat in df.columns:
                score += w * df[feat].astype(float).values
            else:
                missing_terms.append(term)

    if missing_terms:
        uniq = sorted(set(missing_terms))
        _echo(f"Upozorenje: Preskočeno {len(uniq)} termina jer kolone nisu nađene: {uniq[:8]}{' ...' if len(uniq)>8 else ''}")
    return pd.Series(score, index=df.index, name="RDI_v5_raw")

def orient_and_scale(score_raw: pd.Series,
                     ref_df: pd.DataFrame,
                     viability_col: str = "viability_pct",
                     robust_percentiles=(5, 95)) -> (pd.Series, dict):
    """
    1) Odredi smer: ako corr(RDI_raw, viability) > 0  → invertuj (RDI := -RDI),
       jer veći RDI mora značiti lošije (niža viability).
    2) Robustno skaliraj na 0–100 pomoću percentila (5–95) iz ref_df (ako ima),
       u suprotnom iz score_raw. Clamp na [0,100].
    Vraća: (RDI_v5_aligned, RDI_v5_score_0_100), meta info.
    """
    aligned = score_raw.copy()

    # Odredi smer koristeći referentnu tabelu koja sadrži viability (train ili input)
    corr = np.nan
    if ref_df is not None and viability_col in ref_df.columns:
        try:
            x = score_raw.values.astype(float)
            y = ref_df[viability_col].reindex(score_raw.index).values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3:
                corr = np.corrcoef(x[mask], y[mask])[0, 1]
                if corr > 0:
                    aligned = -aligned
                    _echo(f"Automatska orijentacija: invertujem znak (Pearson corr sa viability = {corr:.3f} > 0).")
                else:
                    _echo(f"Automatska orijentacija: znak zadržan (Pearson corr sa viability = {corr:.3f}).")
            else:
                _echo("Upozorenje: nedovoljno podudarih vrednosti za robustnu korelaciju – preskačem orijentaciju.")
        except Exception as e:
            _echo(f"Upozorenje: neuspeh pri računu korelacije (preskačem orijentaciju): {e}")
    else:
        _echo("Info: nema kolone 'viability_pct' u referentnim podacima – preskačem automatsku orijentaciju.")

    # Robustno skaliranje
    p_lo, p_hi = robust_percentiles
    ref_for_scale = ref_df[score_raw.name] if (ref_df is not None and score_raw.name in ref_df.columns) else aligned
    try:
        lo = np.nanpercentile(ref_for_scale, p_lo)
        hi = np.nanpercentile(ref_for_scale, p_hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("Degenerisan opseg za percentilno skaliranje.")
    except Exception:
        lo = np.nanmin(aligned)
        hi = np.nanmax(aligned)
        _echo("Upozorenje: fallback na min–max skaliranje (nije uspelo percentilno).")
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(aligned)), float(np.nanmax(aligned))

    score_0_100 = 100.0 * (aligned - lo) / (hi - lo)
    score_0_100 = np.clip(score_0_100, 0.0, 100.0)
    score_0_100 = pd.Series(score_0_100, index=aligned.index, name="RDI_v5_score_0_100")

    meta = {"corr_with_viability": float(corr) if np.isfinite(corr) else None,
            "scale_low": float(lo), "scale_high": float(hi),
            "percentiles_used": list(robust_percentiles)}
    return aligned.rename("RDI_v5"), score_0_100, meta

def try_quick_start():
    """
    Quick-start: ako se skripta pokrene bez argumenata,
    pokušaj sa lokalnim podrazumevanim imenima fajlova.
    """
    cwd = os.getcwd()
    in_csv = os.path.join(cwd, "redox_data_normalized.csv")
    w_csv  = os.path.join(cwd, "ai_learned_weights_v5.csv")
    out_csv = os.path.join(cwd, "redox_data_with_RDI_v5.csv")

    ok_in = os.path.exists(in_csv)
    ok_w  = os.path.exists(w_csv)
    if ok_in and ok_w:
        _echo("Quick-start: pronađeni podrazumevani fajlovi u trenutnom folderu.")
        return in_csv, w_csv, out_csv
    return None, None, None

def main():
    ap = argparse.ArgumentParser(description="Compute RDI_v5 from normalized CSV and AI weights.")
    ap.add_argument("--input-csv", help="Putanja do *normalizovanog* CSV (npr. redox_data_normalized.csv)")
    ap.add_argument("--weights-csv", help="Putanja do weights CSV (ai_learned_weights_v5.csv)")
    ap.add_argument("--output-csv", help="Gde da snimi rezultat (default: redox_data_with_RDI_v5.csv)")
    ap.add_argument("--train-csv", help="(Opcionalno) referentni CSV za orijentaciju i skaliranje (ako ima viability)")
    ap.add_argument("--robust-low", type=float, default=5.0, help="Donji percentil za robustno skaliranje (default 5)")
    ap.add_argument("--robust-high", type=float, default=95.0, help="Gornji percentil za robustno skaliranje (default 95)")
    args = ap.parse_args()

    # Ako nema argumenata, probaj quick-start
    if len(sys.argv) == 1:
        in_csv, w_csv, out_csv = try_quick_start()
        if in_csv is None:
            ap.print_help()
            sys.exit(1)
        args.input_csv = in_csv
        args.weights_csv = w_csv
        args.output_csv = out_csv

    # Validacije
    if not args.input_csv or not os.path.exists(args.input_csv):
        raise SystemExit("Nije pronađen --input-csv (normalizovani CSV).")
    if not args.weights_csv or not os.path.exists(args.weights_csv):
        raise SystemExit("Nije pronađen --weights-csv (ponderi).")
    if not args.output_csv:
        # default: u isti folder kao ulaz
        base_dir = os.path.dirname(os.path.abspath(args.input_csv))
        args.output_csv = os.path.join(base_dir, "redox_data_with_RDI_v5.csv")

    _echo(f"Učitavam ulaz: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    _echo(f"Učitavam pondere: {args.weights_csv}")
    weights = load_weights(args.weights_csv)

    # (Opcionalno) učitaj train CSV za orijentaciju/skaliranje
    ref_df = None
    if args.train_csv:
        if os.path.exists(args.train_csv):
            _echo(f"Učitavam train CSV (za orijentaciju i skaliranje): {args.train_csv}")
            ref_df = pd.read_csv(args.train_csv)
            # Ako train nema naš score, to nije problem (koristiće aligned kasnije)
            # Ako train nema viability, opet nije problem (preskače orijentaciju)
        else:
            _echo("Upozorenje: --train-csv naveden ali fajl nije nađen. Nastavljam bez njega.")

    # 1) Linearni skor
    score_raw = compute_linear_score(df, weights)
    df["RDI_v5_raw"] = score_raw

    # 2) Orijentacija + skaliranje na 0–100
    aligned, score_0_100, meta = orient_and_scale(
        score_raw, ref_df if ref_df is not None else df,
        robust_percentiles=(args.robust_low, args.robust_high)
    )
    df["RDI_v5"] = aligned
    df["RDI_v5_score_0_100"] = score_0_100

    # 3) Sačuvaj
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    _echo(f"Sačuvano: {args.output_csv}")
    _echo(f"Meta: {meta}")

if __name__ == "__main__":
    main()

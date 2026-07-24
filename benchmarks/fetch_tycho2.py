"""
Download a magnitude-limited Tycho-2 subset from VizieR (I/259/tyc2) and cache it.

Tycho-2 is far denser than the Yale BSC, so a small 7x4 deg field holds enough stars
to solve. We keep V <= --vmax (default 8.0, with margin above the 7.5 validity gate).
Johnson V is approximated from Tycho photometry: V = VT - 0.090*(BT-VT).

    python benchmarks/fetch_tycho2.py --vmax 8.0 --out data/tycho2.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.vizier import Vizier


def fetch(vmax: float) -> pd.DataFrame:
    """Queries VizieR I/259/tyc2 for stars with Johnson V <= vmax; returns a clean DataFrame."""
    # Query a little fainter in VT than the V cut, since V is usually a touch brighter than VT.
    v = Vizier(columns=["RAmdeg", "DEmdeg", "BTmag", "VTmag"],
               column_filters={"VTmag": f"<{vmax + 0.4:.2f}"})
    v.ROW_LIMIT = -1
    print(f"Querying VizieR I/259/tyc2 (VTmag < {vmax + 0.4:.2f}) ... this can take a minute")
    tables = v.get_catalogs("I/259/tyc2")
    tab = tables[0].to_pandas()
    print(f"  raw rows: {len(tab)}")

    ra = tab["RAmdeg"].to_numpy(dtype=float)
    dec = tab["DEmdeg"].to_numpy(dtype=float)
    vt = tab["VTmag"].to_numpy(dtype=float)
    bt = tab["BTmag"].to_numpy(dtype=float)
    # Johnson V; fall back to VT where BT is missing.
    vmag = np.where(np.isfinite(bt), vt - 0.090 * (bt - vt), vt)

    keep = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(vmag) & (vmag <= vmax)
    df = pd.DataFrame({"RA_deg": ra[keep], "DEC_deg": dec[keep], "Vmag": vmag[keep]})
    df = df.sort_values("Vmag").reset_index(drop=True)
    df.insert(0, "HR_clean", np.arange(len(df), dtype=int))  # sequential id 0..N-1
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Cache a magnitude-limited Tycho-2 subset")
    p.add_argument("--vmax", type=float, default=8.0)
    p.add_argument("--out", type=Path, default=Path("data/tycho2.parquet"))
    args = p.parse_args()

    df = fetch(args.vmax)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)
    print(f"  kept {len(df)} stars (V <= {args.vmax}); wrote {args.out}")
    print(f"  V range {df['Vmag'].min():.2f}..{df['Vmag'].max():.2f}; "
          f"density {len(df) / 41253:.2f}/deg^2")


if __name__ == "__main__":
    main()

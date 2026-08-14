import pandas as pd
from sklearn.metrics import cohen_kappa_score

CONSTRUCTS = ["autonomy","trust","surveillance_monitoring","psych_safety",
              "technostress","isolation","work_life_balance","leadership"]

pipe = pd.read_csv("construct_presence.csv")
hum  = pd.read_excel("coder_2.xlsx", sheet_name="Sheet1")

pipe = pipe.rename(columns={pipe.columns[0]: "filename"})
hum  = hum.rename(columns={hum.columns[0]: "filename"})
hum["filename"]  = hum["filename"].astype(str).str.strip()
pipe["filename"] = pipe["filename"].astype(str).str.strip()

pipe_cols = {
    "autonomy":"autonomy","trust":"trust","surveillance_monitoring":"surveillance",
    "psych_safety":"psychological safety","technostress":"technostress",
    "isolation":"isolation","work_life_balance":"work-life balance","leadership":"leadership",
}

print("pipeline rows:", len(pipe), "coder rows:", len(hum))
print("pipeline cols:", list(pipe.columns)[:12])
print("coder cols:", list(hum.columns)[:12])

m = hum.merge(pipe, on="filename", suffixes=("_h","_p"))
print("matched papers:", len(m))

if len(m) == 0:
    print("JOIN FAILED - sample filenames:")
    print(" coder :", hum["filename"].head(3).tolist())
    print(" pipe  :", pipe["filename"].head(3).tolist())
else:
    print()
    print(f"{'construct':24s} {'agree%':>7s} {'kappa':>7s}")
    ks = []
    for c in CONSTRUCTS:
        pcol = pipe_cols[c]
        if f"V2_{c}" not in m.columns or pcol not in m.columns:
            print(f"{c:24s}  MISSING ({pcol})")
            continue
        h = (m[f"V2_{c}"].fillna(0).astype(float) > 0).astype(int)
        p = (m[pcol].fillna(0).astype(float) > 0).astype(int)
        agree = (h == p).mean() * 100
        try:
            k = cohen_kappa_score(h, p)
        except Exception:
            k = float("nan")
        ks.append(k)
        print(f"{c:24s} {agree:6.1f} {k:7.3f}")
    if ks:
        print()
        print(f"MEAN kappa = {sum(ks)/len(ks):.3f}   (n={len(m)})")

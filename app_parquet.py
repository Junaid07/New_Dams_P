
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os

st.set_page_config(page_title="Small Dams — Overview & Filters", page_icon="💧", layout="wide")

PARQUET_PATH = "All_Dams.parquet"
XLSX_PATH = "All_Dams.xlsx"

@st.cache_data(show_spinner=False)
def load_parquet_or_convert():
    try:
        df = pd.read_parquet(PARQUET_PATH)
        loaded_from = "parquet"
    except Exception:
        df, loaded_from = None, None
    if df is None and os.path.exists(XLSX_PATH):
        import openpyxl
        df = pd.read_excel(XLSX_PATH, sheet_name=0, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        try:
            df.to_parquet(PARQUET_PATH, index=False)
            loaded_from = "xlsx→parquet"
        except Exception:
            loaded_from = "xlsx (parquet save failed)"
    if df is None:
        raise RuntimeError("Place All_Dams.parquet or All_Dams.xlsx next to the app.")
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    alias = {c.lower().replace("\\n"," ").replace("  "," ").strip(): c for c in df.columns}
    return df, alias, loaded_from

def best_match(alias, include_words, exclude_words=None):
    if exclude_words is None:
        exclude_words = []
    include_words = [w.lower() for w in include_words]
    exclude_words = [w.lower() for w in exclude_words]
    candidates = []
    for akey, original in alias.items():
        if all(w in akey for w in include_words) and not any(w in akey for w in exclude_words):
            candidates.append((len(akey), original))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

def coerce_num(series):
    if series is None:
        return None
    if series.dtype == object:
        series = series.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(series, errors="coerce")

@st.cache_data(show_spinner=False)
def prepare(df, col_district, col_name, col_height, col_cca, col_year):
    df[col_name] = df[col_name].replace({"": np.nan, "nan": np.nan})
    if col_height: df[col_height] = coerce_num(df[col_height]).astype("float32")
    if col_cca:    df[col_cca]    = coerce_num(df[col_cca]).astype("float32")
    if col_year:
        years = coerce_num(df[col_year])
        df["Age (years)"] = (datetime.now().year - years).astype("Int16")
    else:
        df["Age (years)"] = pd.Series([np.nan]*len(df), dtype="float32")
    df[col_district] = df[col_district].astype("category")
    df[col_name]     = df[col_name].astype("category")
    return df

# ---- Load & columns (same as v6) ----
df, alias, loaded_from = load_parquet_or_convert()

col_district = best_match(alias, ["district"])
col_name     = best_match(alias, ["name of dam"]) or best_match(alias, ["dam name"])
col_height   = best_match(alias, ["height"])
col_cca      = best_match(alias, ["c.c.a"]) or best_match(alias, ["cca"])
col_year     = best_match(alias, ["year of completion"]) or best_match(alias, ["year"])
col_type     = best_match(alias, ["type of dam"])
col_cost     = best_match(alias, ["completion cost"])
col_gross    = best_match(alias, ["gross storage capacity"])
col_live     = best_match(alias, ["live storage"])
col_cap_ch   = best_match(alias, ["capacity of channel"])
col_len_can  = best_match(alias, ["length of canal"])
col_dsl      = best_match(alias, ["dsl"], exclude_words=["length of canal"])
col_npl      = best_match(alias, ["npl"])
col_hfl      = best_match(alias, ["hfl"])
col_river    = best_match(alias, ["river"]) or best_match(alias, ["nullah"])
col_catch    = best_match(alias, ["catchment area"])

required = [col_district, col_name]
missing = [c for c in required if c is None]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = prepare(df, col_district, col_name, col_height, col_cca, col_year)

# ---- UI header ----
st.markdown("<h2>Explore & Filter (names list)</h2>", unsafe_allow_html=True)

work = df.copy()

def range_filter(label, col):
    s = coerce_num(work[col]) if col else None
    if s is None:
        return None
    mn = float(np.nanmin(s)) if np.isfinite(np.nanmin(s)) else 0.0
    mx = float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else 0.0
    if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
        return None
    val = st.slider(label, min_value=float(mn), max_value=float(mx), value=(float(mn), float(mx)))
    return (s.between(val[0], val[1])) | (s.isna())

colA, colB = st.columns(2)
masks = []

with colA:
    masks.append(range_filter("Height (ft) range", col_height))
    masks.append(range_filter("Gross Storage Capacity (Aft) range", col_gross))
    masks.append(range_filter("C.C.A. (Acres) range", col_cca))
    masks.append(range_filter("Length of Canal (ft) range", col_len_can))
    masks.append(range_filter("DSL (ft) range", col_dsl))
with colB:
    masks.append(range_filter("Live storage (Aft) range", col_live))
    masks.append(range_filter("Capacity of Channel (Cfs) range", col_cap_ch))
    masks.append(range_filter("NPL (ft) range", col_npl))
    masks.append(range_filter("HFL (ft) range", col_hfl))
    if col_year:
        s = coerce_num(work[col_year]); 
        if s.notna().any():
            mn, mx = int(np.nanmin(s)), int(np.nanmax(s))
            yr = st.slider("Year of Completion range", min_value=int(mn), max_value=int(mx), value=(int(mn), int(mx)))
            masks.append((s.between(yr[0], yr[1])) | (s.isna()))
    if col_river:
        options = sorted([x for x in work[col_river].dropna().unique().tolist() if x])
        sel = st.multiselect("River / Nullah", options)
        if sel: masks.append(work[col_river].isin(sel))
        else: masks.append(None)
    masks.append(range_filter("Catchment Area (Sq. Km) range", col_catch))

filt = pd.Series(True, index=work.index)
for m in masks:
    if m is not None:
        filt = filt & m
filtered = work[filt].copy()

st.markdown("---")
c1, c2 = st.columns([1,1])
with c1:
    st.metric("Total dams (filtered)", f"{len(filtered):,}")
with c2:
    if col_district:
        counts = filtered.groupby(col_district).size().sort_values(ascending=False).reset_index()
        counts.columns = ["District", "Dams"]
        st.markdown("**Dams by District (filtered)**")
        st.table(counts)

# ---- Names list instead of dataframe to avoid CSS error ----
st.markdown("**Dams (filtered)**")
names = filtered[col_name].astype(str).tolist() if col_name else []
if names:
    # Show as bullets (cap at 200 to keep UI snappy)
    max_show = 200
    bullets = "\\n".join([f"- {n}" for n in names[:max_show]])
    if len(names) > max_show:
        bullets += f"\\n- ... and {len(names)-max_show} more"
    st.markdown(bullets)
else:
    st.info("No dams match the current filters.")

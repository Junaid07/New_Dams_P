
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os

st.set_page_config(page_title="Small Dams — Cards, Comparisons & Highlights", page_icon="💧", layout="wide")

PARQUET_PATH = "All_Dams.parquet"
XLSX_PATH = "All_Dams.xlsx"

# -------------------------
# Loaders
# -------------------------
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
    # Normalize strings and header aliases
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

def build_rank_table(frame, col_name, col_district, metric_col, nice_label, top_n=10):
    t = frame[[col_name, col_district, metric_col]].dropna(subset=[metric_col]).copy()
    t[col_name] = t[col_name].astype(str).replace({"nan": "—", "": "—"})
    t = t.sort_values(metric_col, ascending=False, na_position="last").head(top_n).reset_index(drop=True)
    t.index = t.index + 1
    t.index.name = "Rank"
    t = t.rename(columns={metric_col: nice_label})
    return t

def cmp_sentence(row, frame, metric_col, unit, col_name, scope_label="organization"):
    if metric_col is None or pd.isna(row.get(metric_col, np.nan)):
        return f"{unit['label']}: not available for this dam."
    value = float(row[metric_col])
    scope = frame.dropna(subset=[metric_col])
    if scope.empty:
        return f"{unit['label']}: not available in the {scope_label}."
    max_val = float(scope[metric_col].max())
    min_val = float(scope[metric_col].min())
    max_dam = scope.loc[scope[metric_col].idxmax(), col_name]
    if np.isclose(value, max_val, equal_nan=False):
        return f"{unit['label']}: This is the **highest** in the {scope_label} ({max_val:,.0f} {unit['abbr']} at {max_dam})."
    if np.isclose(value, min_val, equal_nan=False):
        return f"{unit['label']}: This is the **lowest** in the {scope_label} ({min_val:,.0f} {unit['abbr']})."
    diff = max_val - value
    return f"{unit['label']}: **{diff:,.0f} {unit['abbr']} less** than the highest in the {scope_label} ({max_val:,.0f} {unit['abbr']} at {max_dam})."

# -------------------------
# Load
# -------------------------
try:
    df, alias, loaded_from = load_parquet_or_convert()
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

# Column detection
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

# -------------------------
# Title
# -------------------------
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
      <div style="font-size:36px">💧</div>
      <div style="font-size:36px">🏞️</div>
      <div style="font-size:28px;font-weight:800;">Overview of Small Dams in Potohar Zone</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Parquet-optimized. Source: " + (loaded_from or "unknown"))

# Filters
left, right = st.columns([1,1])
with left:
    districts = ["All"] + sorted([str(x) for x in df[col_district].dropna().unique().tolist()])
    district = st.selectbox("District", districts, index=0)
scope_df = df if district == "All" else df[df[col_district].astype(str) == district]
with right:
    dams = ["All"] + sorted([str(x) for x in scope_df[col_name].dropna().unique().tolist()])
    dam = st.selectbox("Dam", dams, index=0)

st.markdown("---")
c1, c2 = st.columns([1,1])

# -------------------------
# Left: Selected + Comparisons
# -------------------------
with c1:
    st.markdown(f"## Selected Dam: {('—' if dam=='All' else dam)}")
    if dam != "All":
        row = scope_df[scope_df[col_name] == dam].iloc[0]
        st.markdown("**Organization comparison**")
        st.write(cmp_sentence(row, df, col_height, {"label":"Height (ft)", "abbr":"ft"}, col_name, "organization"))
        st.write(cmp_sentence(row, df, col_cca, {"label":"C.C.A (Acres)", "abbr":"Acres"}, col_name, "organization"))
        st.write(cmp_sentence(row, df, "Age (years)", {"label":"Age (years)", "abbr":"years"}, col_name, "organization"))
        if district != "All":
            st.markdown(f"**{district} comparison**")
            st.write(cmp_sentence(row, scope_df, col_height, {"label":"Height (ft)", "abbr":"ft"}, col_name, district))
            st.write(cmp_sentence(row, scope_df, col_cca, {"label":"C.C.A (Acres)", "abbr":"Acres"}, col_name, district))
            st.write(cmp_sentence(row, scope_df, "Age (years)", {"label":"Age (years)", "abbr":"years"}, col_name, district))
    else:
        st.info("Pick a dam to see comparison statements.")

# -------------------------
# Right: Organization Highlights
# -------------------------
with c2:
    st.markdown("## 🌟 Highlights (Organization)")

    # Helper CSS for mini-cards
    st.markdown(
        """
        <style>
        .hcard{background:#fff;border:1px solid #eee;border-radius:14px;padding:14px 16px;
               box-shadow:0 2px 8px rgba(0,0,0,0.04);margin-bottom:12px}
        .hlabel{font-size:0.9rem;color:#555;font-weight:600}
        .hvalue{font-size:1.6rem;font-weight:800;margin-top:6px;color:#111}
        .hsub{font-size:0.85rem;color:#777;margin-top:2px}
        </style>
        """,
        unsafe_allow_html=True
    )

    def card(label, value, sub=None):
        v = "—" if value is None else value
        sub_html = f'<div class="hsub">{sub}</div>' if sub else ""
        st.markdown(f'<div class="hcard"><div class="hlabel">{label}</div><div class="hvalue">{v}</div>{sub_html}</div>', unsafe_allow_html=True)

    # Precompute numeric cols for sums/max
    cca_series = coerce_num(df[col_cca]) if col_cca else None
    len_series = coerce_num(df[col_len_can]) if col_len_can else None
    cap_series = coerce_num(df[col_cap_ch]) if col_cap_ch else None
    catch_series = coerce_num(df[col_catch]) if col_catch else None

    # Total dams
    card("Total Dams", f"{len(df):,}")

    # Counts in specific districts if present
    for target in ["Islamabad", "Rawalpindi"]:
        if col_district:
            cnt = int((df[col_district].astype(str) == target).sum())
            card(f"Dams in {target}", f"{cnt:,}")

    # District with max CCA (sum)
    if col_cca:
        g = df.assign(_cca=cca_series).groupby(col_district, dropna=False)["_cca"].sum().sort_values(ascending=False)
        if len(g) > 0:
            top_dist, top_cca = g.index[0], g.iloc[0]
            card("District with Maximum CCA (sum)", f"{top_dist}", sub=f"{top_cca:,.0f} Acres")

    # District with max Length of Canals (sum)
    if col_len_can:
        g = df.assign(_len=len_series).groupby(col_district, dropna=False)["_len"].sum().sort_values(ascending=False)
        if len(g) > 0:
            top_dist, top_len = g.index[0], g.iloc[0]
            card("District with Maximum Canal Length (sum)", f"{top_dist}", sub=f"{top_len:,.0f} ft")

    # Max Capacity of Channel (max single dam)
    if col_cap_ch:
        idx = cap_series.idxmax()
        if pd.notna(idx):
            max_cap = cap_series.loc[idx]
            dam_name = df.loc[idx, col_name]
            dname = df.loc[idx, col_district]
            card("Maximum Capacity of Channel (single)", f"{max_cap:,.0f} Cfs", sub=f"{dam_name} — {dname}")

    # District with max Catchment Area (sum)
    if col_catch:
        g = df.assign(_cat=catch_series).groupby(col_district, dropna=False)["_cat"].sum().sort_values(ascending=False)
        if len(g) > 0:
            top_dist, top_catch = g.index[0], g.iloc[0]
            card("District with Maximum Catchment Area (sum)", f"{top_dist}", sub=f"{top_catch:,.0f} Sq. Km")

# -------------------------
# Details cards (same as before)
# -------------------------
st.markdown("---")
st.subheader("📋 Details")

st.markdown(
    """
    <style>
    .card {background:#fff;border:1px solid #e6e6e6;border-radius:16px;padding:16px 18px;
           box-shadow:0 2px 8px rgba(0,0,0,0.04);margin-bottom:12px;height:100%}
    .card .label {font-size:0.95rem;font-weight:600;color:#444}
    .card .value {font-size:1.6rem;font-weight:800;margin-top:6px;color:#111;word-break:break-word}
    </style>
    """,
    unsafe_allow_html=True,
)

def render_card(title, val):
    disp = "—" if pd.isna(val) or val == "" else val
    st.markdown(f'<div class="card"><div class="label">{title}</div><div class="value">{disp}</div></div>', unsafe_allow_html=True)

if dam != "All":
    row = scope_df[scope_df[col_name] == dam].iloc[0]
    def fmt_num(v, suffix=""):
        if pd.isna(v): return "—"
        try: val = float(v); return f"{val:,.0f}{(" " + suffix) if suffix else ""}"
        except Exception: return f"{v} {suffix}".strip()

    fields = [
        ("Type of Dam", row.get(col_type) if col_type else np.nan),
        ("Completion Cost (million)", fmt_num(row.get(col_cost), "" if col_cost is None else "")),
        ("Gross Storage Capacity (Aft)", fmt_num(row.get(col_gross)) if col_gross else "—"),
        ("Live storage (Aft)", fmt_num(row.get(col_live)) if col_live else "—"),
        ("C.C.A. (Acres)", fmt_num(row.get(col_cca)) if col_cca else "—"),
        ("Capacity of Channel (Cfs)", fmt_num(row.get(col_cap_ch)) if col_cap_ch else "—"),
        ("Length of Canal (ft)", fmt_num(row.get(col_len_can), "ft") if col_len_can else "—"),
        ("DSL (ft)", fmt_num(row.get(col_dsl), "ft") if col_dsl else "—"),
        ("NPL (ft)", fmt_num(row.get(col_npl), "ft") if col_npl else "—"),
        ("HFL (ft)", fmt_num(row.get(col_hfl), "ft") if col_hfl else "—"),
        ("River / Nullah", row.get(col_river) if col_river else np.nan),
        ("Year of Completion", fmt_num(row.get(col_year), "")),
        ("Catchment Area (Sq. Km)", fmt_num(row.get(col_catch)) if col_catch else "—"),
    ]
    ncols = 4
    for i in range(0, len(fields), ncols):
        cols = st.columns(ncols)
        for j, (label, value) in enumerate(fields[i:i+ncols]):
            with cols[j]:
                render_card(label, value)

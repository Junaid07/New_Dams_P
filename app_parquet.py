
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from datetime import datetime
import os

st.set_page_config(page_title="Small Dams (Parquet Fast)", page_icon="🏞️", layout="wide")

PARQUET_PATH = "All_Dams.parquet"
XLSX_PATH = "All_Dams.xlsx"

# -------------------------
# Loaders
# -------------------------
@st.cache_data(show_spinner=False)
def load_parquet_or_convert():
    """
    Load from Parquet for speed.
    If Parquet is missing but XLSX exists, read XLSX once and cache in Parquet.
    """
    # Try Parquet first
    try:
        df = pd.read_parquet(PARQUET_PATH)  # needs pyarrow or fastparquet
        loaded_from = "parquet"
    except Exception:
        df = None
        loaded_from = None

    # Fallback: build parquet from xlsx if needed
    if df is None and os.path.exists(XLSX_PATH):
        try:
            import openpyxl  # fail early with nice error if missing
            df = pd.read_excel(XLSX_PATH, sheet_name=0, engine="openpyxl")
            df.columns = [str(c).strip() for c in df.columns]
            try:
                df.to_parquet(PARQUET_PATH, index=False)
                loaded_from = "xlsx->parquet"
            except Exception:
                loaded_from = "xlsx (parquet save failed)"
        except Exception as e:
            raise RuntimeError(f"Could not load Parquet and XLSX fallback failed: {e}")

    if df is None:
        raise RuntimeError("Neither Parquet nor XLSX could be loaded. Place All_Dams.parquet or All_Dams.xlsx next to the app.")

    # Normalize strings
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    alias = {c.lower().replace("\\n"," "): c for c in df.columns}
    return df, alias, loaded_from

def find_col(alias, keys):
    for k in alias.keys():
        for t in keys:
            if t in k:
                return alias[k]
    return None

def coerce_num(series):
    if series.dtype == object:
        series = series.str.replace(",", "", regex=False).str.replace("%","", regex=False)
    return pd.to_numeric(series, errors="coerce")

@st.cache_data(show_spinner=False)
def prepare(df, col_district, col_name, col_height, col_cca, col_year, col_lat, col_lon):
    df[col_name] = df[col_name].replace({"": np.nan, "nan": np.nan})
    df = df.dropna(subset=[col_lat, col_lon]).copy()
    df[col_lat] = coerce_num(df[col_lat]).astype("float32")
    df[col_lon] = coerce_num(df[col_lon]).astype("float32")

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

def comparison_text(row, frame, metric_col, nice_label, col_name, scope_label="Organization"):
    if metric_col is None or pd.isna(row.get(metric_col, np.nan)):
        return f"{nice_label}: not available for this dam."
    value = float(row[metric_col])
    scope = frame.dropna(subset=[metric_col])
    if scope.empty:
        return f"{nice_label}: not available in {scope_label}."
    max_val = scope[metric_col].max()
    min_val = scope[metric_col].min()
    max_dam = scope.loc[scope[metric_col].idxmax(), col_name]
    if np.isclose(value, max_val, equal_nan=False):
        return f"{nice_label}: This is the **highest** in the {scope_label} ({value:,.0f})."
    if np.isclose(value, min_val, equal_nan=False):
        return f"{nice_label}: This is the **lowest** in the {scope_label} ({value:,.0f})."
    diff = max_val - value
    return f"{nice_label}: {value:,.0f} — **{diff:,.0f} lower** than the highest in the {scope_label} ({max_val:,.0f} at {max_dam})."

# -------------------------
# Load
# -------------------------
try:
    df, alias, loaded_from = load_parquet_or_convert()
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

col_district = find_col(alias, ["district"])
col_name     = find_col(alias, ["name of dam", "dam name"])
col_lat      = find_col(alias, ["latitude"])
col_lon      = find_col(alias, ["longitude"])
col_height   = find_col(alias, ["height"])
col_cca      = find_col(alias, ["c.c.a", "cca"])
col_year     = find_col(alias, ["year of completion", "year"])

required = [col_district, col_name, col_lat, col_lon]
missing = [c for c in required if c is None]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = prepare(df, col_district, col_name, col_height, col_cca, col_year, col_lat, col_lon)

# -------------------------
# UI + Map
# -------------------------
st.title("🏞️ Small Dams Explorer (Parquet)")
st.caption("Loads super fast from Parquet. " + (f"Source: {loaded_from}" if loaded_from else ""))

cols = st.columns([1,1,1])
districts = ["All"] + sorted([str(x) for x in df[col_district].dropna().unique().tolist()])
with cols[0]:
    district = st.selectbox("District", districts, index=0)
scope_df = df if district == "All" else df[df[col_district].astype(str) == district]
dams = ["All"] + sorted([str(x) for x in scope_df[col_name].dropna().unique().tolist()])
with cols[1]:
    dam = st.selectbox("Dam", dams, index=0)
with cols[2]:
    perf_mode = st.toggle("Performance mode (skip map)", value=False)

st.markdown("---")
c1, c2 = st.columns([1,2])

with c1:
    st.subheader("🎯 Selected")
    if dam != "All":
        row = scope_df[scope_df[col_name] == dam].iloc[0]
        st.markdown(f"**{row[col_name]}**")
        st.markdown("**Organization comparison**")
        st.write(comparison_text(row, df, col_height, "Height (ft)", col_name, "organization"))
        st.write(comparison_text(row, df, col_cca, "C.C.A (Acres)", col_name, "organization"))
        st.write(comparison_text(row, df, "Age (years)", "Age (years)", col_name, "organization"))
        if district != "All":
            st.markdown(f"**{district} comparison**")
            st.write(comparison_text(row, scope_df, col_height, "Height (ft)", col_name, f"{district}"))
            st.write(comparison_text(row, scope_df, col_cca, "C.C.A (Acres)", col_name, f"{district}"))
            st.write(comparison_text(row, scope_df, "Age (years)", "Age (years)", col_name, f"{district}"))
    else:
        st.info("Pick a dam to see comparison statements.")

with c2:
    st.subheader("🗺️ Map")
    if not perf_mode:
        center_lat = float(scope_df[col_lat].median()) if len(scope_df) else 30.0
        center_lon = float(scope_df[col_lon].median()) if len(scope_df) else 70.0
        mdf = scope_df[[col_lon, col_lat, col_name, col_district]].copy()
        mdf[col_name] = mdf[col_name].astype(str).replace({"nan": "—", "": "—"})
        mdf["is_selected"] = (mdf[col_name] == dam) if dam != "All" else False

        base = pdk.Layer(
            "TileLayer",
            data="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
            subdomains=["a","b","c","d"],
        )

        layer_all = pdk.Layer(
            "ScatterplotLayer",
            data=mdf[~mdf["is_selected"]],
            get_position=[col_lon, col_lat],
            get_radius=2000,
            get_fill_color=[30, 30, 30, 160],
            get_line_color=[255,255,255,120],
            line_width_min_pixels=0.5,
            pickable=True,
        )

        layer_sel = pdk.Layer(
            "ScatterplotLayer",
            data=mdf[mdf["is_selected"]],
            get_position=[col_lon, col_lat],
            get_radius=4000,
            get_fill_color=[255, 215, 0, 220],
            get_line_color=[255,255,255,255],
            line_width_min_pixels=2,
            pickable=True,
        )

        tooltip = {"html": f"<b>{col_name}</b>: {{{{{col_name}}}}}<br/><b>District</b>: {{{{{col_district}}}}}"}

        deck = pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=8),
            layers=[base, layer_all, layer_sel],
            tooltip=tooltip,
            map_provider=None,
        )
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.info("Performance mode is ON — map skipped.")

# Rankings
st.markdown("---")
tabs = st.tabs(["Organization-wide"] + ([f"{district} only"] if district != "All" else []))

with tabs[0]:
    st.markdown("### Top by Height (ft)")
    st.dataframe(build_rank_table(df, col_name, col_district, col_height, "Height (ft)"), use_container_width=True)
    st.markdown("### Top by C.C.A (Acres)")
    st.dataframe(build_rank_table(df, col_name, col_district, col_cca, "C.C.A (Acres)"), use_container_width=True)
    st.markdown("### Top by Age (years)")
    st.dataframe(build_rank_table(df, col_name, col_district, "Age (years)", "Age (years)"), use_container_width=True)

if district != "All":
    with tabs[1]:
        st.markdown(f"### Top by Height (ft) — {district}")
        st.dataframe(build_rank_table(scope_df, col_name, col_district, col_height, "Height (ft)"), use_container_width=True)
        st.markdown(f"### Top by C.C.A (Acres) — {district}")
        st.dataframe(build_rank_table(scope_df, col_name, col_district, col_cca, "C.C.A (Acres)"), use_container_width=True)
        st.markdown(f"### Top by Age (years) — {district}")
        st.dataframe(build_rank_table(scope_df, col_name, col_district, "Age (years)", "Age (years)"), use_container_width=True)

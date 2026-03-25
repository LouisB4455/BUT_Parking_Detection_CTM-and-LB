import os
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st


RESULT_CSV_DEFAULT = "resultats_modele_final.csv"
MANUAL_CSV_DEFAULT = "check_manuel_results.csv"
IMAGE_PREFIX = "ModeleFinal_"

ERROR_LABELS = {
    "err1": "Voiture non detectee",
    "err2": "Fausse detection",
    "err3": "Stationnement sauvage",
    "err4": "Voiture partielle",
    "err5": "Image inexploitable",
    "err6": "Obstacle non voiture",
    "err7": "Place non visible",
    "err8": "Voiture sur 2 places",
    "err9": "Double detection",
    "err10": "Erreur cadrage de place",
}


def normalize_image_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    base = os.path.basename(name)
    if base.startswith(IMAGE_PREFIX):
        return base[len(IMAGE_PREFIX) :]
    return base


def extract_timestamp_from_name(name: str):
    normalized = normalize_image_name(name)
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4})", normalized)
    if not m:
        return pd.NaT
    date_part = m.group(1)
    hhmm = m.group(2)
    try:
        return datetime.strptime(f"{date_part} {hhmm}", "%Y-%m-%d %H%M")
    except ValueError:
        return pd.NaT


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["image_key"] = df["image"].map(normalize_image_name)
    df["timestamp"] = df["image_key"].map(extract_timestamp_from_name)

    numeric_cols = [
        "free_places",
        "occupied_places",
        "total_places",
        "cars_detected",
        "illegal_parked",
        "alignment_ok",
        "match_count",
        "inliers",
        "inlier_ratio",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["occupancy_rate"] = (df["occupied_places"] / df["total_places"]) * 100.0
    df = df.sort_values(by=["timestamp", "image_key"], na_position="last").reset_index(drop=True)
    return df


def load_manual(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    mdf = pd.read_csv(path)
    if "image" not in mdf.columns:
        return pd.DataFrame()

    mdf["image_key"] = mdf["image"].map(normalize_image_name)
    mdf["timestamp"] = mdf["image_key"].map(extract_timestamp_from_name)

    for i in range(1, 11):
        col = f"err{i}"
        if col not in mdf.columns:
            mdf[col] = 0
        mdf[col] = pd.to_numeric(mdf[col], errors="coerce").fillna(0).astype(int)

    if "places_detectees" not in mdf.columns:
        mdf["places_detectees"] = 0
    mdf["places_detectees"] = pd.to_numeric(mdf["places_detectees"], errors="coerce").fillna(0)

    # Keep latest manual pass per image.
    mdf = mdf.drop_duplicates(subset=["image_key"], keep="last")
    mdf = mdf.sort_values(by=["timestamp", "image_key"], na_position="last").reset_index(drop=True)
    return mdf


def render_kpis(df: pd.DataFrame) -> None:
    total_images = int(len(df))
    avg_occupied = float(df["occupied_places"].mean()) if total_images else 0.0
    avg_free = float(df["free_places"].mean()) if total_images else 0.0
    avg_illegal = float(df["illegal_parked"].mean()) if total_images else 0.0
    align_rate = float(df["alignment_ok"].mean() * 100.0) if total_images else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Images", f"{total_images}")
    c2.metric("Occupees moy.", f"{avg_occupied:.1f}")
    c3.metric("Libres moy.", f"{avg_free:.1f}")
    c4.metric("Illegales moy.", f"{avg_illegal:.2f}")
    c5.metric("Recalage OK", f"{align_rate:.1f}%")


def render_main_charts(df: pd.DataFrame) -> None:
    st.subheader("Evolution des places")
    curve_df = df[["timestamp", "occupied_places", "free_places", "cars_detected"]].copy()
    curve_df = curve_df.melt(
        id_vars=["timestamp"],
        value_vars=["occupied_places", "free_places", "cars_detected"],
        var_name="serie",
        value_name="valeur",
    )
    fig_places = px.line(
        curve_df,
        x="timestamp",
        y="valeur",
        color="serie",
        markers=True,
        title="Places occupees/libres et voitures detectees",
    )
    st.plotly_chart(fig_places, use_container_width=True)

    st.subheader("Taux d'occupation")
    fig_occ = px.area(
        df,
        x="timestamp",
        y="occupancy_rate",
        title="Courbe du taux d'occupation (%)",
    )
    fig_occ.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_occ, use_container_width=True)

    c1, c2 = st.columns(2)

    fig_illegal = px.bar(
        df,
        x="timestamp",
        y="illegal_parked",
        title="Stationnements illegaux par image",
    )
    c1.plotly_chart(fig_illegal, use_container_width=True)

    fig_align = px.scatter(
        df,
        x="timestamp",
        y="inlier_ratio",
        color=df["alignment_ok"].map({1: "OK", 0: "KO"}),
        title="Qualite du recalage (inlier ratio)",
        labels={"color": "recalage"},
    )
    c2.plotly_chart(fig_align, use_container_width=True)


def render_manual_section(df: pd.DataFrame, manual_df: pd.DataFrame) -> None:
    st.subheader("Controle manuel")
    if manual_df.empty:
        st.info("Aucun check manuel detecte.")
        return

    error_cols = [f"err{i}" for i in range(1, 11)]
    totals = manual_df[error_cols].sum().rename(index=ERROR_LABELS).reset_index()
    totals.columns = ["categorie", "total"]

    fig_errors = px.bar(
        totals,
        x="categorie",
        y="total",
        title="Total des erreurs manuelles par categorie",
    )
    st.plotly_chart(fig_errors, use_container_width=True)

    merged = df.merge(
        manual_df[["image_key", "places_detectees"] + error_cols],
        on="image_key",
        how="left",
        suffixes=("", "_manual"),
    )

    merged["ecart_places_detectees"] = merged["free_places"] - merged["places_detectees"].fillna(0)

    fig_gap = px.bar(
        merged,
        x="timestamp",
        y="ecart_places_detectees",
        title="Ecart: free_places modele - places_detectees manuel",
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    cols_to_show = [
        "image",
        "free_places",
        "occupied_places",
        "cars_detected",
        "illegal_parked",
        "alignment_ok",
        "places_detectees",
        "ecart_places_detectees",
    ]
    st.dataframe(merged[cols_to_show], use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Monitoring Modele Final", layout="wide")
    st.title("Monitoring Parking - Modele Final")

    result_csv = RESULT_CSV_DEFAULT
    manual_csv = MANUAL_CSV_DEFAULT

    if not os.path.exists(result_csv):
        st.error(f"CSV resultats introuvable: {result_csv}")
        st.stop()

    df = load_results(result_csv)
    manual_df = load_manual(manual_csv)

    if df.empty:
        st.warning("Le CSV resultats est vide.")
        st.stop()

    with st.sidebar:
        st.header("Monitoring")
        st.write("Mode sans input: sources fixes")
        st.write(f"Resultats: {result_csv}")
        st.write(f"Check manuel: {manual_csv}")
        st.markdown("---")
        st.write(f"Images chargees: {len(df)}")
        if not manual_df.empty:
            st.write(f"Checks manuels uniques: {len(manual_df)}")

    render_kpis(df)
    render_main_charts(df)
    render_manual_section(df, manual_df)


if __name__ == "__main__":
    main()

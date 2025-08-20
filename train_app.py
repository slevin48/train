from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (
    PUNCTUALITY_JSON,
    PRICING_JSON,
    STATIONS_JSON,
    analyse_pricing,
    analyse_punctuality,
    analyse_ridership,
    estimate_emissions,
    load_json_records,
    load_ridership_csv,
)

# ---------------------------------------------
# Data loading (cached)
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> dict:
    """Load all datasets and compute intermediate results (robust + cached)."""
    # Ridership: let analysis.load_ridership_csv resolve paths / download / fallback
    df_ridership = load_ridership_csv(None)
    totals, top10 = analyse_ridership(df_ridership)

    # Punctuality sample
    df_punctuality = load_json_records(PUNCTUALITY_JSON)
    punctuality_summary, cause_means = analyse_punctuality(df_punctuality)

    # Pricing + stations samples
    df_price = load_json_records(PRICING_JSON)
    # Coerce types
    df_price["prix_min"] = pd.to_numeric(df_price["prix_min"], errors="coerce")
    df_price["prix_max"] = pd.to_numeric(df_price["prix_max"], errors="coerce")
    df_price["origine_uic"] = pd.to_numeric(df_price["origine_uic"], errors="coerce").astype("Int64")
    df_price["destination_uic"] = pd.to_numeric(df_price["destination_uic"], errors="coerce").astype("Int64")

    df_stations = load_json_records(STATIONS_JSON)
    df_price = analyse_pricing(df_price, df_stations)

    return {
        "ridership_totals": totals,
        "ridership_top10": top10,
        "punctuality_summary": punctuality_summary,
        "cause_means": cause_means,
        "pricing": df_price,
    }


# ---------------------------------------------
# UI
# ---------------------------------------------
def main():
    st.set_page_config(page_title="Portefeuille Analyse SNCF", layout="wide")
    st.title("Portefeuille d'analyse de données SNCF")
    st.markdown(
        """
        Ce tableau de bord interactif présente une sélection d'analyses
        réalisées à partir des données ouvertes de la SNCF. Les objectifs
        incluent l'optimisation des trajets, l'analyse de la ponctualité, l'étude
        de la fréquentation et l'évaluation de l'impact environnemental.
        """
    )

    data = load_data()

    # ---------------- Ridership ----------------
    st.header("Fréquentation des gares")
    st.subheader("Évolution annuelle du nombre de voyageurs")
    totals = data["ridership_totals"].reset_index().rename(columns={"index": "année"})
    totals["total_passengers_m"] = totals["total_passengers"] / 1e6
    fig_trend = px.line(
        totals,
        x="année",
        y="total_passengers_m",
        markers=True,
        labels={"année": "Année", "total_passengers_m": "Voyageurs (millions)"},
        title="Évolution annuelle de la fréquentation (millions de voyageurs)",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Top 10 des gares les plus fréquentées (année la plus récente)")
    top10 = data["ridership_top10"].copy()
    ycol = "total_voyageurs_2024" if "total_voyageurs_2024" in top10.columns else top10.columns[-1]
    fig_top = px.bar(
        top10.sort_values(ycol),
        x=ycol,
        y="nom_gare",
        orientation="h",
        labels={ycol: "Voyageurs", "nom_gare": "Gare"},
        title="Top 10 des gares",
    )
    st.plotly_chart(fig_top, use_container_width=True)

    # ---------------- Punctuality ----------------
    st.header("Ponctualité des TGV (échantillon)")
    st.caption("Échantillon de 10 liaisons illustrant des indicateurs clés.")
    st.write("**Synthèse**", data["punctuality_summary"])
    cause_means = data["cause_means"].reset_index().rename(columns={"index": "cause", 0: "pourcentage"})
    fig_cause = px.bar(
        cause_means.sort_values("pourcentage"),
        x="pourcentage",
        y="cause",
        orientation="h",
        labels={"pourcentage": "Pourcentage (%)", "cause": "Cause"},
        title="Répartition moyenne des causes de retard (échantillon)",
    )
    st.plotly_chart(fig_cause, use_container_width=True)

    # ---------------- Pricing ----------------
    st.header("Tarifs Intercités (échantillon)")
    df_price = data["pricing"].copy()
    if not df_price.empty:
        df_price["route"] = df_price["origine"] + " → " + df_price["destination"]
        fig_price = px.bar(
            df_price,
            x="route",
            y=["prix_min", "prix_max"],
            labels={"value": "Prix (€)", "route": "Liaison", "variable": "Type"},
            barmode="group",
            title="Tarifs minimum et maximum par liaison (échantillon)",
        )
        fig_price.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Données de tarifs non disponibles dans l'échantillon.")

    # ---------------- Emissions ----------------
    st.header("Impact environnemental")
    st.markdown(
        """Estimation des émissions de CO₂ pour différentes distances en utilisant
        14 gCO₂/km (train) et 55 gCO₂/km (voiture)."""
    )
    distances = [100, 300, 500, 700]
    emissions = [estimate_emissions(d) for d in distances]
    df_em = pd.DataFrame(
        {
            "distance": distances,
            "train": [e[0] for e in emissions],
            "voiture": [e[1] for e in emissions],
        }
    )
    fig_em = px.bar(
        df_em.melt(id_vars="distance", var_name="mode", value_name="emissions"),
        x="distance",
        y="emissions",
        color="mode",
        barmode="group",
        labels={"distance": "Distance (km)", "emissions": "Émissions (kg CO₂)"},
        title="Comparaison des émissions CO₂ par distance",
    )
    st.plotly_chart(fig_em, use_container_width=True)

    st.caption(
        "Conseil: pour des analyses plus riches, ajoutez le fichier complet "
        "`frequentation-gares.csv` dans `data/` du dépôt. L'app l'utilisera automatiquement."
    )


if __name__ == "__main__":
    main()

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
    st.set_page_config(
        page_title="Portefeuille Analyse SNCF",
        page_icon="🚆",
        layout="wide",
    )

    # Subtle CSS polish
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
            h1, h2, h3 {letter-spacing: 0.2px;}
            div[data-testid="stMetricValue"] {font-size: 1.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Default display settings (no sidebar)
    template = "plotly_white"
    show_grid = True

    st.title("Portefeuille d'analyse de données SNCF")
    st.markdown(
        """
        Ce tableau de bord interactif présente une sélection d'analyses réalisées à partir des
        données ouvertes de la SNCF. Explorez la fréquentation, la ponctualité, les tarifs et
        l'impact environnemental via les onglets ci-dessous.
        """
    )

    data = load_data()

    # ---------------- KPIs ----------------
    totals_df = data["ridership_totals"].copy()
    current_total = totals_df.iloc[-1, 0]
    prev_total = totals_df.iloc[-2, 0] if len(totals_df) > 1 else None
    delta_pct = ((current_total - prev_total) / prev_total * 100) if prev_total and prev_total != 0 else None
    punctuality_summary = data["punctuality_summary"]
    avg_delay = float(punctuality_summary.loc[0, "avg_delay_arrival"]) if not punctuality_summary.empty else float("nan")
    cancel_rate = float(punctuality_summary.loc[0, "cancellation_rate"]) if not punctuality_summary.empty else float("nan")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Voyageurs (année la plus récente)",
            f"{current_total/1e6:.1f} M",
            None if delta_pct is None else f"{delta_pct:.1f}% vs année précédente",
        )
    with col2:
        st.metric("Retard moyen à l'arrivée", f"{avg_delay:.1f} min")
    with col3:
        st.metric("Taux d'annulation", f"{cancel_rate*100:.2f}%")

    st.divider()

    # ---------------- Tabs ----------------
    tab_freq, tab_punct, tab_price, tab_em = st.tabs([
        "Fréquentation",
        "Ponctualité",
        "Tarifs",
        "CO₂",
    ])

    # Fréquentation
    with tab_freq:
        st.subheader("Évolution annuelle du nombre de voyageurs")
        totals = totals_df.reset_index()
        first_col = totals.columns[0]
        totals = totals.rename(columns={first_col: "année"})
        totals["total_passengers_m"] = totals["total_passengers"] / 1e6
        fig_trend = px.line(
            totals,
            x="année",
            y="total_passengers_m",
            markers=True,
            labels={"année": "Année", "total_passengers_m": "Voyageurs (millions)"},
            title="Évolution annuelle de la fréquentation (millions de voyageurs)",
        )
        fig_trend.update_layout(template=template, margin=dict(t=60, r=30, b=40, l=40))
        fig_trend.update_xaxes(showgrid=show_grid)
        fig_trend.update_yaxes(showgrid=show_grid)
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
        fig_top.update_layout(template=template, margin=dict(t=60, r=30, b=40, l=40))
        fig_top.update_xaxes(showgrid=show_grid)
        fig_top.update_yaxes(showgrid=show_grid)
        st.plotly_chart(fig_top, use_container_width=True)

    # Ponctualité
    with tab_punct:
        st.subheader("Ponctualité des TGV (échantillon)")
        st.caption("Échantillon de 10 liaisons illustrant des indicateurs clés.")
        st.dataframe(punctuality_summary, use_container_width=True)
        cause_means = data["cause_means"].reset_index().rename(columns={"index": "cause", 0: "pourcentage"})
        fig_cause = px.bar(
            cause_means.sort_values("pourcentage"),
            x="pourcentage",
            y="cause",
            orientation="h",
            labels={"pourcentage": "Pourcentage (%)", "cause": "Cause"},
            title="Répartition moyenne des causes de retard (échantillon)",
        )
        fig_cause.update_layout(template=template, margin=dict(t=60, r=30, b=40, l=40))
        fig_cause.update_xaxes(showgrid=show_grid)
        fig_cause.update_yaxes(showgrid=show_grid)
        st.plotly_chart(fig_cause, use_container_width=True)

    # Tarifs
    with tab_price:
        st.subheader("Tarifs Intercités (échantillon)")
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
            fig_price.update_layout(template=template, xaxis_tickangle=-45, margin=dict(t=60, r=30, b=40, l=40))
            fig_price.update_xaxes(showgrid=show_grid)
            fig_price.update_yaxes(showgrid=show_grid)
            st.plotly_chart(fig_price, use_container_width=True)
            # Petits indicateurs complémentaires
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Liaisons (échantillon)", f"{len(df_price)}")
            with c2:
                st.metric("Prix moyen (min → max)", f"{df_price['prix_min'].mean():.0f}€ → {df_price['prix_max'].mean():.0f}€")
        else:
            st.info("Données de tarifs non disponibles dans l'échantillon.")

    # CO2
    with tab_em:
        st.subheader("Impact environnemental")
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
        fig_em.update_layout(template=template, margin=dict(t=60, r=30, b=40, l=40))
        fig_em.update_xaxes(showgrid=show_grid)
        fig_em.update_yaxes(showgrid=show_grid)
        st.plotly_chart(fig_em, use_container_width=True)

    st.caption(
        "Conseil: pour des analyses plus riches, ajoutez le fichier complet "
        "`frequentation-gares.csv` dans `data/` du dépôt. L'app l'utilisera automatiquement."
    )


if __name__ == "__main__":
    main()
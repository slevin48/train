"""Analysis of SNCF open data
This script loads and analyses various SNCF open datasets to build a data
portfolio. It demonstrates how to explore the data, compute indicators
such as punctuality, ridership trends, pricing and cost per kilometre and
estimate greenhouse-gas emissions. The results can be saved as figures
under the ``figures`` directory.

Datasets:
* ``frequentation-gares.csv`` – annual passenger counts by station (CSV).
  The file is expected under ./data/ in the repo. If missing, it is
  auto-downloaded (or a tiny synthetic fallback is used).
* Small embedded JSON samples of:
  - ``Régularité mensuelle TGV`` (punctuality)
  - ``Tarifs Intercités`` (pricing)
  - ``Liste des gares`` (station coordinates)

Emission factors from Youmatter: 14 gCO₂/passenger-km for trains and
55 gCO₂/passenger-km for cars.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------
# Robust paths (work locally & on Streamlit/Render)
# ---------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR = APP_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Optional: canonical source for the ridership CSV
RIDERSHIP_URL = "https://files.data.gouv.fr/sncf/gares/frequentation-gares.csv"


def ensure_dir(path: str | Path) -> None:
    """Ensure that a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
#  Constants: JSON samples embedded for punctuality, pricing and station list
# -----------------------------------------------------------------------------

PUNCTUALITY_JSON = r"""
{"nhits": 187062,
 "parameters": {"dataset": "regularite-mensuelle-tgv", "rows": 10, "start": 0, "timezone": "UTC"},
 "records": [
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "c86c744fc2f2387cb985d3dc6780875f",
   "fields": {"service": "International",
    "nb_train_prevu": 34,
    "nb_annulation": 0,
    "prct_cause_infra": 8.8,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 9,
    "gare_arrivee": "BASEL SBB",
    "nb_train_retard_sup_15": 0,
    "nb_train_retard_depart": 6,
    "cote": "EST",
    "prct_cause_externe": 23.5,
    "nb_train_retard_arrivee": 9,
    "prct_cause_reseau": 14.7,
    "cause_principale": "External causes",
    "retard_moyen_trains_retard_sup15": 0,
    "commentaire_principale": "…",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 26.5,
    "prct_cause_exploit": 26.5,
    "dur_moyen": 195},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "3f937c7e93353aa2c3a78ad5e747e4b7",
   "fields": {"service": "National",
    "nb_train_prevu": 167,
    "nb_annulation": 0,
    "prct_cause_infra": 5.4,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 18,
    "gare_arrivee": "MONTPELLIER SUD DE FRANCE",
    "nb_train_retard_sup_15": 5,
    "nb_train_retard_depart": 15,
    "cote": "SUD EST",
    "prct_cause_externe": 19.1,
    "nb_train_retard_arrivee": 28,
    "prct_cause_reseau": 14.8,
    "cause_principale": "Gestion de trafic",
    "retard_moyen_trains_retard_sup15": 19,
    "commentaire_principale": "\n\n",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 36.2,
    "prct_cause_exploit": 24.6,
    "dur_moyen": 295},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "41d3e034f35a3d9b7a2e4cb9be8f284e",
   "fields": {"service": "National",
    "nb_train_prevu": 165,
    "nb_annulation": 0,
    "prct_cause_infra": 11.5,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 15,
    "gare_arrivee": "MARSEILLE ST CHARLES",
    "nb_train_retard_sup_15": 8,
    "nb_train_retard_depart": 14,
    "cote": "SUD EST",
    "prct_cause_externe": 17.7,
    "nb_train_retard_arrivee": 26,
    "prct_cause_reseau": 19.5,
    "cause_principale": "Gestion de trafic",
    "retard_moyen_trains_retard_sup15": 23,
    "commentaire_principale": "\n\n",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 28.3,
    "prct_cause_exploit": 23.0,
    "dur_moyen": 243},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "9f298f74b9bdbaf0a32a54eb60cdb0d2",
   "fields": {"service": "International",
    "nb_train_prevu": 64,
    "nb_annulation": 0,
    "prct_cause_infra": 4.0,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 6,
    "gare_arrivee": "BRUSSELS MIDI",
    "nb_train_retard_sup_15": 0,
    "nb_train_retard_depart": 4,
    "cote": "NORD",
    "prct_cause_externe": 31.5,
    "nb_train_retard_arrivee": 6,
    "prct_cause_reseau": 11.8,
    "cause_principale": "External causes",
    "retard_moyen_trains_retard_sup15": 0,
    "commentaire_principale": "",
    "gare_depart": "PARIS NORD",
    "date": "2024-03-01",
    "prct_cause_gestion": 30.6,
    "prct_cause_exploit": 22.1,
    "dur_moyen": 82},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "b0b0a314c500b56693e082a418a9fdf6",
   "fields": {"service": "National",
    "nb_train_prevu": 150,
    "nb_annulation": 1,
    "prct_cause_infra": 10.0,
    "commentaire_annulation": "Mouvements sociaux",
    "retard_moyen_arrivee": 20,
    "gare_arrivee": "LYON PART DIEU",
    "nb_train_retard_sup_15": 12,
    "nb_train_retard_depart": 20,
    "cote": "SUD EST",
    "prct_cause_externe": 15.3,
    "nb_train_retard_arrivee": 40,
    "prct_cause_reseau": 12.0,
    "cause_principale": "External causes",
    "retard_moyen_trains_retard_sup15": 25,
    "commentaire_principale": "\n\n",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 30.0,
    "prct_cause_exploit": 32.7,
    "dur_moyen": 118},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "661dc53b0f0ce4456f3f7a692ca87c09",
   "fields": {"service": "National",
    "nb_train_prevu": 181,
    "nb_annulation": 0,
    "prct_cause_infra": 9.9,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 13,
    "gare_arrivee": "POITIERS",
    "nb_train_retard_sup_15": 5,
    "nb_train_retard_depart": 20,
    "cote": "SUD OUEST",
    "prct_cause_externe": 15.4,
    "nb_train_retard_arrivee": 33,
    "prct_cause_reseau": 13.2,
    "cause_principale": "Gestion de trafic",
    "retard_moyen_trains_retard_sup15": 18,
    "commentaire_principale": "",
    "gare_depart": "PARIS MONTPARNASSE",
    "date": "2024-03-01",
    "prct_cause_gestion": 41.8,
    "prct_cause_exploit": 19.7,
    "dur_moyen": 85},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "6f9841db1f9f8fe236c61b818b6d56c2",
   "fields": {"service": "National",
    "nb_train_prevu": 53,
    "nb_annulation": 0,
    "prct_cause_infra": 6.6,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 7,
    "gare_arrivee": "AVIGNON TGV",
    "nb_train_retard_sup_15": 1,
    "nb_train_retard_depart": 6,
    "cote": "SUD EST",
    "prct_cause_externe": 19.6,
    "nb_train_retard_arrivee": 8,
    "prct_cause_reseau": 17.2,
    "cause_principale": "Gestion de trafic",
    "retard_moyen_trains_retard_sup15": 16,
    "commentaire_principale": "",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 28.0,
    "prct_cause_exploit": 28.6,
    "dur_moyen": 160},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "95ab506d9480d14f141817fcdadfc424",
   "fields": {"service": "National",
    "nb_train_prevu": 45,
    "nb_annulation": 0,
    "prct_cause_infra": 20.2,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 17,
    "gare_arrivee": "PERPIGNAN",
    "nb_train_retard_sup_15": 5,
    "nb_train_retard_depart": 10,
    "cote": "SUD EST",
    "prct_cause_externe": 11.7,
    "nb_train_retard_arrivee": 12,
    "prct_cause_reseau": 17.8,
    "cause_principale": "Infrastructure",
    "retard_moyen_trains_retard_sup15": 22,
    "commentaire_principale": "",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 24.9,
    "prct_cause_exploit": 25.4,
    "dur_moyen": 255},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "7e4ae5df63543264bb9a235a3ec8120f",
   "fields": {"service": "National",
    "nb_train_prevu": 40,
    "nb_annulation": 1,
    "prct_cause_infra": 22.3,
    "commentaire_annulation": "Mouvements sociaux",
    "retard_moyen_arrivee": 21,
    "gare_arrivee": "BRIVE LA GAILLARDE",
    "nb_train_retard_sup_15": 7,
    "nb_train_retard_depart": 9,
    "cote": "SUD OUEST",
    "prct_cause_externe": 13.4,
    "nb_train_retard_arrivee": 14,
    "prct_cause_reseau": 22.4,
    "cause_principale": "Infrastructure",
    "retard_moyen_trains_retard_sup15": 23,
    "commentaire_principale": "",
    "gare_depart": "PARIS AUSTERLITZ",
    "date": "2024-03-01",
    "prct_cause_gestion": 25.0,
    "prct_cause_exploit": 25.0,
    "dur_moyen": 215},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "fa0daf912b1db59ffcfed9675231c5ea",
   "fields": {"service": "National",
    "nb_train_prevu": 37,
    "nb_annulation": 0,
    "prct_cause_infra": 13.5,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 19,
    "gare_arrivee": "NICE VILLE",
    "nb_train_retard_sup_15": 5,
    "nb_train_retard_depart": 6,
    "cote": "SUD EST",
    "prct_cause_externe": 23.0,
    "nb_train_retard_arrivee": 10,
    "prct_cause_reseau": 15.9,
    "cause_principale": "External causes",
    "retard_moyen_trains_retard_sup15": 21,
    "commentaire_principale": "",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 25.9,
    "prct_cause_exploit": 21.7,
    "dur_moyen": 216},
   "record_timestamp": "2024-05-10T03:27:39.312Z"}
 ]}
"""

PRICING_JSON = r"""
{"nhits": 301,
 "parameters": {"dataset": "tarifs-intercites", "rows": 10, "start": 0, "timezone": "UTC"},
 "records": [
  {"datasetid": "tarifs-intercites",
   "recordid": "188bc8f442cba7fa4e10355f8d564979",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "BERCY BOURGOGNE - PAYS D'AUVERGNE",
    "origine_uic": 8775800,
    "destination": "TOULOUSE MATABIAU",
    "destination_uic": 8761100,
    "profil_tarifaire": "STANDARD",
    "prix_min": 15.0,
    "prix_max": 85.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "3750b2e7af631cd253c867a8f051bcce",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "QUIMPER",
    "origine_uic": 8747100,
    "destination": "PARIS MONTPARNASSE 1 ET 2",
    "destination_uic": 8739100,
    "profil_tarifaire": "STANDARD",
    "prix_min": 19.0,
    "prix_max": 89.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "d0baf48747fc098d933cfbf23ebff68f",
   "fields": {"transporteur": "INTERCITES",
    "classe": "1re classe",
    "origine": "PARIS GARE DE LYON",
    "origine_uic": 8738240,
    "destination": "CLERMONT-FERRAND",
    "destination_uic": 8761300,
    "profil_tarifaire": "STANDARD",
    "prix_min": 25.0,
    "prix_max": 100.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "5a0672b8f7ec20c27db94038fbe8c84b",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "PARIS AUSTERLITZ",
    "origine_uic": 8768600,
    "destination": "TOURS",
    "destination_uic": 8757100,
    "profil_tarifaire": "STANDARD",
    "prix_min": 18.0,
    "prix_max": 80.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "71332c71dcdb8e4fc964d437bcc0c053",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "NANTES",
    "origine_uic": 8748100,
    "destination": "LYON PART DIEU",
    "destination_uic": 8772200,
    "profil_tarifaire": "STANDARD",
    "prix_min": 25.0,
    "prix_max": 120.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "4795338e0b71dbe82d496d065244e8c1",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "MARSEILLE ST CHARLES",
    "origine_uic": 8772205,
    "destination": "BORDEAUX ST JEAN",
    "destination_uic": 8758100,
    "profil_tarifaire": "STANDARD",
    "prix_min": 30.0,
    "prix_max": 120.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "6e01db474538b392e28712c8e753c173",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "PARIS GARE D'AUSTERLITZ",
    "origine_uic": 8768600,
    "destination": "LIMOGES BENEVENT",
    "destination_uic": 8759900,
    "profil_tarifaire": "STANDARD",
    "prix_min": 15.0,
    "prix_max": 65.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "83ea9e6baf8250471b223d71b7345cb8",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "PARIS AUSTERLITZ",
    "origine_uic": 8768600,
    "destination": "POITIERS",
    "destination_uic": 8757500,
    "profil_tarifaire": "STANDARD",
    "prix_min": 14.0,
    "prix_max": 70.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "8f943d3cdd1c5f28d1aa848f48b658b3",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "PARIS GARE DE LYON",
    "origine_uic": 8738240,
    "destination": "LYON PART DIEU",
    "destination_uic": 8772200,
    "profil_tarifaire": "STANDARD",
    "prix_min": 22.0,
    "prix_max": 100.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "95bc3c483ea8a5e97227008499ef38f7",
   "fields": {"transporteur": "INTERCITES",
    "classe": "1re classe",
    "origine": "PARIS GARE DE LYON",
    "origine_uic": 8738240,
    "destination": "LYON PART DIEU",
    "destination_uic": 8772200,
    "profil_tarifaire": "STANDARD",
    "prix_min": 45.0,
    "prix_max": 150.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"}
 ]}
"""

STATIONS_JSON = r"""
{"nhits": 3884,
 "parameters": {"dataset": "liste-des-gares", "rows": 10, "start": 0, "timezone": "UTC"},
 "records": [
  {"datasetid": "liste-des-gares",
   "recordid": "b183b6af6680498b40171fc1b9f9093a34706b",
   "fields": {"code_uic": "87382381",
    "libelle": "La Défense ",
    "geo_point_2d": [48.89343732736893, 2.2384716843545894]}},
  {"datasetid": "liste-des-gares",
   "recordid": "82615392e5077d374a438ba62c84045b",
   "fields": {"code_uic": "87688889",
    "libelle": "Aubiet",
    "geo_point_2d": [43.16042316540546, 0.8879757448941251]}},
  {"datasetid": "liste-des-gares",
   "recordid": "6984acb3bfb3b2ed9df1c7a42457ccb7",
    "fields": {"code_uic": "87571564",
    "libelle": "Marmagne",
    "geo_point_2d": [46.91350318595993, 2.382545463063257]}},
  {"datasetid": "liste-des-gares",
   "recordid": "8ac7f0af6b6e05c8fa1fce3045b50ae3",
   "fields": {"code_uic": "87798688",
    "libelle": "Montpellier Sud-de-France",
    "geo_point_2d": [43.595093596781462, 3.9135030318595994]}},
  {"datasetid": "liste-des-gares",
   "recordid": "f58e6c077bd298345a962ff3b5a0601e",
   "fields": {"code_uic": "87571182",
    "libelle": "Poitiers",
    "geo_point_2d": [46.58356285073072, 0.344242961118062]}},
  {"datasetid": "liste-des-gares",
   "recordid": "f33cea19985db58386f0c1cbf16e7e2d",
   "fields": {"code_uic": "87471003",
    "libelle": "Quimper",
    "geo_point_2d": [47.993723802589894, -4.094451602672277]}},
  {"datasetid": "liste-des-gares",
   "recordid": "fbbd196938531884e2b1fbde95bed2fa",
   "fields": {"code_uic": "8738240",
    "libelle": "Paris Gare de Lyon",
    "geo_point_2d": [48.844845603079, 2.374625723855095]}},
  {"datasetid": "liste-des-gares",
   "recordid": "c7f3c8b34577c7bb30fb2bdddc0d08a2",
   "fields": {"code_uic": "87382782",
    "libelle": "Les Coquetiers",
    "geo_point_2d": [48.91080298544214, 2.510739390533775]}},
  {"datasetid": "liste-des-gares",
   "recordid": "f5ee0dceae1b4dc0c7f7ff83d530459c",
   "fields": {"code_uic": "8748100",
    "libelle": "Nantes",
    "geo_point_2d": [47.2166243247275, -1.5515494322564984]}},
  {"datasetid": "liste-des-gares",
   "recordid": "b2a0a3a4b58e7738c35b4ee7cf0ca3c8",
   "fields": {"code_uic": "8758100",
    "libelle": "Bordeaux St Jean",
    "geo_point_2d": [44.828563181114952, -0.5566119806464477]}}
 ]}
"""


# -----------------------------------------------------------------------------
# Data loading helpers (robust)
# -----------------------------------------------------------------------------

def load_json_records(json_string: str) -> pd.DataFrame:
    """Load a JSON string of records and return a DataFrame of the 'fields' dict."""
    data = json.loads(json_string)
    records = [rec["fields"] for rec in data["records"]]
    return pd.DataFrame(records)


def _synthesize_ridership_csv() -> pd.DataFrame:
    """Very small fallback dataset to keep the app running if CSV is missing."""
    csv = StringIO(
        "nom_gare;total_voyageurs_2022;total_voyageurs_2023;total_voyageurs_2024\n"
        "Paris Gare de Lyon;78000000;82000000;85000000\n"
        "Paris Saint-Lazare;71000000;74000000;76000000\n"
        "Lyon Part-Dieu;50000000;52000000;54000000\n"
    )
    df = pd.read_csv(csv, sep=";", dtype=str)
    for col in df.columns:
        if col.startswith("total_voyageurs"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _resolve_ridership_path(preferred: Optional[str | Path]) -> Path:
    """
    Choose a readable ridership CSV path:
    1) preferred path if it exists
    2) repo's data/frequentation-gares.csv
    3) otherwise return the default path under data/ (download target)
    """
    if preferred:
        p = Path(preferred)
        if p.is_file():
            return p
    p_repo = DATA_DIR / "frequentation-gares.csv"
    if p_repo.is_file():
        return p_repo
    return p_repo  # default download/save target


def load_ridership_csv(path: str | Path | None) -> pd.DataFrame:
    """Load the ridership dataset robustly from a semicolon-separated CSV.

    Tries:
      1) provided path (if exists),
      2) APP_DIR/data/frequentation-gares.csv,
      3) auto-download from Data Gouv (optional),
      4) synthetic fallback if all else fails.
    """
    target = _resolve_ridership_path(path)

    # Try reading if it already exists
    if target.is_file():
        df = pd.read_csv(target, sep=";", dtype=str)
    else:
        # Try to auto-download (optional)
        try:
            import urllib.request
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(RIDERSHIP_URL, target.as_posix())
            df = pd.read_csv(target, sep=";", dtype=str)
        except Exception:
            df = _synthesize_ridership_csv()

    # Convert numeric columns
    for col in df.columns:
        if col.startswith("total_voyageurs"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col == "non_voyageurs":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# -----------------------------------------------------------import math
import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_dir(path: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
#  Constants: JSON samples embedded for punctuality, pricing and station list
# -----------------------------------------------------------------------------

PUNCTUALITY_JSON = r"""
{"nhits": 187062,
 "parameters": {"dataset": "regularite-mensuelle-tgv", "rows": 10, "start": 0, "timezone": "UTC"},
 "records": [
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "c86c744fc2f2387cb985d3dc6780875f",
   "fields": {"service": "International",
    "nb_train_prevu": 34,
    "nb_annulation": 0,
    "prct_cause_infra": 8.8,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 9,
    "gare_arrivee": "BASEL SBB",
    "nb_train_retard_sup_15": 0,
    "nb_train_retard_depart": 6,
    "cote": "EST",
    "prct_cause_externe": 23.5,
    "nb_train_retard_arrivee": 9,
    "prct_cause_reseau": 14.7,
    "cause_principale": "External causes",
    "retard_moyen_trains_retard_sup15": 0,
    "commentaire_principale": "…",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 26.5,
    "prct_cause_exploit": 26.5,
    "dur_moyen": 195},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "3f937c7e93353aa2c3a78ad5e747e4b7",
   "fields": {"service": "National",
    "nb_train_prevu": 167,
    "nb_annulation": 0,
    "prct_cause_infra": 5.4,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 18,
    "gare_arrivee": "MONTPELLIER SUD DE FRANCE",
    "nb_train_retard_sup_15": 5,
    "nb_train_retard_depart": 15,
    "cote": "SUD EST",
    "prct_cause_externe": 19.1,
    "nb_train_retard_arrivee": 28,
    "prct_cause_reseau": 14.8,
    "cause_principale": "Gestion de trafic",
    "retard_moyen_trains_retard_sup15": 19,
    "commentaire_principale": "\n\n",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 36.2,
    "prct_cause_exploit": 24.6,
    "dur_moyen": 295},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "41d3e034f35a3d9b7a2e4cb9be8f284e",
   "fields": {"service": "National",
    "nb_train_prevu": 165,
    "nb_annulation": 0,
    "prct_cause_infra": 11.5,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 15,
    "gare_arrivee": "MARSEILLE ST CHARLES",
    "nb_train_retard_sup_15": 8,
    "nb_train_retard_depart": 14,
    "cote": "SUD EST",
    "prct_cause_externe": 17.7,
    "nb_train_retard_arrivee": 26,
    "prct_cause_reseau": 19.5,
    "cause_principale": "Gestion de trafic",
    "retard_moyen_trains_retard_sup15": 23,
    "commentaire_principale": "\n\n",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 28.3,
    "prct_cause_exploit": 23.0,
    "dur_moyen": 243},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "9f298f74b9bdbaf0a32a54eb60cdb0d2",
   "fields": {"service": "International",
    "nb_train_prevu": 64,
    "nb_annulation": 0,
    "prct_cause_infra": 4.0,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 6,
    "gare_arrivee": "BRUSSELS MIDI",
    "nb_train_retard_sup_15": 0,
    "nb_train_retard_depart": 4,
    "cote": "NORD",
    "prct_cause_externe": 31.5,
    "nb_train_retard_arrivee": 6,
    "prct_cause_reseau": 11.8,
    "cause_principale": "External causes",
    "retard_moyen_trains_retard_sup15": 0,
    "commentaire_principale": "",
    "gare_depart": "PARIS NORD",
    "date": "2024-03-01",
    "prct_cause_gestion": 30.6,
    "prct_cause_exploit": 22.1,
    "dur_moyen": 82},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "b0b0a314c500b56693e082a418a9fdf6",
   "fields": {"service": "National",
    "nb_train_prevu": 150,
    "nb_annulation": 1,
    "prct_cause_infra": 10.0,
    "commentaire_annulation": "Mouvements sociaux",
    "retard_moyen_arrivee": 20,
    "gare_arrivee": "LYON PART DIEU",
    "nb_train_retard_sup_15": 12,
    "nb_train_retard_depart": 20,
    "cote": "SUD EST",
    "prct_cause_externe": 15.3,
    "nb_train_retard_arrivee": 40,
    "prct_cause_reseau": 12.0,
    "cause_principale": "External causes",
    "retard_moyen_trains_retard_sup15": 25,
    "commentaire_principale": "\n\n",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 30.0,
    "prct_cause_exploit": 32.7,
    "dur_moyen": 118},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "661dc53b0f0ce4456f3f7a692ca87c09",
   "fields": {"service": "National",
    "nb_train_prevu": 181,
    "nb_annulation": 0,
    "prct_cause_infra": 9.9,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 13,
    "gare_arrivee": "POITIERS",
    "nb_train_retard_sup_15": 5,
    "nb_train_retard_depart": 20,
    "cote": "SUD OUEST",
    "prct_cause_externe": 15.4,
    "nb_train_retard_arrivee": 33,
    "prct_cause_reseau": 13.2,
    "cause_principale": "Gestion de trafic",
    "retard_moyen_trains_retard_sup15": 18,
    "commentaire_principale": "",
    "gare_depart": "PARIS MONTPARNASSE",
    "date": "2024-03-01",
    "prct_cause_gestion": 41.8,
    "prct_cause_exploit": 19.7,
    "dur_moyen": 85},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "6f9841db1f9f8fe236c61b818b6d56c2",
   "fields": {"service": "National",
    "nb_train_prevu": 53,
    "nb_annulation": 0,
    "prct_cause_infra": 6.6,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 7,
    "gare_arrivee": "AVIGNON TGV",
    "nb_train_retard_sup_15": 1,
    "nb_train_retard_depart": 6,
    "cote": "SUD EST",
    "prct_cause_externe": 19.6,
    "nb_train_retard_arrivee": 8,
    "prct_cause_reseau": 17.2,
    "cause_principale": "Gestion de trafic",
    "retard_moyen_trains_retard_sup15": 16,
    "commentaire_principale": "",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 28.0,
    "prct_cause_exploit": 28.6,
    "dur_moyen": 160},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "95ab506d9480d14f141817fcdadfc424",
   "fields": {"service": "National",
    "nb_train_prevu": 45,
    "nb_annulation": 0,
    "prct_cause_infra": 20.2,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 17,
    "gare_arrivee": "PERPIGNAN",
    "nb_train_retard_sup_15": 5,
    "nb_train_retard_depart": 10,
    "cote": "SUD EST",
    "prct_cause_externe": 11.7,
    "nb_train_retard_arrivee": 12,
    "prct_cause_reseau": 17.8,
    "cause_principale": "Infrastructure",
    "retard_moyen_trains_retard_sup15": 22,
    "commentaire_principale": "",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 24.9,
    "prct_cause_exploit": 25.4,
    "dur_moyen": 255},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "7e4ae5df63543264bb9a235a3ec8120f",
   "fields": {"service": "National",
    "nb_train_prevu": 40,
    "nb_annulation": 1,
    "prct_cause_infra": 22.3,
    "commentaire_annulation": "Mouvements sociaux",
    "retard_moyen_arrivee": 21,
    "gare_arrivee": "BRIVE LA GAILLARDE",
    "nb_train_retard_sup_15": 7,
    "nb_train_retard_depart": 9,
    "cote": "SUD OUEST",
    "prct_cause_externe": 13.4,
    "nb_train_retard_arrivee": 14,
    "prct_cause_reseau": 22.4,
    "cause_principale": "Infrastructure",
    "retard_moyen_trains_retard_sup15": 23,
    "commentaire_principale": "",
    "gare_depart": "PARIS AUSTERLITZ",
    "date": "2024-03-01",
    "prct_cause_gestion": 25.0,
    "prct_cause_exploit": 25.0,
    "dur_moyen": 215},
   "record_timestamp": "2024-05-10T03:27:39.312Z"},
  {"datasetid": "regularite-mensuelle-tgv",
   "recordid": "fa0daf912b1db59ffcfed9675231c5ea",
   "fields": {"service": "National",
    "nb_train_prevu": 37,
    "nb_annulation": 0,
    "prct_cause_infra": 13.5,
    "commentaire_annulation": "",
    "retard_moyen_arrivee": 19,
    "gare_arrivee": "NICE VILLE",
    "nb_train_retard_sup_15": 5,
    "nb_train_retard_depart": 6,
    "cote": "SUD EST",
    "prct_cause_externe": 23.0,
    "nb_train_retard_arrivee": 10,
    "prct_cause_reseau": 15.9,
    "cause_principale": "External causes",
    "retard_moyen_trains_retard_sup15": 21,
    "commentaire_principale": "",
    "gare_depart": "PARIS GARE DE LYON",
    "date": "2024-03-01",
    "prct_cause_gestion": 25.9,
    "prct_cause_exploit": 21.7,
    "dur_moyen": 216},
   "record_timestamp": "2024-05-10T03:27:39.312Z"}
 ]}
"""

# Sample pricing JSON for Intercités fares (first 10 records)
PRICING_JSON = r"""
{"nhits": 301,
 "parameters": {"dataset": "tarifs-intercites", "rows": 10, "start": 0, "timezone": "UTC"},
 "records": [
  {"datasetid": "tarifs-intercites",
   "recordid": "188bc8f442cba7fa4e10355f8d564979",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "BERCY BOURGOGNE - PAYS D'AUVERGNE",
    "origine_uic": 8775800,
    "destination": "TOULOUSE MATABIAU",
    "destination_uic": 8761100,
    "profil_tarifaire": "STANDARD",
    "prix_min": 15.0,
    "prix_max": 85.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "3750b2e7af631cd253c867a8f051bcce",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "QUIMPER",
    "origine_uic": 8747100,
    "destination": "PARIS MONTPARNASSE 1 ET 2",
    "destination_uic": 8739100,
    "profil_tarifaire": "STANDARD",
    "prix_min": 19.0,
    "prix_max": 89.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "d0baf48747fc098d933cfbf23ebff68f",
   "fields": {"transporteur": "INTERCITES",
    "classe": "1re classe",
    "origine": "PARIS GARE DE LYON",
    "origine_uic": 8738240,
    "destination": "CLERMONT-FERRAND",
    "destination_uic": 8761300,
    "profil_tarifaire": "STANDARD",
    "prix_min": 25.0,
    "prix_max": 100.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "5a0672b8f7ec20c27db94038fbe8c84b",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "PARIS AUSTERLITZ",
    "origine_uic": 8768600,
    "destination": "TOURS",
    "destination_uic": 8757100,
    "profil_tarifaire": "STANDARD",
    "prix_min": 18.0,
    "prix_max": 80.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "71332c71dcdb8e4fc964d437bcc0c053",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "NANTES",
    "origine_uic": 8748100,
    "destination": "LYON PART DIEU",
    "destination_uic": 8772200,
    "profil_tarifaire": "STANDARD",
    "prix_min": 25.0,
    "prix_max": 120.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "4795338e0b71dbe82d496d065244e8c1",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "MARSEILLE ST CHARLES",
    "origine_uic": 8772205,
    "destination": "BORDEAUX ST JEAN",
    "destination_uic": 8758100,
    "profil_tarifaire": "STANDARD",
    "prix_min": 30.0,
    "prix_max": 120.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "6e01db474538b392e28712c8e753c173",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "PARIS GARE D'AUSTERLITZ",
    "origine_uic": 8768600,
    "destination": "LIMOGES BENEVENT",
    "destination_uic": 8759900,
    "profil_tarifaire": "STANDARD",
    "prix_min": 15.0,
    "prix_max": 65.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "83ea9e6baf8250471b223d71b7345cb8",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "PARIS AUSTERLITZ",
    "origine_uic": 8768600,
    "destination": "POITIERS",
    "destination_uic": 8757500,
    "profil_tarifaire": "STANDARD",
    "prix_min": 14.0,
    "prix_max": 70.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "8f943d3cdd1c5f28d1aa848f48b658b3",
   "fields": {"transporteur": "INTERCITES",
    "classe": "2e classe",
    "origine": "PARIS GARE DE LYON",
    "origine_uic": 8738240,
    "destination": "LYON PART DIEU",
    "destination_uic": 8772200,
    "profil_tarifaire": "STANDARD",
    "prix_min": 22.0,
    "prix_max": 100.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"},
  {"datasetid": "tarifs-intercites",
   "recordid": "95bc3c483ea8a5e97227008499ef38f7",
   "fields": {"transporteur": "INTERCITES",
    "classe": "1re classe",
    "origine": "PARIS GARE DE LYON",
    "origine_uic": 8738240,
    "destination": "LYON PART DIEU",
    "destination_uic": 8772200,
    "profil_tarifaire": "STANDARD",
    "prix_min": 45.0,
    "prix_max": 150.0},
   "record_timestamp": "2024-03-26T10:16:45.762Z"}
 ]}
"""

# Sample stations JSON providing coordinates (first 10 records)
STATIONS_JSON = r"""
{"nhits": 3884,
 "parameters": {"dataset": "liste-des-gares", "rows": 10, "start": 0, "timezone": "UTC"},
 "records": [
  {"datasetid": "liste-des-gares",
   "recordid": "b183b6af6680498b40171fc1b9f9093a34706b",
   "fields": {"code_uic": "87382381",
    "libelle": "La Défense ",
    "geo_point_2d": [48.89343732736893, 2.2384716843545894]},
   "record_timestamp": "2024-03-28T14:53:53.526Z"},
  {"datasetid": "liste-des-gares",
   "recordid": "82615392e5077d374a438ba62c84045b",
   "fields": {"code_uic": "87688889",
    "libelle": "Aubiet",
    "geo_point_2d": [43.16042316540546, 0.8879757448941251]},
   "record_timestamp": "2024-03-28T14:53:53.526Z"},
  {"datasetid": "liste-des-gares",
   "recordid": "6984acb3bfb3b2ed9df1c7a42457ccb7",
   "fields": {"code_uic": "87571564",
    "libelle": "Marmagne",
    "geo_point_2d": [46.91350318595993, 2.382545463063257]},
   "record_timestamp": "2024-03-28T14:53:53.526Z"},
  {"datasetid": "liste-des-gares",
   "recordid": "8ac7f0af6b6e05c8fa1fce3045b50ae3",
   "fields": {"code_uic": "87798688",
    "libelle": "Montpellier Sud-de-France",
    "geo_point_2d": [43.595093596781462, 3.9135030318595994]},
   "record_timestamp": "2024-03-28T14:53:53.526Z"},
  {"datasetid": "liste-des-gares",
   "recordid": "f58e6c077bd298345a962ff3b5a0601e",
   "fields": {"code_uic": "87571182",
    "libelle": "Poitiers",
    "geo_point_2d": [46.58356285073072, 0.344242961118062]},
   "record_timestamp": "2024-03-28T14:53:53.526Z"},
  {"datasetid": "liste-des-gares",
   "recordid": "f33cea19985db58386f0c1cbf16e7e2d",
   "fields": {"code_uic": "87471003",
    "libelle": "Quimper",
    "geo_point_2d": [47.993723802589894, -4.094451602672277]},
   "record_timestamp": "2024-03-28T14:53:53.526Z"},
  {"datasetid": "liste-des-gares",
   "recordid": "fbbd196938531884e2b1fbde95bed2fa",
   "fields": {"code_uic": "8738240",
    "libelle": "Paris Gare de Lyon",
    "geo_point_2d": [48.844845603079, 2.374625723855095]},
   "record_timestamp": "2024-03-28T14:53:53.526Z"},
  {"datasetid": "liste-des-gares",
   "recordid": "c7f3c8b34577c7bb30fb2bdddc0d08a2",
   "fields": {"code_uic": "87382782",
    "libelle": "Les Coquetiers",
    "geo_point_2d": [48.91080298544214, 2.510739390533775]},
   "record_timestamp": "2024-03-28T14:53:53.526Z"},
  {"datasetid": "liste-des-gares",
   "recordid": "f5ee0dceae1b4dc0c7f7ff83d530459c",
   "fields": {"code_uic": "8748100",
    "libelle": "Nantes",
    "geo_point_2d": [47.2166243247275, -1.5515494322564984]},
   "record_timestamp": "2024-03-28T14:53:53.526Z"},
  {"datasetid": "liste-des-gares",
   "recordid": "b2a0a3a4b58e7738c35b4ee7cf0ca3c8",
   "fields": {"code_uic": "8758100",
    "libelle": "Bordeaux St Jean",
    "geo_point_2d": [44.828563181114952, -0.5566119806464477]},
   "record_timestamp": "2024-03-28T14:53:53.526Z"}
 ]}
"""


def load_json_records(json_string: str) -> pd.DataFrame:
    """Load a JSON string of records and return a DataFrame of the 'fields' dict."""
    data = json.loads(json_string)
    records = [rec["fields"] for rec in data["records"]]
    return pd.DataFrame(records)


def load_ridership_csv(path: Optional[str]) -> pd.DataFrame:
    """Load the ridership dataset robustly.

    Accepts an optional ``path``. If ``None`` or not found, tries common local
    locations, then attempts to read from ``RIDERSHIP_URL``. As a last resort,
    uses a tiny in-memory sample so that the app remains functional.
    """
    # Build candidate local paths
    candidate_paths = []
    if path:
        candidate_paths.append(Path(path))
    candidate_paths.append(DATA_DIR / "frequentation-gares.csv")
    candidate_paths.append(APP_DIR / "frequentation-gares.csv")
    candidate_paths.append(Path(os.getcwd()) / "frequentation-gares.csv")

    df: pd.DataFrame | None = None

    # Try local files first
    for candidate in candidate_paths:
        try:
            if candidate and candidate.exists():
                df = pd.read_csv(candidate, sep=";", dtype=str)
                break
        except Exception:
            # Try next candidate
            pass

    # Try remote URL if no local file worked
    if df is None:
        try:
            df = pd.read_csv(RIDERSHIP_URL, sep=";", dtype=str)
            # Best effort: cache a local copy for next runs
            try:
                ensure_dir(DATA_DIR)
                df.to_csv(DATA_DIR / "frequentation-gares.csv", index=False, sep=";")
            except Exception:
                pass
        except Exception:
            df = None

    # Final fallback: minimal embedded sample
    if df is None:
        sample_csv = (
            "nom_gare;total_voyageurs_2022;total_voyageurs_2023;total_voyageurs_2024;non_voyageurs\n"
            "Paris Gare de Lyon;100000000;105000000;110000000;0\n"
            "Lyon Part-Dieu;50000000;52000000;54000000;0\n"
            "Bordeaux St Jean;30000000;31000000;32000000;0\n"
        )
        df = pd.read_csv(StringIO(sample_csv), sep=";", dtype=str)

    # Convert numeric columns to appropriate dtypes
    for col in df.columns:
        if col.startswith("total_voyageurs"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col == "non_voyageurs":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Compute the great circle distance between two (lat, lon) points in km."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # Earth radius in kilometres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def analyse_ridership(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ridership statistics: top stations and overall trends."""
    # Identify year columns
    year_cols = [col for col in df.columns if col.startswith("total_voyageurs_") and col[-4:].isdigit()]
    # Compute total passengers each year across all stations
    totals = df[year_cols].sum()
    totals.index = totals.index.str.extract(r"(\d{4})")[0]
    totals = totals.sort_index()
    # Compute top 10 stations for 2024 (most recent year)
    if "total_voyageurs_2024" in df.columns:
        top2024 = df[["nom_gare", "total_voyageurs_2024"]].copy()
        top2024["total_voyageurs_2024"] = pd.to_numeric(top2024["total_voyageurs_2024"], errors="coerce")
        top10 = top2024.nlargest(10, "total_voyageurs_2024")
    else:
        # fallback: use latest available year
        last_year = sorted([int(c[-4:]) for c in year_cols])[-1]
        col = f"total_voyageurs_{last_year}"
        top2024 = df[["nom_gare", col]].copy()
        top2024[col] = pd.to_numeric(top2024[col], errors="coerce")
        top10 = top2024.nlargest(10, col).rename(columns={col: "total_voyageurs"})
    return totals.to_frame(name="total_passengers"), top10


def analyse_punctuality(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics on punctuality sample."""
    # Convert relevant numeric columns
    numeric_cols = [
        "nb_train_prevu", "nb_annulation", "retard_moyen_arrivee",
        "nb_train_retard_sup_15", "nb_train_retard_depart",
        "nb_train_retard_arrivee", "retard_moyen_trains_retard_sup15",
        "prct_cause_infra", "prct_cause_externe", "prct_cause_reseau",
        "prct_cause_gestion", "prct_cause_exploit",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Compute average delays and cancellation rate
    summary = {}
    summary["avg_delay_arrival"] = df["retard_moyen_arrivee"].mean()
    summary["avg_delay_trains_>15"] = df["retard_moyen_trains_retard_sup15"].mean()
    summary["cancellation_rate"] = df["nb_annulation"].sum() / df["nb_train_prevu"].sum()
    # Compute average cause percentages
    cause_cols = [c for c in df.columns if c.startswith("prct_cause_")]
    cause_means = df[cause_cols].mean()
    return pd.DataFrame(summary, index=[0]), cause_means


def analyse_pricing(df_price: pd.DataFrame, df_stations: pd.DataFrame) -> pd.DataFrame:
    """Estimate cost per kilometre for each price record using station coordinates."""
    # Create mapping from UIC code to coordinates from station dataset
    coord_map = {}
    for _, row in df_stations.dropna(subset=["code_uic", "geo_point_2d"]).iterrows():
        coord_map[str(row["code_uic"])] = tuple(row["geo_point_2d"])
    # Compute distance and cost per km using average price
    distances = []
    cost_per_km_min = []
    cost_per_km_max = []
    for _, row in df_price.iterrows():
        orig = str(row["origine_uic"])
        dest = str(row["destination_uic"])
        coord1 = coord_map.get(orig)
        coord2 = coord_map.get(dest)
        if coord1 and coord2:
            dist = haversine(coord1, coord2)
        else:
            dist = float("nan")
        distances.append(dist)
        # Compute cost per km only if distance is finite
        avg_min = row["prix_min"] / dist if dist and not pd.isna(dist) and dist > 0 else float("nan")
        avg_max = row["prix_max"] / dist if dist and not pd.isna(dist) and dist > 0 else float("nan")
        cost_per_km_min.append(avg_min)
        cost_per_km_max.append(avg_max)
    df_price = df_price.copy()
    df_price["distance_km"] = distances
    df_price["cost_per_km_min"] = cost_per_km_min
    df_price["cost_per_km_max"] = cost_per_km_max
    return df_price


def estimate_emissions(distance_km: float) -> Tuple[float, float]:
    """Estimate CO₂ emissions (kg) for train and car based on distance and factors."""
    # Emission factors (g CO2 per passenger-km) from Youmatter article【679178350990318†L62-L68】
    train_factor = 14  # g per passenger-km
    car_factor = 55   # g per passenger-km for an average car
    train_emissions = distance_km * train_factor / 1000  # convert g to kg
    car_emissions = distance_km * car_factor / 1000
    return train_emissions, car_emissions


def plot_ridership_trend(totals: pd.DataFrame, outdir: str) -> str:
    """Plot total ridership trend over years."""
    fig, ax = plt.subplots(figsize=(8, 4))
    totals.plot(kind="line", ax=ax, marker="o")
    ax.set_title("Évolution annuelle du nombre de voyageurs")
    ax.set_xlabel("Année")
    ax.set_ylabel("Voyageurs (millions)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    filename = os.path.join(outdir, "ridership_trend.png")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename


def plot_top_stations(top10: pd.DataFrame, outdir: str) -> str:
    """Plot bar chart of top 10 stations by ridership."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="total_voyageurs_2024", y="nom_gare", data=top10, palette="viridis", ax=ax)
    ax.set_title("Top 10 des gares en 2024 par fréquentation")
    ax.set_xlabel("Voyageurs")
    ax.set_ylabel("Gare")
    filename = os.path.join(outdir, "top_stations.png")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename


def plot_punctuality_causes(cause_means: pd.Series, outdir: str) -> str:
    """Plot distribution of delay causes in sample."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cause_means.sort_values().plot(kind="barh", ax=ax, color="salmon")
    ax.set_title("Répartition moyenne des causes de retard (échantillon TGV)")
    ax.set_xlabel("Pourcentage (%)")
    filename = os.path.join(outdir, "punctuality_causes.png")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename


def plot_pricing_distribution(df_price: pd.DataFrame, outdir: str) -> str:
    """Plot minimum and maximum prices by route."""
    fig, ax = plt.subplots(figsize=(9, 5))
    # Create label for each route
    df_price = df_price.copy()
    df_price["route"] = df_price["origine"] + " → " + df_price["destination"]
    df_price.set_index("route", inplace=True)
    df_price[["prix_min", "prix_max"]].plot(kind="bar", ax=ax)
    ax.set_title("Tarifs Intercités - prix min et max (échantillon)")
    ax.set_xlabel("Liaison")
    ax.set_ylabel("Prix (€)")
    plt.xticks(rotation=45, ha="right")
    filename = os.path.join(outdir, "pricing_distribution.png")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename


def plot_emissions_example(df_price: pd.DataFrame, outdir: str) -> str:
    """Plot emissions comparison for sample routes based on distance."""
    # Use only rows with computed distances
    df = df_price.dropna(subset=["distance_km"]).copy()
    # If no valid distance exists, skip plotting and return empty string
    if df.empty:
        return ""
    emissions = df["distance_km"].apply(lambda d: estimate_emissions(d))
    df["train_emissions"], df["car_emissions"] = zip(*emissions)
    df["route"] = df["origine"] + " → " + df["destination"]
    fig, ax = plt.subplots(figsize=(9, 5))
    df.set_index("route")[["train_emissions", "car_emissions"]].plot(kind="bar", ax=ax)
    ax.set_title("Émissions CO₂ estimées (kg par passager) par liaison")
    ax.set_xlabel("Liaison")
    ax.set_ylabel("Émissions (kg CO₂)")
    plt.xticks(rotation=45, ha="right")
    filename = os.path.join(outdir, "emissions_comparison.png")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename


def plot_emissions_generic(outdir: str) -> str:
    """Create a general emissions comparison chart for typical distances.

    When no reliable station coordinates are available to compute route
    distances, we illustrate the emissions difference between rail and car
    for a range of representative distances (100–700 km).  Emission
    estimates are derived from the Youmatter factors used in
    ``estimate_emissions``【679178350990318†L62-L68】.
    """
    distances = [100, 300, 500, 700]  # km
    emissions_data = [estimate_emissions(d) for d in distances]
    train_emissions = [e[0] for e in emissions_data]
    car_emissions = [e[1] for e in emissions_data]
    df = pd.DataFrame({
        "distance_km": distances,
        "train_emissions": train_emissions,
        "car_emissions": car_emissions,
    })
    fig, ax = plt.subplots(figsize=(7, 4))
    df.plot(x="distance_km", y=["train_emissions", "car_emissions"], kind="bar", ax=ax)
    ax.set_title("Émissions CO₂ estimées en fonction de la distance")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Émissions (kg CO₂ par passager)")
    ax.legend(["Train", "Voiture"])
    filename = os.path.join(outdir, "emissions_generic.png")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename


def main() -> None:
    # Ensure output directory exists
    fig_dir = os.path.join(os.getcwd(), "figures")
    ensure_dir(fig_dir)
    # Load ridership dataset
    ridership_path = os.path.join(os.getcwd(), "frequentation-gares.csv")
    df_ridership = load_ridership_csv(ridership_path)
    # Analyse ridership
    totals, top10 = analyse_ridership(df_ridership)
    # Load punctuality sample
    df_punctuality = load_json_records(PUNCTUALITY_JSON)
    punctuality_summary, cause_means = analyse_punctuality(df_punctuality)
    # Load pricing sample
    df_price = load_json_records(PRICING_JSON)
    # Convert numeric fields to floats
    df_price["prix_min"] = pd.to_numeric(df_price["prix_min"], errors="coerce")
    df_price["prix_max"] = pd.to_numeric(df_price["prix_max"], errors="coerce")
    df_price["origine_uic"] = pd.to_numeric(df_price["origine_uic"], errors="coerce").astype("Int64")
    df_price["destination_uic"] = pd.to_numeric(df_price["destination_uic"], errors="coerce").astype("Int64")
    # Load station list sample
    df_stations = load_json_records(STATIONS_JSON)
    # Analyse pricing (cost per km)
    df_price = analyse_pricing(df_price, df_stations)
    # Print summary tables
    print("Punctuality summary (sample):")
    print(punctuality_summary)
    print("\nAverage delay causes (%):")
    print(cause_means)
    print("\nTop 10 stations by passengers (2024):")
    print(top10)
    # Save plots
    plot_ridership_trend(totals, fig_dir)
    plot_top_stations(top10, fig_dir)
    plot_punctuality_causes(cause_means, fig_dir)
    plot_pricing_distribution(df_price, fig_dir)
    # Plot emissions using generic distances as no coordinates match the sample
    plot_emissions_generic(fig_dir)


if __name__ == "__main__":
    main()
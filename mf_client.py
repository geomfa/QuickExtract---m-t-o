"""
Client pour les données climatologiques de Météo-France.
Source : data.gouv.fr (open data, aucun compte requis).

Fichiers horaires  : BASE/HOR/H_{DEPT}_latest-{a1}-{a2}.csv.gz   — fraîcheur J-1/J-2
Fichiers quotidiens: BASE/QUO/Q_{DEPT}_latest-{a1}-{a2}.csv.gz   — idem, moins volumineux
Fichiers décennaux : BASE/QUO/Q_{DEPT}_{a1}-{a2}.csv.gz          — historique

Format commun : séparateur ';', encodage latin-1.
Colonnes communes : NUM_POSTE, NOM_USUEL, LAT, LON, ALTI.
Date horaire  : AAAAMMJJHH  (%Y%m%d%H, heure UTC)
Date quotidien: AAAAMMJJ    (%Y%m%d)
"""

import io
import gzip
import math
import datetime
import requests
import pandas as pd
import streamlit as st

BASE_URL_HOR = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR"
BASE_URL_QUO = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/QUO"
BASE_URL_MEN = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/MEN"
GEO_API      = "https://geo.api.gouv.fr"

ANNEE_NORMALE_DEBUT = 1991
ANNEE_NORMALE_FIN   = 2020

COLONNES_METEO = {
    "t_celsius": ["T"],
    "u_pct":     ["U"],
    "ff_ms":     ["FF"],
    "dd_deg":    ["DD"],
    "fx_ms":     ["FXY", "FX", "FXXY"],
    "rr1_mm":    ["RR1"],
    "rr_mm":     ["RR"],   # quotidien
    "pmer_hpa":  ["PMER"],
    "n_octas":   ["N"],
    "vv_m":      ["VV"],
}


class DonneesClimatoError(Exception):
    pass


# ==============================================================================
# UTILITAIRES
# ==============================================================================

def _normaliser_dept(code):
    code = str(code).strip().upper()
    return ("0" + code) if len(code) == 1 else code


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _telecharger_gz(url):
    """Télécharge et décompresse un CSV.gz. Retourne DataFrame ou None."""
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        raw = gzip.decompress(resp.content)
        return pd.read_csv(
            io.BytesIO(raw), sep=";", encoding="latin-1",
            low_memory=False, dtype={"NUM_POSTE": str},
        )
    except Exception:
        return None


def _candidats_hor(dept):
    a = datetime.datetime.utcnow().year
    return [
        f"H_{dept}_latest-{a-1}-{a}.csv.gz",
        f"H_{dept}_latest-{a}-{a+1}.csv.gz",
        f"H_{dept}_latest-{a-2}-{a-1}.csv.gz",
    ]


def _candidats_quo_recents(dept):
    a = datetime.datetime.utcnow().year
    return [
        f"Q_{dept}_latest-{a-1}-{a}.csv.gz",
        f"Q_{dept}_latest-{a}-{a+1}.csv.gz",
        f"Q_{dept}_latest-{a-2}-{a-1}.csv.gz",
    ]


def _candidats_men_decennaux(dept):
    """
    Fichiers MENSUELS Météo-France (BASE/MEN) couvrant 1991-2020.

    Structure réelle des fichiers mensuels (source : doc Météo-France / meteo.data.gouv) :
      M_{dept}_{debut}-1949.csv.gz     — historique ancien
      M_{dept}_previous-1950-2022.csv.gz — 1950 à ~2022, contient 1991-2020
      M_{dept}_latest-2023-2024.csv.gz   — 2 dernières années

    On cible en priorité le fichier "previous" qui couvre 1991-2020.
    Le nom exact de la borne haute varie selon les mises à jour annuelles,
    on teste plusieurs variantes.
    """
    a = datetime.datetime.utcnow().year
    candidats = []
    # previous : borne haute variable selon l'année de mise à jour
    for fin in range(a - 1, a - 6, -1):
        candidats.append(
            (f"M_{dept}_previous-1950-{fin}.csv.gz",
             f"{BASE_URL_MEN}/M_{dept}_previous-1950-{fin}.csv.gz")
        )
    # Variante avec borne basse différente (certains depts commencent après 1950)
    for debut in range(1950, 1960):
        for fin in range(a - 1, a - 4, -1):
            candidats.append(
                (f"M_{dept}_previous-{debut}-{fin}.csv.gz",
                 f"{BASE_URL_MEN}/M_{dept}_previous-{debut}-{fin}.csv.gz")
            )
    return candidats[:12]  # limiter les tentatives


# ==============================================================================
# GEOCODAGE
# ==============================================================================

@st.cache_data(ttl=86400, show_spinner=False)
def commune_depuis_insee(code_insee):
    code = str(code_insee).strip().zfill(5)
    try:
        resp = requests.get(
            f"{GEO_API}/communes/{code}",
            params={"fields": "nom,centre,codeDepartement", "format": "json"},
            timeout=10,
        )
        if resp.status_code == 404:
            raise DonneesClimatoError(f"Code INSEE '{code}' introuvable.")
        resp.raise_for_status()
        data = resp.json()
        return data["nom"], data["centre"]["coordinates"][1], \
               data["centre"]["coordinates"][0], data["codeDepartement"]
    except DonneesClimatoError:
        raise
    except Exception as e:
        raise DonneesClimatoError(f"Erreur API geo.api.gouv.fr : {e}")


# ==============================================================================
# TELECHARGEMENT HORAIRE
# ==============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def telecharger_horaire_departement(code_dept):
    dept = _normaliser_dept(code_dept)
    for nom in _candidats_hor(dept):
        df = _telecharger_gz(f"{BASE_URL_HOR}/{nom}")
        if df is not None:
            df["_fichier_source"] = nom
            return df
    raise DonneesClimatoError(
        f"Aucun fichier horaire trouvé pour le département '{dept}'."
    )


# ==============================================================================
# TELECHARGEMENT QUOTIDIEN (normales décennales)
# ==============================================================================

@st.cache_data(ttl=86400, show_spinner=False)
def telecharger_mensuel_decennal(code_dept):
    """
    Télécharge les fichiers MENSUELS couvrant 1991-2020 et les concatène.
    Les fichiers mensuels (BASE/MEN) contiennent directement les agrégats
    T°min/moy/max et précipitations par mois — aucune agrégation nécessaire.
    Retourne DataFrame ou None si indisponible.
    """
    dept = _normaliser_dept(code_dept)
    dfs = []
    for nom, url in _candidats_men_decennaux(dept):
        df = _telecharger_gz(url)
        if df is not None:
            df["_fichier_source"] = nom
            dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


# ==============================================================================
# INSPECTION DES STATIONS (étape intermédiaire)
# ==============================================================================

def inspecter_stations(df_brut, lat, lon, n=10):
    """
    Retourne un DataFrame décrivant les n stations les plus proches avec :
      - NUM_POSTE, NOM_USUEL, LAT, LON, ALTI, distance_km
      - derniere_date     : dernière date disponible dans le fichier
      - fraicheur_jours   : nombre de jours depuis la dernière mesure
      - nebulo_dispo      : True si colonne N présente et non vide
    """
    cols_base = [c for c in ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI"]
                 if c in df_brut.columns]
    df_sta = (
        df_brut[cols_base]
        .drop_duplicates(subset=["NUM_POSTE"])
        .reset_index(drop=True)
        .copy()
    )

    if "LAT" not in df_sta.columns:
        return df_sta

    df_sta["distance_km"] = df_sta.apply(
        lambda r: round(_haversine_km(lat, lon, float(r["LAT"]), float(r["LON"])), 1),
        axis=1,
    )
    df_sta = df_sta.sort_values("distance_km").head(n).reset_index(drop=True)

    # Détection colonne date
    col_date = next((c for c in df_brut.columns if c.upper().startswith("AAAA")), None)
    col_n    = "N" if "N" in df_brut.columns else None

    fraicheurs    = []
    dernieres_dt  = []
    nebulos       = []

    now = datetime.datetime.utcnow()

    for _, row in df_sta.iterrows():
        sid    = str(row["NUM_POSTE"])
        subset = df_brut[df_brut["NUM_POSTE"].astype(str) == sid]

        # Dernière date
        if col_date and not subset.empty:
            fmt = "%Y%m%d%H" if "HH" in col_date.upper() else "%Y%m%d"
            dts = pd.to_datetime(
                subset[col_date].astype(str).str.strip(),
                format=fmt, errors="coerce",
            ).dropna()
            if not dts.empty:
                derniere = dts.max().to_pydatetime()
                dernieres_dt.append(derniere.strftime("%d/%m/%Y %Hh"))
                fraicheurs.append((now - derniere).days)
            else:
                dernieres_dt.append("n/d")
                fraicheurs.append(None)
        else:
            dernieres_dt.append("n/d")
            fraicheurs.append(None)

        # Nébulosité
        if col_n:
            vals = pd.to_numeric(subset[col_n], errors="coerce").dropna()
            nebulos.append(len(vals) > 0)
        else:
            nebulos.append(False)

    df_sta["derniere_date"]   = dernieres_dt
    df_sta["fraicheur_jours"] = fraicheurs
    df_sta["nebulo_dispo"]    = nebulos

    return df_sta


def inspecter_decennales(df_sta, df_decennal):
    """
    Ajoute une colonne 'decennales_dispo' au DataFrame stations
    indiquant si des données 1991-2020 existent pour chaque station.
    """
    if df_decennal is None or df_decennal.empty:
        df_sta = df_sta.copy()
        df_sta["decennales_dispo"] = False
        return df_sta

    ids_dispo = set(df_decennal["NUM_POSTE"].astype(str).unique())
    df_sta = df_sta.copy()
    df_sta["decennales_dispo"] = df_sta["NUM_POSTE"].astype(str).isin(ids_dispo)
    return df_sta


# ==============================================================================
# REFERENTIEL STATIONS & FILTRAGE
# ==============================================================================

def stations_du_fichier(df_brut):
    cols = [c for c in ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI"]
            if c in df_brut.columns]
    if "NUM_POSTE" not in cols:
        return pd.DataFrame()
    return df_brut[cols].drop_duplicates(subset=["NUM_POSTE"]).reset_index(drop=True)


def stations_proches(df_brut, lat, lon, n=5):
    df = stations_du_fichier(df_brut)
    if df.empty or "LAT" not in df.columns:
        return df
    df = df.copy()
    df["distance_km"] = df.apply(
        lambda r: round(_haversine_km(lat, lon, float(r["LAT"]), float(r["LON"])), 1),
        axis=1,
    )
    return df.sort_values("distance_km").head(n).reset_index(drop=True)


def filtrer_periode(df_brut, date_debut, date_fin, col_date=None):
    if col_date is None:
        col_date = next(
            (c for c in df_brut.columns if c.upper().startswith("AAAA")), None
        )
    if col_date is None:
        raise DonneesClimatoError(
            f"Colonne date introuvable. Colonnes : {list(df_brut.columns)}"
        )

    fmt = "%Y%m%d%H" if len(str(df_brut[col_date].iloc[0])) >= 10 else "%Y%m%d"

    df = df_brut.copy()
    df["date_dt"] = pd.to_datetime(
        df[col_date].astype(str).str.strip(), format=fmt, errors="coerce"
    )
    df = df.dropna(subset=["date_dt"])

    d1 = pd.Timestamp(date_debut)
    d2 = pd.Timestamp(date_fin)
    if d1.tzinfo is not None:
        d1 = d1.tz_localize(None)
    if d2.tzinfo is not None:
        d2 = d2.tz_localize(None)

    return df[(df["date_dt"] >= d1) & (df["date_dt"] <= d2)].sort_values("date_dt")


# ==============================================================================
# NORMALISATION
# ==============================================================================

def normaliser_variables(df):
    df = df.copy()
    for nom_logique, candidats in COLONNES_METEO.items():
        for c in candidats:
            if c in df.columns:
                df[nom_logique] = pd.to_numeric(df[c], errors="coerce")
                break
    return df


def colonne_presente(df, nom_logique):
    return nom_logique in df.columns and df[nom_logique].notna().any()


# ==============================================================================
# AGREGATION MULTI-STATIONS
# ==============================================================================

def agreger_multi_stations(df_obs, df_stations):
    if df_obs.empty or df_stations.empty:
        return pd.DataFrame()

    poids_map = {
        str(row["NUM_POSTE"]): 1.0 / max(row["distance_km"], 0.1)
        for _, row in df_stations.iterrows()
    }
    df = df_obs.copy()
    df["NUM_POSTE"] = df["NUM_POSTE"].astype(str)
    df["_poids"]    = df["NUM_POSTE"].map(poids_map).fillna(0.0)

    cols_num = [c for c in COLONNES_METEO if c in df.columns]

    def _wavg(g):
        out = {}
        for col in cols_num:
            valide = g[[col, "_poids"]].dropna(subset=[col])
            out[col] = (
                (valide[col] * valide["_poids"]).sum() / valide["_poids"].sum()
                if not valide.empty else float("nan")
            )
        out["n_stations"] = g["NUM_POSTE"].nunique()
        return pd.Series(out)

    return df.groupby("date_dt").apply(_wavg).reset_index()


# ==============================================================================
# NORMALES DECENNALES 1991-2020
# ==============================================================================

def calculer_normales(df_mensuel, ids_stations):
    """
    Calcule les normales mensuelles 1991-2020 à partir des fichiers MENSUELS.

    Les fichiers mensuels Météo-France contiennent directement les valeurs
    agrégées par mois. Colonnes typiques :
      AAAAMMJJ (date au 1er du mois), NUM_POSTE, TN (T°min), TM (T°moy),
      TX (T°max), RR (précipitations mensuelles cumulées).

    Retourne DataFrame avec : mois (1-12), t_moy_norm, t_min_norm,
    t_max_norm, precip_norm. None si données insuffisantes.
    """
    if df_mensuel is None or df_mensuel.empty:
        return None

    df = df_mensuel[
        df_mensuel["NUM_POSTE"].astype(str).isin([str(i) for i in ids_stations])
    ].copy()

    if df.empty:
        return None

    # Colonnes du fichier mensuel (nomenclature Météo-France BASE/MEN)
    # TN=min, TM=moy, TX=max, RR=précip. Certains fichiers anciens utilisent
    # T_MIN, T_MOY, T_MAX — on tente les deux nomenclatures.
    rename = {}
    for src, dst in [
        ("TN", "t_min_src"), ("T_MIN", "t_min_src"), ("TNFD", "t_min_src"),
        ("TM", "t_moy_src"), ("T_MOY", "t_moy_src"),
        ("TX", "t_max_src"), ("T_MAX", "t_max_src"), ("TXFD", "t_max_src"),
        ("RR", "rr_src"),    ("RRAB", "rr_src"),
    ]:
        if src in df.columns and dst not in rename.values():
            rename[src] = dst
    df = df.rename(columns=rename)

    # Colonne date : AAAAMMJJ ou AAAAMM
    col_date = next(
        (c for c in df.columns if c.upper().startswith("AAAA")), None
    )
    if col_date is None:
        return None

    n_chars = df[col_date].astype(str).str.strip().str.len().mode().iloc[0]
    fmt = "%Y%m%d" if n_chars >= 8 else "%Y%m"

    df["date_dt"] = pd.to_datetime(
        df[col_date].astype(str).str.strip().str[:8 if fmt == "%Y%m%d" else 6],
        format=fmt, errors="coerce",
    )
    df = df.dropna(subset=["date_dt"])
    df = df[
        (df["date_dt"].dt.year >= ANNEE_NORMALE_DEBUT) &
        (df["date_dt"].dt.year <= ANNEE_NORMALE_FIN)
    ]

    if df.empty:
        return None

    df["mois"] = df["date_dt"].dt.month

    for col in ["t_min_src", "t_moy_src", "t_max_src", "rr_src"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg = {}
    if "t_moy_src" in df.columns and df["t_moy_src"].notna().any():
        agg["t_moy_norm"] = ("t_moy_src", "mean")
    if "t_min_src" in df.columns and df["t_min_src"].notna().any():
        agg["t_min_norm"] = ("t_min_src", "mean")
    if "t_max_src" in df.columns and df["t_max_src"].notna().any():
        agg["t_max_norm"] = ("t_max_src", "mean")
    if "rr_src" in df.columns and df["rr_src"].notna().any():
        agg["precip_norm"] = ("rr_src", "mean")

    if not agg:
        return None

    return df.groupby("mois").agg(**agg).reset_index()


# ==============================================================================
# DEBUG
# ==============================================================================

def debug_colonnes(df_brut):
    src = df_brut["_fichier_source"].iloc[0] if "_fichier_source" in df_brut.columns else "?"
    st.write(f"Fichier : `{src}` — {len(df_brut):,} lignes")
    st.write(f"Colonnes ({len(df_brut.columns)}) :", list(df_brut.columns))
    st.dataframe(df_brut.head(3), use_container_width=True)

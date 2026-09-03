"""
Client pour les données climatologiques de Météo-France.
Source : data.gouv.fr (open data, aucun compte requis).

Fichiers horaires  (BASE/HOR) :
  H_{DEPT}_latest-{a1}-{a2}.csv.gz          — fraîcheur J-1/J-2, ~1-2 ans
  Colonne date : AAAAMMJJHH (%Y%m%d%H)

Fichiers quotidiens (BASE/QUOT) — pour les normales sur ~10 ans :
  Q_{DEPT}_latest-{a1}-{a2}_RR-T-Vent.csv.gz   — 2 dernières années
  Q_{DEPT}_previous-1950-{a}_RR-T-Vent.csv.gz  — 1950 à ~2021, couvre 10 ans
  Q_{DEPT}_avant-1949_RR-T-Vent.csv.gz          — historique < 1950
  Colonne date : AAAAMMJJ (%Y%m%d)
  Variables : T (T°moy), TN (min), TX (max), RR (précip mm)

Format commun : séparateur ';', encodage latin-1.
Colonnes communes : NUM_POSTE, NOM_USUEL, LAT, LON, ALTI.
"""

import io
import gzip
import math
import datetime
import requests
import pandas as pd
import streamlit as st

BASE_URL_HOR       = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR"
BASE_URL_QUOT      = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/QUOT"
BASE_URL_DECADAGRO = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/DECADAGRO"
GEO_API      = "https://geo.api.gouv.fr"


COLONNES_METEO = {
    "t_celsius": ["TM", "T"],       # TM = T°moy quotidien, T = horaire
    "t_min":     ["TN"],            # T°min quotidien
    "t_max":     ["TX"],            # T°max quotidien
    "u_pct":     ["U"],
    "ff_ms":     ["FF"],
    "dd_deg":    ["DD"],
    "fx_ms":     ["FXY", "FX", "FXXY"],
    "rr1_mm":    ["RR1"],
    "rr_mm":     ["RR"],            # précip quotidienne cumulée
    "pmer_hpa":  ["PMER"],
    "n_octas":   ["N"],
    "vv_m":      ["VV"],
    # ETP officielle Penman-Monteith FAO-56 (fichier décadaire agro) — nom de
    # colonne non vérifié directement (fichier source inaccessible en test),
    # plusieurs candidats testés par ordre de probabilité.
    "etp_officielle_mm": ["ETP", "ETPMON", "ETPP", "ETPPM"],
}

# Colonnes à conserver lors du téléchargement, pour limiter l'empreinte
# mémoire (les fichiers Météo-France ont 40-60 colonnes, dont de nombreux
# indicateurs qualité QT/QN/... jamais utilisés par l'app).
COLONNES_BASE = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI"]
COLONNES_DATE = ["AAAAMMJJHH", "AAAAMMJJ"]
COLONNES_UTILES = (
    COLONNES_BASE + COLONNES_DATE
    + [c for cands in COLONNES_METEO.values() for c in cands]
)


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


def _telecharger_gz(url, colonnes_utiles=None):
    """
    Télécharge et décompresse un CSV.gz. Retourne DataFrame ou None.

    Si colonnes_utiles est fourni, ne charge QUE ces colonnes (celles
    présentes dans le fichier) — réduit fortement l'empreinte mémoire,
    critique sur les fichiers quotidiens qui couvrent plusieurs décennies.
    """
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        raw = gzip.decompress(resp.content)

        if colonnes_utiles is not None:
            # Lecture de l'en-tête seul pour déterminer les colonnes réellement présentes
            entete = pd.read_csv(
                io.BytesIO(raw), sep=";", encoding="latin-1", nrows=0
            )
            usecols = [c for c in colonnes_utiles if c in entete.columns]
            return pd.read_csv(
                io.BytesIO(raw), sep=";", encoding="latin-1",
                low_memory=False, dtype={"NUM_POSTE": str},
                usecols=usecols,
            )

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

@st.cache_data(ttl=1800, show_spinner=False, max_entries=3)
def telecharger_horaire_departement(code_dept):
    dept = _normaliser_dept(code_dept)
    for nom in _candidats_hor(dept):
        df = _telecharger_gz(f"{BASE_URL_HOR}/{nom}", colonnes_utiles=COLONNES_UTILES)
        if df is not None:
            df["_fichier_source"] = nom
            return df
    raise DonneesClimatoError(
        f"Aucun fichier horaire trouvé pour le département '{dept}'."
    )


# ==============================================================================
# TELECHARGEMENT QUOTIDIEN (normales sur ~10 ans)
# ==============================================================================

def _candidats_quot_previous(dept):
    """
    Candidats pour le fichier quotidien couvrant 1950-~2021 (RR-T-Vent).
    Noms réels confirmés par Météo-France :
      Q_{dept}_previous-1950-{annee}_RR-T-Vent.csv.gz
    La borne haute varie selon les mises à jour annuelles — on teste plusieurs.
    """
    a = datetime.datetime.utcnow().year
    candidats = []
    for fin in range(a - 1, a - 8, -1):
        candidats.append(
            f"Q_{dept}_previous-1950-{fin}_RR-T-Vent.csv.gz"
        )
    # Variante sans suffixe _RR-T-Vent (anciens fichiers)
    for fin in range(a - 1, a - 4, -1):
        candidats.append(f"Q_{dept}_previous-1950-{fin}.csv.gz")
    return candidats


@st.cache_data(ttl=86400, show_spinner=False, max_entries=3)
def telecharger_quotidien_previous(code_dept, n_annees=10):
    """
    Télécharge le fichier quotidien historique pour un département et ne
    conserve QUE les n dernières années (par défaut 10) + les colonnes
    utiles. Le fichier source couvre 1950-~2024 sur toutes les stations
    du département : sans ce filtrage, il peut représenter plusieurs
    millions de lignes en mémoire, largement suffisant pour faire sauter
    le process sur un environnement à mémoire limitée (Streamlit Cloud).

    Retourne DataFrame filtré (n dernières années) ou None si indisponible.
    """
    dept = _normaliser_dept(code_dept)
    for nom in _candidats_quot_previous(dept):
        df = _telecharger_gz(f"{BASE_URL_QUOT}/{nom}", colonnes_utiles=COLONNES_UTILES)
        if df is not None:
            col_date = next((c for c in df.columns if c.upper().startswith("AAAA")), None)
            if col_date is not None:
                annees = pd.to_numeric(
                    df[col_date].astype(str).str.slice(0, 4), errors="coerce"
                )
                annee_max = annees.max()
                if pd.notna(annee_max):
                    df = df[annees >= (annee_max - n_annees + 1)].copy()
            df["_fichier_source"] = nom
            return df
    return None


# ==============================================================================
# INSPECTION DES STATIONS (étape intermédiaire)
# ==============================================================================

def inspecter_stations(df_brut, df_quot, df_decadagro, lat, lon, n=10):
    """
    Retourne un DataFrame décrivant les n stations les plus proches avec :
      - NUM_POSTE, NOM_USUEL, LAT, LON, ALTI, distance_km
      - derniere_date     : dernière date disponible dans le fichier horaire
      - fraicheur_jours   : nb de jours depuis la dernière mesure
      - nebulo_dispo      : True si colonne N présente et non vide
      - historique_dispo  : True si la station est présente dans le fichier quotidien
      - etp_dispo         : True si des valeurs ETP officielles exploitables existent
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

    col_date = next((c for c in df_brut.columns if c.upper().startswith("AAAA")), None)
    col_n    = "N" if "N" in df_brut.columns else None

    fraicheurs   = []
    dernieres_dt = []
    nebulos      = []
    now          = datetime.datetime.utcnow()

    for _, row in df_sta.iterrows():
        sid    = str(row["NUM_POSTE"])
        subset = df_brut[df_brut["NUM_POSTE"].astype(str) == sid]

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

        if col_n:
            vals = pd.to_numeric(subset[col_n], errors="coerce").dropna()
            nebulos.append(len(vals) > 0)
        else:
            nebulos.append(False)

    df_sta["derniere_date"]   = dernieres_dt
    df_sta["fraicheur_jours"] = fraicheurs
    df_sta["nebulo_dispo"]    = nebulos

    # Historique disponible : station présente dans le fichier quotidien
    if df_quot is not None and not df_quot.empty and "NUM_POSTE" in df_quot.columns:
        ids_hist = set(df_quot["NUM_POSTE"].astype(str).unique())
        df_sta["historique_dispo"] = df_sta["NUM_POSTE"].astype(str).isin(ids_hist)
    else:
        df_sta["historique_dispo"] = False

    # ETP officielle disponible : station présente dans le fichier décadaire
    # agro ET disposant de valeurs ETP réellement exploitables (la colonne
    # peut exister dans le fichier mais être vide pour certaines stations).
    if df_decadagro is not None and not df_decadagro.empty and "NUM_POSTE" in df_decadagro.columns:
        df_dec_norm = normaliser_variables(df_decadagro)
        if "etp_officielle_mm" in df_dec_norm.columns:
            ids_etp = set(
                df_dec_norm.loc[df_dec_norm["etp_officielle_mm"].notna(), "NUM_POSTE"]
                .astype(str).unique()
            )
        else:
            ids_etp = set()
        df_sta["etp_dispo"] = df_sta["NUM_POSTE"].astype(str).isin(ids_etp)
    else:
        df_sta["etp_dispo"] = False

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

def calculer_normales(df_quot, ids_stations, n_annees=10):
    """
    Calcule les normales mensuelles sur les n dernières années disponibles
    à partir du fichier quotidien historique (Q_dept_previous-1950-XXXX).

    Variables utilisées : TM (T°moy), TN (T°min), TX (T°max), RR (précip mm).
    Agrégation : moyenne mensuelle des T°, cumul mensuel des précip, puis
    moyenne de ces valeurs mensuelles sur les n dernières années.

    Retourne DataFrame avec : mois (1-12), t_moy_norm, t_min_norm,
    t_max_norm, precip_norm. None si données insuffisantes.
    """
    if df_quot is None or df_quot.empty:
        return None

    df = df_quot[
        df_quot["NUM_POSTE"].astype(str).isin([str(i) for i in ids_stations])
    ].copy()
    if df.empty:
        return None

    df = normaliser_variables(df)

    col_date = next((c for c in df.columns if c.upper().startswith("AAAA")), None)
    if col_date is None:
        return None

    df["date_dt"] = pd.to_datetime(
        df[col_date].astype(str).str.strip().str[:8],
        format="%Y%m%d", errors="coerce"
    )
    df = df.dropna(subset=["date_dt"])
    if df.empty:
        return None

    annee_max = df["date_dt"].dt.year.max()
    annee_min = annee_max - n_annees + 1
    df = df[df["date_dt"].dt.year >= annee_min].copy()
    if df.empty:
        return None

    df["mois"]  = df["date_dt"].dt.month
    df["annee"] = df["date_dt"].dt.year

    # Agrégation journalière → mensuelle → normale
    agg_j = {}
    if colonne_presente(df, "t_celsius"):
        agg_j["t_moy_j"] = ("t_celsius", "mean")
    if colonne_presente(df, "t_min"):
        agg_j["t_min_j"] = ("t_min", "mean")
    if colonne_presente(df, "t_max"):
        agg_j["t_max_j"] = ("t_max", "mean")
    if colonne_presente(df, "rr_mm"):
        agg_j["rr_j"] = ("rr_mm", "sum")
    elif colonne_presente(df, "rr1_mm"):
        agg_j["rr_j"] = ("rr1_mm", "sum")

    if not agg_j:
        return None

    df_mens = df.groupby(["annee", "mois"]).agg(**agg_j).reset_index()

    agg_norm = {}
    if "t_moy_j" in df_mens.columns:
        agg_norm["t_moy_norm"] = ("t_moy_j", "mean")
    if "t_min_j" in df_mens.columns:
        agg_norm["t_min_norm"] = ("t_min_j", "mean")
    if "t_max_j" in df_mens.columns:
        agg_norm["t_max_norm"] = ("t_max_j", "mean")
    if "rr_j" in df_mens.columns:
        agg_norm["precip_norm"] = ("rr_j", "mean")

    result = df_mens.groupby("mois").agg(**agg_norm).reset_index()
    result.attrs["annee_min"] = int(annee_min)
    result.attrs["annee_max"] = int(annee_max)
    return result



# ==============================================================================
# EVAPOTRANSPIRATION OFFICIELLE (Penman-Monteith, fichier quotidien "autres-parametres")
# ==============================================================================
#
# Source : fichier quotidien Météo-France Q_{dept}_{serie}_autres-parametres.csv.gz
# — le fichier compagnon de Q_{dept}_{serie}_RR-T-Vent.csv.gz (déjà utilisé pour
# les normales de température), contenant notamment la variable ETP.
#
# Le fichier DECADAGRO initialement ciblé s'est révélé introuvable (404
# systématique, quel que soit le millésime testé). Le fichier "autres-parametres"
# est confirmé comme la bonne source par la documentation d'un outil tiers
# d'exploitation de ces mêmes données (loicduffar/meteo.data-Tools) :
# "Ce script traite uniquement le fichier RR-T-vent [...] à l'exclusion du
# fichier autres-parametres contenant notamment l'ETP."
#
# ⚠️ Nom exact de la colonne ETP non confirmé directement (aucun accès au
# fichier réel possible en test). Plusieurs candidats sont testés automatiquement.
# En cas d'échec silencieux, activer le Mode debug dans l'app pour inspecter les
# colonnes réelles du fichier téléchargé et ajuster si besoin la liste
# "etp_officielle_mm" dans COLONNES_METEO.

def _candidats_quot_autres_params(dept):
    """
    Fichier quotidien 'autres-parametres' — même bucket, même département,
    même logique de nommage que _candidats_quot_previous (RR-T-Vent) :
    la borne haute de la période "previous" change chaque année, on teste
    plusieurs valeurs récentes.
    """
    a = datetime.datetime.utcnow().year
    candidats = [
        f"Q_{dept}_previous-1950-{fin}_autres-parametres.csv.gz"
        for fin in range(a - 1, a - 8, -1)
    ]
    # Repli sans suffixe, au cas où le fichier ne le porterait pas
    candidats += [
        f"Q_{dept}_previous-1950-{fin}.csv.gz"
        for fin in range(a - 1, a - 4, -1)
    ]
    return candidats


@st.cache_data(ttl=86400, show_spinner=False, max_entries=3)
def telecharger_quotidien_etp(code_dept, n_annees=10):
    """
    Télécharge le fichier quotidien "autres-parametres" (contient l'ETP
    officielle Penman-Monteith) pour un département, filtré aux n dernières
    années disponibles.

    Retourne DataFrame ou None si le fichier est indisponible (auquel cas
    l'app affiche alors "ETP non disponible").
    """
    dept = _normaliser_dept(code_dept)
    for nom in _candidats_quot_autres_params(dept):
        df = _telecharger_gz(f"{BASE_URL_QUOT}/{nom}", colonnes_utiles=COLONNES_UTILES)
        if df is None:
            continue

        col_date = next((c for c in df.columns if c.upper().startswith("AAAA")), None)
        if col_date is not None:
            annees = pd.to_numeric(df[col_date].astype(str).str.slice(0, 4), errors="coerce")
            annee_max = annees.max()
            if pd.notna(annee_max):
                df = df[annees >= (annee_max - n_annees + 1)].copy()

        df["_fichier_source"] = nom
        return df
    return None


def calculer_etp_officielle_mensuelle(df_quot_etp, ids_stations, n_annees=10):
    """
    Calcule la moyenne mensuelle de l'ETP officielle (Penman-Monteith) sur
    les n dernières années, à partir du fichier quotidien "autres-parametres".

    Retourne DataFrame (mois, etp_officielle_mm) avec attrs annee_min/max,
    ou None si la colonne ETP n'a pas été reconnue dans le fichier
    (schéma inattendu) ou si les données sont insuffisantes.
    """
    if df_quot_etp is None or df_quot_etp.empty:
        return None

    df = df_quot_etp[
        df_quot_etp["NUM_POSTE"].astype(str).isin([str(i) for i in ids_stations])
    ].copy()
    if df.empty:
        return None

    df = normaliser_variables(df)
    if not colonne_presente(df, "etp_officielle_mm"):
        return None

    col_date = next((c for c in df.columns if c.upper().startswith("AAAA")), None)
    if col_date is None:
        return None

    df["annee"] = pd.to_numeric(df[col_date].astype(str).str.slice(0, 4), errors="coerce")
    df["mois"]  = pd.to_numeric(df[col_date].astype(str).str.slice(4, 6), errors="coerce")
    df = df.dropna(subset=["annee", "mois"])
    if df.empty:
        return None

    annee_max = df["annee"].max()
    annee_min = annee_max - n_annees + 1
    df = df[df["annee"] >= annee_min]
    if df.empty:
        return None

    # Moyenne quotidienne → mensuelle (moyenne des jours du mois) → normale
    df_mens = df.groupby(["annee", "mois"])["etp_officielle_mm"].mean().reset_index()
    result  = df_mens.groupby("mois")["etp_officielle_mm"].mean().reset_index()
    result.attrs["annee_min"] = int(annee_min)
    result.attrs["annee_max"] = int(annee_max)
    return result


# ==============================================================================
# DEBUG
# ==============================================================================

def debug_colonnes(df_brut):
    src = df_brut["_fichier_source"].iloc[0] if "_fichier_source" in df_brut.columns else "?"
    st.write(f"Fichier : `{src}` — {len(df_brut):,} lignes")
    st.write(f"Colonnes ({len(df_brut.columns)}) :", list(df_brut.columns))
    st.dataframe(df_brut.head(3), use_container_width=True)
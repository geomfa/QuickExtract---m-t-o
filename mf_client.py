"""
Client pour les données climatologiques horaires de Météo-France.
Source : data.gouv.fr (open data, aucun compte requis).

URL des fichiers :
  https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/HOR/
      H_{DEPT}_latest-{annee1}-{annee2}.csv.gz   -> données récentes, mise à jour quotidienne (J-1/J-2)
      H_{DEPT}_previous-{annee1}-{annee2}.csv.gz -> période précédente
      H_{DEPT}_{annee1}-{annee2}.csv.gz          -> historique par décennie

Format des fichiers (doc officielle Météo-France) :
  - Séparateur : ;
  - Encodage   : latin-1
  - Colonnes communes : NUM_POSTE, NOM_USUEL, LAT, LON, ALTI
  - Colonne date horaire : AAAAMMJJHH  (format %Y%m%d%H, heure UTC)
  - Variables principales :
      T     -> température (°C, déjà en °C contrairement à SYNOP/ODS qui est en Kelvin)
      U     -> humidité relative (%)
      FF    -> vitesse vent moy. 10 min (m/s)
      DD    -> direction vent moy. 10 min (degrés)
      FXY   -> rafale maximale (m/s)
      RR1   -> précipitations sur 1h (mm)
      PMER  -> pression au niveau mer (hPa)
      N     -> nébulosité (octas)
      VV    -> visibilité (m)

Note : les colonnes préfixées Q indiquent la qualité de la mesure (1 = valide).
"""

import io
import gzip
import math
import datetime
import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR"
GEO_API  = "https://geo.api.gouv.fr"


class DonneesClimatoError(Exception):
    pass


# ==============================================================================
# RESOLUTION ZONE D'ETUDE
# ==============================================================================

@st.cache_data(ttl=86400, show_spinner=False)
def commune_depuis_insee(code_insee):
    """
    Retourne (nom, lat, lon, code_dept) depuis l'API geo.api.gouv.fr.
    Lève DonneesClimatoError si le code INSEE est inconnu.
    """
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
        nom   = data["nom"]
        lat   = data["centre"]["coordinates"][1]
        lon   = data["centre"]["coordinates"][0]
        dept  = data["codeDepartement"]
        return nom, lat, lon, dept
    except DonneesClimatoError:
        raise
    except Exception as e:
        raise DonneesClimatoError(f"Erreur API geo.api.gouv.fr : {e}")


def dept_depuis_insee(code_insee):
    """Extrait le code département depuis un code INSEE commune."""
    code = str(code_insee).strip().zfill(5)
    # DOM : 971-976 -> 3 chiffres
    if code[:3] in ("971", "972", "973", "974", "976"):
        return code[:3]
    return code[:2]


# ==============================================================================
# TELECHARGEMENT DES FICHIERS CSV.GZ
# ==============================================================================

def _candidats_fichiers(dept):
    """
    Construit la liste des noms de fichiers candidats à tester, du plus récent
    au plus ancien. On tente 3 tranches 'latest' autour de l'année courante
    pour être robuste sans maintenance annuelle.
    """
    annee = datetime.datetime.utcnow().year
    return [
        f"H_{dept}_latest-{annee - 1}-{annee}.csv.gz",
        f"H_{dept}_latest-{annee}-{annee + 1}.csv.gz",
        f"H_{dept}_latest-{annee - 2}-{annee - 1}.csv.gz",
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def telecharger_horaire_departement(code_dept):
    """
    Télécharge et décompresse le fichier horaire le plus récent pour un département.

    Returns:
        DataFrame brut avec une colonne '_fichier_source' ajoutée.

    Raises:
        DonneesClimatoError si aucun fichier n'est accessible.
    """
    dept = _normaliser_dept(code_dept)
    candidats = _candidats_fichiers(dept)
    derniere_erreur = None

    for nom in candidats:
        url = f"{BASE_URL}/{nom}"
        try:
            resp = requests.get(url, timeout=90)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            derniere_erreur = e
            continue

        try:
            raw = gzip.decompress(resp.content)
        except OSError as e:
            derniere_erreur = e
            continue

        df = pd.read_csv(
            io.BytesIO(raw),
            sep=";",
            encoding="latin-1",
            low_memory=False,
            dtype={"NUM_POSTE": str},
        )
        df["_fichier_source"] = nom
        return df

    raise DonneesClimatoError(
        f"Aucun fichier horaire trouvé pour le département '{dept}'. "
        f"Fichiers testés : {candidats}. Dernière erreur : {derniere_erreur}"
    )


def _normaliser_dept(code):
    """Normalise un code département : '29' -> '29', '9' -> '09', '971' -> '971'."""
    code = str(code).strip().upper()
    if len(code) == 1:
        return "0" + code
    return code


# ==============================================================================
# REFERENTIEL STATIONS
# ==============================================================================

def stations_du_fichier(df_brut):
    """Extrait les stations distinctes (une ligne par station)."""
    cols = [c for c in ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI"] if c in df_brut.columns]
    if "NUM_POSTE" not in cols:
        return pd.DataFrame()
    return (
        df_brut[cols]
        .drop_duplicates(subset=["NUM_POSTE"])
        .reset_index(drop=True)
    )


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def stations_proches(df_brut, lat, lon, n=5):
    """
    Retourne les n stations les plus proches du point (lat, lon).
    Ajoute une colonne 'distance_km'.
    """
    df = stations_du_fichier(df_brut)
    if df.empty or "LAT" not in df.columns or "LON" not in df.columns:
        return df

    df = df.copy()
    df["distance_km"] = df.apply(
        lambda r: round(_haversine_km(lat, lon, float(r["LAT"]), float(r["LON"])), 1),
        axis=1,
    )
    return df.sort_values("distance_km").head(n).reset_index(drop=True)


# ==============================================================================
# FILTRAGE TEMPOREL
# ==============================================================================

def filtrer_periode(df_brut, date_debut, date_fin):
    """
    Filtre le DataFrame brut sur une fenêtre temporelle.
    La colonne date horaire est AAAAMMJJHH (format %Y%m%d%H, heure UTC).

    Returns:
        DataFrame filtré avec colonne 'date_dt' (datetime naïf UTC) ajoutée.
    """
    col_date = "AAAAMMJJHH"
    if col_date not in df_brut.columns:
        # Tentative de détection automatique
        candidats = [c for c in df_brut.columns if c.upper().startswith("AAAA")]
        if not candidats:
            raise DonneesClimatoError(
                f"Colonne date introuvable. Colonnes disponibles : {list(df_brut.columns)}"
            )
        col_date = candidats[0]

    df = df_brut.copy()
    df["date_dt"] = pd.to_datetime(
        df[col_date].astype(str).str.strip(),
        format="%Y%m%d%H",
        errors="coerce",
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
# NORMALISATION DES VARIABLES METEO
# ==============================================================================

# Correspondance nom logique -> candidats de colonnes dans le fichier
COLONNES_METEO = {
    "t_celsius": ["T"],
    "u_pct":     ["U"],
    "ff_ms":     ["FF"],
    "dd_deg":    ["DD"],
    "fx_ms":     ["FXY", "FX", "FXXY"],
    "rr1_mm":    ["RR1"],
    "pmer_hpa":  ["PMER"],
    "n_octas":   ["N"],
    "vv_m":      ["VV"],
}


def normaliser_variables(df):
    """
    Ajoute les colonnes normalisées à partir du DataFrame brut filtré.
    Seules les colonnes effectivement présentes dans le fichier sont créées.
    """
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
# AGREGATION MULTI-STATIONS (ponderation inverse-distance)
# ==============================================================================

def agreger_multi_stations(df_obs, df_stations):
    """
    Agrège les observations de plusieurs stations en une série temporelle unique,
    pondérée par l'inverse de la distance au point d'intérêt.

    Args:
        df_obs      : DataFrame filtré avec colonne 'date_dt' et 'NUM_POSTE'
        df_stations : DataFrame stations avec colonnes 'NUM_POSTE', 'distance_km'

    Returns:
        DataFrame agrégé indexé par 'date_dt', une ligne par pas de temps.
    """
    if df_obs.empty or df_stations.empty:
        return pd.DataFrame()

    poids_map = {
        str(row["NUM_POSTE"]): 1.0 / max(row["distance_km"], 0.1)
        for _, row in df_stations.iterrows()
    }

    df = df_obs.copy()
    df["NUM_POSTE"] = df["NUM_POSTE"].astype(str)
    df["_poids"] = df["NUM_POSTE"].map(poids_map).fillna(0.0)

    cols_num = [c for c in COLONNES_METEO if c in df.columns]

    def _wavg(g):
        out = {}
        for col in cols_num:
            valide = g[[col, "_poids"]].dropna(subset=[col])
            if valide.empty:
                out[col] = float("nan")
            else:
                out[col] = (valide[col] * valide["_poids"]).sum() / valide["_poids"].sum()
        out["n_stations"] = g["NUM_POSTE"].nunique()
        return pd.Series(out)

    return df.groupby("date_dt").apply(_wavg).reset_index()


# ==============================================================================
# PIPELINE COMPLET
# ==============================================================================

def pipeline_horaire(code_dept, lat, lon, date_debut, date_fin, n_stations=5):
    """
    Télécharge, filtre, sélectionne les stations proches et agrège.

    Returns:
        tuple (df_agg, df_stations, df_obs_brut, nom_fichier)
        - df_agg        : série agrégée multi-stations
        - df_stations   : référentiel des stations retenues
        - df_obs_brut   : observations brutes filtrées (toutes stations)
        - nom_fichier   : nom du fichier source téléchargé
    """
    df_brut = telecharger_horaire_departement(code_dept)
    nom_fichier = df_brut["_fichier_source"].iloc[0] if "_fichier_source" in df_brut.columns else ""

    df_sta = stations_proches(df_brut, lat, lon, n=n_stations)
    if df_sta.empty:
        raise DonneesClimatoError("Aucune station trouvée dans le fichier.")

    ids = df_sta["NUM_POSTE"].tolist()
    df_filtre = df_brut[df_brut["NUM_POSTE"].astype(str).isin([str(i) for i in ids])]
    df_obs = filtrer_periode(df_filtre, date_debut, date_fin)

    if df_obs.empty:
        return pd.DataFrame(), df_sta, pd.DataFrame(), nom_fichier

    df_obs = normaliser_variables(df_obs)
    df_agg = agreger_multi_stations(df_obs, df_sta)

    return df_agg, df_sta, df_obs, nom_fichier


# ==============================================================================
# UTILITAIRE DEBUG
# ==============================================================================

def debug_colonnes(df_brut):
    """Affiche dans Streamlit la structure du fichier téléchargé."""
    import streamlit as st
    src = df_brut["_fichier_source"].iloc[0] if "_fichier_source" in df_brut.columns else "?"
    st.write(f"Fichier : `{src}` — {len(df_brut):,} lignes")
    st.write(f"Colonnes ({len(df_brut.columns)}) :", list(df_brut.columns))
    st.dataframe(df_brut.head(3), use_container_width=True)

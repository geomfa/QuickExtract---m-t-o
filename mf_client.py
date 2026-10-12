"""
Client pour les données climatologiques de Météo-France.
Source : bucket public Météo-France (open data, aucun compte requis).

Fichiers horaires (BASE/HOR) :
  H_{DEPT}_latest-{a1}-{a2}.csv.gz
  Colonne date : AAAAMMJJHH. Variables : T, U, FF, DD, RR1, N...

Fichiers quotidiens (BASE/QUOT) :
  Q_{DEPT}_previous-1950-{a}_RR-T-Vent.csv.gz         T°, précipitations (normales)
  Q_{DEPT}_previous-1950-{a}_autres-parametres.csv.gz  contient l'ETP (Penman-Monteith)
  Q_{DEPT}_latest-{a1}-{a2}_autres-parametres.csv.gz   ETP de la période récente
  Colonne date : AAAAMMJJ

Format commun : séparateur ';', encodage latin-1.
Colonnes communes : NUM_POSTE, NOM_USUEL, LAT, LON, ALTI.

Règles de calcul
  - Une grandeur cumulative (précipitations, ETP) est sommée par jour puis par
    mois ; une grandeur intensive (température) est moyennée.
  - Un mois n'entre dans une normale ou un bilan que s'il est renseigné à au
    moins SEUIL_COUVERTURE_MOIS (90 %) ; un cumul est alors ramené au mois entier.
  - Les normales sont d'abord calculées station par station, puis moyennées
    entre stations (jamais sommées entre stations).
"""

import io
import gzip
import math
import calendar
import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st

BASE_URL_HOR  = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR"
BASE_URL_QUOT = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/QUOT"
GEO_API       = "https://geo.api.gouv.fr"

COLONNES_METEO = {
    "t_celsius": ["TM", "T"],       # TM = T° moyenne quotidienne, T = horaire
    "t_min":     ["TN"],
    "t_max":     ["TX"],
    "u_pct":     ["U"],
    "ff_ms":     ["FF"],
    "dd_deg":    ["DD"],
    "fx_ms":     ["FXY", "FX", "FXXY"],
    "rr1_mm":    ["RR1"],
    "rr_mm":     ["RR"],            # précipitation quotidienne
    "pmer_hpa":  ["PMER"],
    "n_octas":   ["N"],
    "vv_m":      ["VV"],
    # ETP Penman-Monteith (fichier quotidien "autres-parametres"). Le nom exact
    # de la colonne est testé parmi plusieurs candidats ; le mode debug de l'app
    # affiche les colonnes réelles en cas de doute.
    "etp_officielle_mm": ["ETPMON", "ETP", "ETPP", "ETPPM", "ETPGRILLE"],
}

# Groupes de variables, utilisés par l'app pour harmoniser les stations.
COLS_METEO = ["t_celsius", "rr1_mm", "u_pct"]
COLS_VENT  = ["ff_ms", "dd_deg"]

COLONNES_BASE = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI"]
COLONNES_DATE = ["AAAAMMJJHH", "AAAAMMJJ"]
COLONNES_UTILES = (
    COLONNES_BASE + COLONNES_DATE
    + [c for cands in COLONNES_METEO.values() for c in cands]
)

SEUIL_COUVERTURE_MOIS = 0.9   # part minimale de jours renseignés pour retenir un mois
SEUILS_INTENSITE_PLUIE = (2.5, 7.6, 15.0)   # mm/h, repères indicatifs

CLASSES_PLUIE = [
    "Faible (0,1 à 2,5 mm/h)",
    "Modérée (2,5 à 7,6 mm/h)",
    "Forte (7,6 à 15 mm/h)",
    "Très forte (plus de 15 mm/h)",
]


class DonneesClimatoError(Exception):
    pass


# ==============================================================================
# UTILITAIRES
# ==============================================================================

def _utcnow():
    """Date et heure UTC courantes, sans fuseau (comparables aux données)."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


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


def _col_date(df):
    return next((c for c in df.columns if c.upper().startswith("AAAA")), None)


def _parser_dates(df, col_date=None):
    """Convertit la colonne date (AAAAMMJJHH ou AAAAMMJJ) en datetime."""
    col_date = col_date or _col_date(df)
    if col_date is None:
        raise DonneesClimatoError(
            f"Colonne date introuvable. Colonnes : {list(df.columns)}"
        )
    fmt = "%Y%m%d%H" if col_date.upper().endswith("HH") else "%Y%m%d"
    num = pd.to_numeric(df[col_date], errors="coerce").astype("Int64").astype(str)
    return pd.to_datetime(num, format=fmt, errors="coerce")


def _serie_logique(df, nom_logique):
    """Série numérique d'une variable logique (première colonne candidate présente)."""
    for c in COLONNES_METEO.get(nom_logique, []):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    if nom_logique == "etp_officielle_mm":
        # Repli : toute colonne dont le nom commence par ETP
        autre = next((c for c in df.columns if c.upper().startswith("ETP")), None)
        if autre is not None:
            return pd.to_numeric(df[autre], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _poids_stations(df_stations):
    """Poids inverse-distance par station (dict NUM_POSTE -> poids)."""
    if df_stations is None or len(df_stations) == 0:
        return {}
    if "distance_km" not in df_stations.columns:
        return {str(c): 1.0 for c in df_stations["NUM_POSTE"]}
    return {
        str(r["NUM_POSTE"]): 1.0 / max(float(r["distance_km"]), 0.1)
        for _, r in df_stations.iterrows()
    }


def _telecharger_gz(url, colonnes_utiles=None, prefixes=()):
    """
    Télécharge et décompresse un CSV.gz. Retourne DataFrame ou None.
    Si colonnes_utiles est fourni, ne charge que ces colonnes (celles présentes) :
    indispensable pour tenir dans la mémoire de Streamlit Cloud. `prefixes`
    ajoute les colonnes dont le nom commence par l'un d'eux (ex. "ETP").
    """
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        raw = gzip.decompress(resp.content)

        if colonnes_utiles is not None:
            entete = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1", nrows=0)
            usecols = [c for c in entete.columns
                       if c in colonnes_utiles
                       or any(c.upper().startswith(p) for p in prefixes)]
            return pd.read_csv(
                io.BytesIO(raw), sep=";", encoding="latin-1",
                low_memory=False, dtype={"NUM_POSTE": str}, usecols=usecols,
            )
        return pd.read_csv(
            io.BytesIO(raw), sep=";", encoding="latin-1",
            low_memory=False, dtype={"NUM_POSTE": str},
        )
    except Exception:
        return None


def _filtrer_dernieres_annees(df, n_annees):
    """Ne garde que les n dernières années présentes dans le fichier."""
    col = _col_date(df)
    if col is None or df.empty:
        return df
    num = pd.to_numeric(df[col], errors="coerce")
    diviseur = 1_000_000 if col.upper().endswith("HH") else 10_000
    annees = num // diviseur
    annee_max = annees.max()
    if pd.isna(annee_max):
        return df
    return df[annees >= (annee_max - n_annees + 1)].copy()


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
        return (data["nom"], data["centre"]["coordinates"][1],
                data["centre"]["coordinates"][0], data["codeDepartement"])
    except DonneesClimatoError:
        raise
    except Exception as e:
        raise DonneesClimatoError(f"Erreur API geo.api.gouv.fr : {e}")


# ==============================================================================
# TELECHARGEMENTS
# ==============================================================================

def _candidats_hor(dept):
    a = _utcnow().year
    return [
        f"H_{dept}_latest-{a-1}-{a}.csv.gz",
        f"H_{dept}_latest-{a}-{a+1}.csv.gz",
        f"H_{dept}_latest-{a-2}-{a-1}.csv.gz",
    ]


def _candidats_quot_previous(dept, suffixe="RR-T-Vent"):
    a = _utcnow().year
    return [f"Q_{dept}_previous-1950-{fin}_{suffixe}.csv.gz"
            for fin in range(a - 1, a - 8, -1)]


def _candidats_quot_latest(dept, suffixe):
    a = _utcnow().year
    return [
        f"Q_{dept}_latest-{a-1}-{a}_{suffixe}.csv.gz",
        f"Q_{dept}_latest-{a}-{a+1}_{suffixe}.csv.gz",
        f"Q_{dept}_latest-{a-2}-{a-1}_{suffixe}.csv.gz",
    ]


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


@st.cache_data(ttl=86400, show_spinner=False, max_entries=3)
def telecharger_quotidien_previous(code_dept, n_annees=10):
    """Fichier quotidien T°/précipitations, limité aux n dernières années."""
    dept = _normaliser_dept(code_dept)
    for nom in _candidats_quot_previous(dept, "RR-T-Vent"):
        df = _telecharger_gz(f"{BASE_URL_QUOT}/{nom}", colonnes_utiles=COLONNES_UTILES)
        if df is not None:
            df = _filtrer_dernieres_annees(df, n_annees)
            df["_fichier_source"] = nom
            return df
    return None


@st.cache_data(ttl=86400, show_spinner=False, max_entries=3)
def telecharger_quotidien_etp(code_dept, n_annees=10):
    """Fichier quotidien "autres-parametres" (ETP), période historique."""
    dept = _normaliser_dept(code_dept)
    for nom in _candidats_quot_previous(dept, "autres-parametres"):
        df = _telecharger_gz(f"{BASE_URL_QUOT}/{nom}", colonnes_utiles=COLONNES_UTILES,
                             prefixes=("ETP",))
        if df is not None:
            df = _filtrer_dernieres_annees(df, n_annees)
            df["_fichier_source"] = nom
            return df
    return None


@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def telecharger_quotidien_etp_latest(code_dept):
    """Fichier quotidien "autres-parametres" (ETP), période récente."""
    dept = _normaliser_dept(code_dept)
    for nom in _candidats_quot_latest(dept, "autres-parametres"):
        df = _telecharger_gz(f"{BASE_URL_QUOT}/{nom}", colonnes_utiles=COLONNES_UTILES,
                             prefixes=("ETP",))
        if df is not None:
            df["_fichier_source"] = nom
            return df
    return None


def combiner_etp(df_previous, df_latest):
    """
    Réunit l'ETP historique et récente. En cas de doublon (même station, même
    jour), la valeur du fichier récent l'emporte. Retourne None si aucun fichier.
    """
    frames = [d for d in (df_previous, df_latest) if d is not None and not d.empty]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    col = _col_date(df)
    if col is None:
        return df
    return df.drop_duplicates(subset=["NUM_POSTE", col], keep="last").reset_index(drop=True)


def plage_fichier_horaire(df_brut):
    """(première, dernière) date du fichier horaire, ou (None, None)."""
    if df_brut is None or df_brut.empty:
        return None, None
    col = _col_date(df_brut)
    if col is None:
        return None, None
    num = pd.to_numeric(df_brut[col], errors="coerce")
    fmt = "%Y%m%d%H" if col.upper().endswith("HH") else "%Y%m%d"
    bornes = []
    for v in (num.min(), num.max()):
        if pd.isna(v):
            return None, None
        bornes.append(pd.to_datetime(str(int(v)), format=fmt, errors="coerce"))
    return bornes[0], bornes[1]


# ==============================================================================
# INSPECTION DES STATIONS
# ==============================================================================

def _ids_avec_valeurs(df, nom_logique):
    if df is None or len(df) == 0 or "NUM_POSTE" not in df.columns:
        return set()
    vals = _serie_logique(df, nom_logique)
    return set(df.loc[vals.notna(), "NUM_POSTE"].astype(str).unique())


def _referentiel_stations(*dfs):
    """Table des stations (code, nom, coordonnées) à partir de plusieurs fichiers."""
    frames = []
    for d in dfs:
        if d is None or len(d) == 0:
            continue
        cols = [c for c in COLONNES_BASE if c in d.columns]
        if "NUM_POSTE" in cols and "LAT" in cols and "LON" in cols:
            frames.append(d[cols].drop_duplicates(subset=["NUM_POSTE"]))
    if not frames:
        return pd.DataFrame()
    ref = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["NUM_POSTE"])
    ref["NUM_POSTE"] = ref["NUM_POSTE"].astype(str)
    ref["LAT"] = pd.to_numeric(ref["LAT"], errors="coerce")
    ref["LON"] = pd.to_numeric(ref["LON"], errors="coerce")
    return ref.dropna(subset=["LAT", "LON"]).reset_index(drop=True)


def inspecter_stations(df_brut, df_quot, df_etp, lat, lon, n=10, n_etp_sup=3):
    """
    Décrit les n stations les plus proches, plus (au plus) n_etp_sup stations
    supplémentaires disposant d'ETP si aucune ne figure parmi les n premières.

    Colonnes : NUM_POSTE, NOM_USUEL, LAT, LON, ALTI, distance_km, derniere_date,
    fraicheur_jours, nebulo_dispo, historique_dispo, vent_dispo, meteo_dispo,
    etp_dispo.
    """
    ref = _referentiel_stations(df_brut, df_etp)
    if ref.empty:
        return ref

    ref["distance_km"] = [
        round(_haversine_km(lat, lon, a, b), 1) for a, b in zip(ref["LAT"], ref["LON"])
    ]
    ref = ref.sort_values("distance_km").reset_index(drop=True)

    ids_etp = _ids_avec_valeurs(df_etp, "etp_officielle_mm")

    df_sta = ref.head(n)
    if n_etp_sup:
        deja_etp = df_sta["NUM_POSTE"].isin(ids_etp).sum()
        manque = max(n_etp_sup - int(deja_etp), 0)
        if manque:
            sup = ref.iloc[n:]
            sup = sup[sup["NUM_POSTE"].isin(ids_etp)].head(manque)
            df_sta = pd.concat([df_sta, sup], ignore_index=True)
    df_sta = df_sta.sort_values("distance_km").reset_index(drop=True)
    ids = set(df_sta["NUM_POSTE"])

    # Fraîcheur : dernière mesure horaire par station
    derniere = pd.Series(dtype="datetime64[ns]")
    if df_brut is not None and not df_brut.empty:
        sous = df_brut[df_brut["NUM_POSTE"].astype(str).isin(ids)]
        if not sous.empty:
            tmp = pd.DataFrame({
                "NUM_POSTE": sous["NUM_POSTE"].astype(str).values,
                "_dt": _parser_dates(sous).values,
            })
            derniere = tmp.groupby("NUM_POSTE")["_dt"].max()
    maintenant = _utcnow()
    dernieres, fraicheurs = [], []
    for code in df_sta["NUM_POSTE"]:
        d = derniere.get(code, pd.NaT) if len(derniere) else pd.NaT
        if pd.isna(d):
            dernieres.append("n/d")
            fraicheurs.append(None)
        else:
            dernieres.append(pd.Timestamp(d).strftime("%d/%m/%Y %Hh"))
            fraicheurs.append((maintenant - pd.Timestamp(d).to_pydatetime()).days)
    df_sta["derniere_date"] = dernieres
    df_sta["fraicheur_jours"] = fraicheurs

    ids_n = _ids_avec_valeurs(df_brut, "n_octas")
    ids_t = _ids_avec_valeurs(df_brut, "t_celsius")
    ids_ff = _ids_avec_valeurs(df_brut, "ff_ms")
    ids_dd = _ids_avec_valeurs(df_brut, "dd_deg")
    ids_hist = _ids_avec_valeurs(df_quot, "t_celsius") | _ids_avec_valeurs(df_quot, "rr_mm")

    df_sta["nebulo_dispo"] = df_sta["NUM_POSTE"].isin(ids_n)
    df_sta["historique_dispo"] = df_sta["NUM_POSTE"].isin(ids_hist)
    df_sta["meteo_dispo"] = df_sta["NUM_POSTE"].isin(ids_t)
    df_sta["vent_dispo"] = df_sta["NUM_POSTE"].isin(ids_ff & ids_dd)
    df_sta["etp_dispo"] = df_sta["NUM_POSTE"].isin(ids_etp)
    return df_sta


# ==============================================================================
# FILTRAGE, NORMALISATION
# ==============================================================================

def filtrer_periode(df_brut, date_debut, date_fin, col_date=None):
    df = df_brut.copy()
    df["date_dt"] = _parser_dates(df, col_date)
    df = df.dropna(subset=["date_dt"])
    d1 = pd.Timestamp(date_debut)
    d2 = pd.Timestamp(date_fin)
    if d1.tzinfo is not None:
        d1 = d1.tz_localize(None)
    if d2.tzinfo is not None:
        d2 = d2.tz_localize(None)
    return df[(df["date_dt"] >= d1) & (df["date_dt"] <= d2)].sort_values("date_dt")


def normaliser_variables(df):
    df = df.copy()
    for nom_logique, candidats in COLONNES_METEO.items():
        for c in candidats:
            if c in df.columns:
                df[nom_logique] = pd.to_numeric(df[c], errors="coerce")
                break
    return df


def colonne_presente(df, nom_logique):
    return (df is not None and nom_logique in df.columns
            and df[nom_logique].notna().any())


# ==============================================================================
# AGREGATION MULTI-STATIONS (observations)
# ==============================================================================

def agreger_multi_stations(df_obs, df_stations, colonnes=None):
    """
    Moyenne pondérée inverse-distance des stations, pas de temps par pas de temps.

    colonnes : variables logiques à agréger (par défaut toutes celles présentes).
    La direction du vent est moyennée vectoriellement (une moyenne arithmétique
    de 350 et 10 degrés donnerait 180).
    """
    if df_obs is None or df_obs.empty or df_stations is None or len(df_stations) == 0:
        return pd.DataFrame()

    poids_map = _poids_stations(df_stations)
    df = df_obs.copy()
    df["NUM_POSTE"] = df["NUM_POSTE"].astype(str)
    df["_poids"] = df["NUM_POSTE"].map(poids_map).fillna(0.0)
    df = df[df["_poids"] > 0]
    if df.empty:
        return pd.DataFrame()

    candidates = colonnes if colonnes is not None else list(COLONNES_METEO)
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return pd.DataFrame()

    out = pd.DataFrame(index=pd.Index(sorted(df["date_dt"].unique()), name="date_dt"))
    for col in cols:
        v = df[col]
        wv = df["_poids"].where(v.notna(), 0.0)
        if col == "dd_deg":
            r = np.radians(v)
            tmp = pd.DataFrame({
                "date_dt": df["date_dt"],
                "sx": (wv * np.sin(r)).fillna(0.0),
                "cx": (wv * np.cos(r)).fillna(0.0),
            })
            s = tmp.groupby("date_dt")[["sx", "cx"]].sum()
            ang = pd.Series(np.degrees(np.arctan2(s["sx"], s["cx"])) % 360, index=s.index)
            ang[(s["sx"].abs() + s["cx"].abs()) < 1e-12] = np.nan
            out[col] = ang
        else:
            tmp = pd.DataFrame({
                "date_dt": df["date_dt"],
                "num": v.fillna(0.0) * wv,
                "den": wv,
            })
            s = tmp.groupby("date_dt")[["num", "den"]].sum()
            out[col] = s["num"] / s["den"].replace(0.0, np.nan)

    out["n_stations"] = df.groupby("date_dt")["NUM_POSTE"].nunique()
    return out.reset_index()


# ==============================================================================
# AGREGATS JOURNALIERS / MENSUELS / ANNUELS (sur observations agrégées)
# ==============================================================================

def _jours_entiers(date_debut, date_fin):
    """Jours calendaires entièrement compris dans la période."""
    d1, d2 = pd.Timestamp(date_debut), pd.Timestamp(date_fin)
    premier = d1.normalize() if d1 == d1.normalize() else d1.normalize() + pd.Timedelta(days=1)
    dernier = d2.normalize() if d2.hour >= 23 else d2.normalize() - pd.Timedelta(days=1)
    if dernier < premier:
        return pd.DatetimeIndex([])
    return pd.date_range(premier, dernier, freq="D")


def agreger_journalier(df_agg, pas_heures=1):
    """
    Table journalière : t_moy, t_min, t_max (min/max des valeurs du jour),
    precip_tot, vent, humidité, et couverture_precip (part des pas de temps
    attendus qui portent une valeur de précipitation).
    """
    if df_agg is None or df_agg.empty:
        return pd.DataFrame()
    d = df_agg.copy()
    d["date_seule"] = d["date_dt"].dt.normalize()

    spec = {}
    if colonne_presente(d, "t_celsius"):
        spec.update(t_moy=("t_celsius", "mean"), t_min=("t_celsius", "min"),
                    t_max=("t_celsius", "max"))
    if colonne_presente(d, "rr1_mm"):
        spec["precip_tot"] = ("rr1_mm", lambda s: s.sum(min_count=1))
        spec["precip_n"] = ("rr1_mm", "count")
    if colonne_presente(d, "ff_ms"):
        spec.update(vent_moy_ms=("ff_ms", "mean"), vent_max_ms=("ff_ms", "max"))
    if colonne_presente(d, "u_pct"):
        spec["humidite_moy"] = ("u_pct", "mean")
    if not spec:
        return pd.DataFrame()

    res = d.groupby("date_seule").agg(**spec).reset_index()
    if "precip_n" in res.columns:
        res["couverture_precip"] = (res["precip_n"] / (24.0 / pas_heures)).clip(upper=1.0)
    res["date_dt"] = res["date_seule"]
    res["mois"] = res["date_dt"].dt.month
    res["annee"] = res["date_dt"].dt.year
    return res


def _agreger_periodes(agg_j, cle, date_debut, date_fin):
    """Mois (cle='mois') ou années (cle='annee') à partir de la table journalière."""
    if agg_j is None or agg_j.empty:
        return pd.DataFrame()
    keys = ["annee", "mois"] if cle == "mois" else ["annee"]
    spec = {}
    if "t_moy" in agg_j.columns:
        spec.update(t_moy=("t_moy", "mean"), t_min=("t_min", "mean"), t_max=("t_max", "mean"))
    if "vent_moy_ms" in agg_j.columns:
        spec["vent_moy_ms"] = ("vent_moy_ms", "mean")
        spec["vent_max_ms"] = ("vent_max_ms", "max")
    if "humidite_moy" in agg_j.columns:
        spec["humidite_moy"] = ("humidite_moy", "mean")
    spec["jours_ok"] = ("date_dt", "count")
    has_p = "precip_tot" in agg_j.columns
    if has_p:
        spec["precip_somme"] = ("precip_tot", lambda s: s.sum(min_count=1))
        spec["jours_precip"] = ("precip_tot", "count")

    res = agg_j.groupby(keys).agg(**spec).reset_index()

    if cle == "mois":
        res["jours_periode"] = [calendar.monthrange(int(a), int(m))[1]
                                for a, m in zip(res["annee"], res["mois"])]
    else:
        res["jours_periode"] = [366 if calendar.isleap(int(a)) else 365 for a in res["annee"]]

    entiers = _jours_entiers(date_debut, date_fin)
    if cle == "mois":
        compte = pd.Series(1, index=entiers).groupby([entiers.year, entiers.month]).sum()
        res["periode_incluse"] = [
            compte.get((int(a), int(m)), 0) == dim
            for a, m, dim in zip(res["annee"], res["mois"], res["jours_periode"])
        ]
    else:
        compte = pd.Series(1, index=entiers).groupby(entiers.year).sum()
        res["periode_incluse"] = [
            compte.get(int(a), 0) == dim for a, dim in zip(res["annee"], res["jours_periode"])
        ]

    res["couverture"] = res["jours_ok"] / res["jours_periode"]
    res["retenu"] = res["periode_incluse"] & (res["couverture"] >= SEUIL_COUVERTURE_MOIS)

    if has_p:
        # Cumul ramené à la période entière quand quelques jours manquent.
        ok = res["jours_precip"] > 0
        res["precip_tot"] = np.where(
            ok, res["precip_somme"] * res["jours_periode"] / res["jours_precip"].replace(0, np.nan),
            np.nan,
        )
        res = res.drop(columns=["precip_somme", "jours_precip"])

    if cle == "mois":
        res["date_dt"] = pd.to_datetime(
            {"year": res["annee"], "month": res["mois"], "day": 1}
        )
    return res


def agreger_mensuel(agg_j, date_debut, date_fin):
    return _agreger_periodes(agg_j, "mois", date_debut, date_fin)


def agreger_annuel(agg_j, date_debut, date_fin):
    return _agreger_periodes(agg_j, "annee", date_debut, date_fin)


# ==============================================================================
# NORMALES (10 dernières années)
# ==============================================================================

def _preparer_quotidien(df, ids_stations, n_annees):
    """Filtre stations et années, ajoute date_dt, annee, mois. None si vide."""
    if df is None or df.empty:
        return None
    ids = [str(i) for i in ids_stations]
    sous = df[df["NUM_POSTE"].astype(str).isin(ids)].copy()
    if sous.empty:
        return None
    sous["NUM_POSTE"] = sous["NUM_POSTE"].astype(str)
    sous["date_dt"] = _parser_dates(sous)
    sous = sous.dropna(subset=["date_dt"])
    if sous.empty:
        return None
    sous["annee"] = sous["date_dt"].dt.year
    sous["mois"] = sous["date_dt"].dt.month
    annee_max = sous["annee"].max()
    return sous[sous["annee"] >= annee_max - n_annees + 1].copy()


def _mensuel_par_station(df, spec):
    """
    Valeur mensuelle par station. spec : {sortie: (colonne, 'mean'|'sum')}.
    Un mois renseigné à moins de SEUIL_COUVERTURE_MOIS est écarté ; un cumul
    est ramené au mois entier.
    """
    keys = ["NUM_POSTE", "annee", "mois"]
    res = None
    for sortie, (col, how) in spec.items():
        if col not in df.columns or df[col].notna().sum() == 0:
            continue
        g = df.groupby(keys)[col]
        val = g.mean() if how == "mean" else g.sum(min_count=1)
        part = pd.DataFrame({sortie: val, sortie + "__n": g.count()})
        res = part if res is None else res.join(part, how="outer")
    if res is None:
        return None
    res = res.reset_index()
    dim = pd.to_datetime(
        {"year": res["annee"], "month": res["mois"], "day": 1}
    ).dt.days_in_month
    seuil = np.ceil(SEUIL_COUVERTURE_MOIS * dim)
    for sortie, (col, how) in spec.items():
        if sortie not in res.columns:
            continue
        n = res[sortie + "__n"]
        if how == "sum":
            res[sortie] = res[sortie] * dim / n.replace(0, np.nan)
        res.loc[n < seuil, sortie] = np.nan
        res = res.drop(columns=[sortie + "__n"])
    return res


def _moyenne_entre_stations(res, cols, poids):
    """Moyenne pondérée entre stations pour chaque (annee, mois), valeurs absentes ignorées."""
    w = res["NUM_POSTE"].map(poids).fillna(1.0) if poids else pd.Series(1.0, index=res.index)
    out = None
    for c in cols:
        wv = w.where(res[c].notna(), 0.0)
        tmp = pd.DataFrame({
            "annee": res["annee"], "mois": res["mois"],
            "num": res[c].fillna(0.0) * wv, "den": wv,
        })
        s = tmp.groupby(["annee", "mois"])[["num", "den"]].sum()
        serie = (s["num"] / s["den"].replace(0.0, np.nan)).rename(c)
        out = serie.to_frame() if out is None else out.join(serie, how="outer")
    return out.reset_index() if out is not None else None


def calculer_normales(df_quot, ids_stations, n_annees=10, poids=None):
    """
    Normales mensuelles sur les n dernières années disponibles.
    Colonnes : mois, t_moy_norm, t_min_norm, t_max_norm, precip_norm, n_annees.
    attrs : annee_min, annee_max (années ayant réellement contribué).
    """
    base = _preparer_quotidien(df_quot, ids_stations, n_annees)
    if base is None:
        return None
    base = normaliser_variables(base)

    spec = {
        "t_moy_j": ("t_celsius", "mean"),
        "t_min_j": ("t_min", "mean"),
        "t_max_j": ("t_max", "mean"),
    }
    if colonne_presente(base, "rr_mm"):
        spec["rr_j"] = ("rr_mm", "sum")
    elif colonne_presente(base, "rr1_mm"):
        spec["rr_j"] = ("rr1_mm", "sum")

    par_station = _mensuel_par_station(base, spec)
    if par_station is None:
        return None
    cols = [c for c in spec if c in par_station.columns]
    mens = _moyenne_entre_stations(par_station, cols, poids)
    if mens is None or mens.empty:
        return None

    noms = {"t_moy_j": "t_moy_norm", "t_min_j": "t_min_norm",
            "t_max_j": "t_max_norm", "rr_j": "precip_norm"}
    agg = {noms[c]: (c, "mean") for c in cols}
    agg["n_annees"] = (cols[0], "count")
    res = mens.groupby("mois").agg(**agg).reset_index()

    utiles = mens.dropna(subset=cols, how="all")
    if utiles.empty:
        return None
    res.attrs["annee_min"] = int(utiles["annee"].min())
    res.attrs["annee_max"] = int(utiles["annee"].max())
    return res


def calculer_etp_officielle_mensuelle(df_etp, ids_stations, n_annees=10, poids=None):
    """
    Normale mensuelle d'ETP (cumul mensuel moyenné sur les années retenues).
    Colonnes : mois, etp_officielle_mm. None si la colonne ETP est introuvable
    ou les données insuffisantes.
    """
    base = _preparer_quotidien(df_etp, ids_stations, n_annees)
    if base is None:
        return None
    base["etp_officielle_mm"] = _serie_logique(base, "etp_officielle_mm")
    if base["etp_officielle_mm"].notna().sum() == 0:
        return None

    par_station = _mensuel_par_station(base, {"etp_m": ("etp_officielle_mm", "sum")})
    if par_station is None:
        return None
    mens = _moyenne_entre_stations(par_station, ["etp_m"], poids)
    mens = mens.dropna(subset=["etp_m"]) if mens is not None else None
    if mens is None or mens.empty:
        return None

    res = (mens.groupby("mois")["etp_m"].mean()
           .rename("etp_officielle_mm").reset_index())
    res.attrs["annee_min"] = int(mens["annee"].min())
    res.attrs["annee_max"] = int(mens["annee"].max())
    return res


def cumul_normale_jours(df_norm, colonne, jours):
    """
    Cumul de la normale mensuelle `colonne` (mm/mois) sur une liste de jours :
    chaque jour compte pour 1/(nombre de jours du mois) de la normale du mois.
    Retourne None si un mois concerné n'a pas de normale.
    """
    if df_norm is None or colonne not in df_norm.columns or len(jours) == 0:
        return None
    table = dict(zip(df_norm["mois"], df_norm[colonne]))
    total = 0.0
    for j in pd.DatetimeIndex(jours):
        val = table.get(j.month)
        if val is None or pd.isna(val):
            return None
        total += val / calendar.monthrange(j.year, j.month)[1]
    return total


# ==============================================================================
# ETP : PERIODE ET JOURNALIER
# ==============================================================================

def recuperer_etp_journaliere(df_etp, ids_stations, date_debut, date_fin, poids=None):
    """
    ETP quotidienne moyennée (pondérée) sur les stations, entre deux dates.
    Colonnes : date_dt, etp_mm, n_stations. None si indisponible.
    """
    if df_etp is None or df_etp.empty:
        return None
    ids = [str(i) for i in ids_stations]
    sous = df_etp[df_etp["NUM_POSTE"].astype(str).isin(ids)].copy()
    if sous.empty:
        return None
    sous["NUM_POSTE"] = sous["NUM_POSTE"].astype(str)
    sous["etp"] = _serie_logique(sous, "etp_officielle_mm")
    sous["date_dt"] = _parser_dates(sous)
    sous = sous.dropna(subset=["date_dt", "etp"])
    d1 = pd.Timestamp(date_debut).normalize()
    d2 = pd.Timestamp(date_fin).normalize()
    sous = sous[(sous["date_dt"] >= d1) & (sous["date_dt"] <= d2)]
    if sous.empty:
        return None

    w = sous["NUM_POSTE"].map(poids).fillna(1.0) if poids else pd.Series(1.0, index=sous.index)
    tmp = pd.DataFrame({"date_dt": sous["date_dt"], "num": sous["etp"] * w, "den": w,
                        "NUM_POSTE": sous["NUM_POSTE"]})
    s = tmp.groupby("date_dt").agg(num=("num", "sum"), den=("den", "sum"),
                                   n_stations=("NUM_POSTE", "nunique"))
    res = pd.DataFrame({"etp_mm": s["num"] / s["den"], "n_stations": s["n_stations"]})
    return res.reset_index().sort_values("date_dt").reset_index(drop=True)


def calculer_etp_periode(df_etp, ids_stations, date_debut, date_fin, poids=None):
    """
    ETP observée par mois sur la période : annee, mois, etp_observee_mm,
    jours_valides. Les mois en bord de période sont partiels (cumul brut).
    """
    jour = recuperer_etp_journaliere(df_etp, ids_stations, date_debut, date_fin, poids)
    if jour is None or jour.empty:
        return None
    jour["annee"] = jour["date_dt"].dt.year
    jour["mois"] = jour["date_dt"].dt.month
    res = jour.groupby(["annee", "mois"]).agg(
        etp_observee_mm=("etp_mm", "sum"), jours_valides=("etp_mm", "count")
    ).reset_index()
    return res


# ==============================================================================
# INTENSITE DES PRECIPITATIONS
# ==============================================================================

def calculer_intensite_pluie(df_agg):
    """
    Indicateurs descriptifs sur les précipitations horaires (rr1_mm, mm/h).
    Retourne None si indisponible. Valeurs issues de la moyenne des stations :
    les pointes locales sont lissées.
    """
    if not colonne_presente(df_agg, "rr1_mm"):
        return None
    d = df_agg[["date_dt", "rr1_mm"]].dropna()
    d = d.assign(rr1_mm=d["rr1_mm"].clip(lower=0))
    if d.empty:
        return None

    pluie = d[d["rr1_mm"] >= 0.1]
    i = d["rr1_mm"].idxmax()
    s1, s2, s3 = SEUILS_INTENSITE_PLUIE
    bornes = [(0.1, s1), (s1, s2), (s2, s3), (s3, np.inf)]
    heures = [int(((pluie["rr1_mm"] >= a) & (pluie["rr1_mm"] < b)).sum()) for a, b in bornes]
    return {
        "intensite_max_mmh": float(d.loc[i, "rr1_mm"]),
        "date_max": d.loc[i, "date_dt"],
        "total_heures_pluie": int(len(pluie)),
        "repartition": pd.DataFrame({"classe": CLASSES_PLUIE, "heures": heures}),
    }


# ==============================================================================
# DEBUG
# ==============================================================================

def debug_colonnes(df_brut):
    src = df_brut["_fichier_source"].iloc[0] if "_fichier_source" in df_brut.columns else "?"
    st.write(f"Fichier : `{src}` — {len(df_brut):,} lignes")
    st.write(f"Colonnes ({len(df_brut.columns)}) :", list(df_brut.columns))
    st.dataframe(df_brut.head(3), width="stretch")
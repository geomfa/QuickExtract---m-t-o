"""
Client SYNOP (Opendatasoft) — utilisé pour toutes les analyses portant sur
des périodes de plus de 30 jours, où le fichier horaire départemental
Météo-France ("latest", limité à ~1-2 ans glissants) ne suffit pas.

Le réseau SYNOP est plus clairsemé que le réseau climatologique complet
(quelques dizaines de stations en France contre plusieurs milliers), mais
son historique horaire remonte à plusieurs décennies sur Opendatasoft,
ce qui permet des analyses sur 1 an et plus sans les limites du fichier
"latest" départemental.

Sortie normalisée pour rester compatible avec les fonctions génériques de
mf_client (agreger_multi_stations, colonne_presente...) : mêmes noms de
colonnes logiques (t_celsius, rr1_mm, ff_ms, dd_deg, u_pct), mêmes colonnes
NUM_POSTE / distance_km / date_dt.
"""

import math
import time
import requests
import pandas as pd
import streamlit as st

SYNOP_BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1"
    "/catalog/datasets/donnees-synop-essentielles-omm/records"
)
CHAMPS_SYNOP = "date,numer_sta,nom,t,u,ff,dd,rr1,rr3,pmer,n,vv"


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


@st.cache_data(ttl=3600, show_spinner=False)
def stations_synop_proches(lat, lon, n=3):
    """
    Retourne les n stations SYNOP les plus proches d'un point.
    Élargit progressivement le rayon de recherche jusqu'à trouver assez
    de stations (le réseau SYNOP est clairsemé).
    """
    seen = {}
    for rayon_km in [50, 100, 200, 400, 600]:
        where = f"distance(coordonnees, GEOM'POINT({lon} {lat})', {rayon_km}km)"
        params = {
            "select": "numer_sta,nom,coordonnees",
            "where": where,
            "order_by": "date DESC",
            "limit": 100,
        }
        try:
            resp = requests.get(SYNOP_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            records = resp.json().get("results", [])
        except Exception:
            records = []

        for r in records:
            sid = r.get("numer_sta")
            if not sid or sid in seen:
                continue
            coords = r.get("coordonnees")
            if not coords:
                continue
            if isinstance(coords, dict):
                slat, slon = coords.get("lat"), coords.get("lon")
            elif isinstance(coords, list) and len(coords) >= 2:
                slat, slon = coords[0], coords[1]
            else:
                continue
            if slat is None or slon is None:
                continue
            seen[sid] = {
                "NUM_POSTE":   sid,
                "NOM_USUEL":   r.get("nom", ""),
                "LAT":         slat,
                "LON":         slon,
                "distance_km": round(_haversine_km(lat, lon, slat, slon), 1),
            }
        if len(seen) >= n:
            break

    if not seen:
        return pd.DataFrame()
    return (
        pd.DataFrame(seen.values())
        .sort_values("distance_km")
        .head(n)
        .reset_index(drop=True)
    )


def _fetch_tranche(where):
    """Pagine une requête ODS jusqu'à épuisement, borné à offset < 10000."""
    all_records = []
    offset, limit = 0, 100
    while True:
        params = {
            "select": CHAMPS_SYNOP, "where": where,
            "order_by": "date ASC", "limit": limit, "offset": offset,
        }
        try:
            resp = requests.get(SYNOP_BASE_URL, params=params, timeout=60)
            resp.raise_for_status()
            records = resp.json().get("results", [])
        except Exception:
            break
        if not records:
            break
        all_records.extend(records)
        offset += limit
        if len(records) < limit or offset >= 9900:
            break
        time.sleep(0.15)
    return all_records


def _fetch_station_periode(numer_sta, date_debut, date_fin):
    """
    Télécharge une station sur une période, découpée mois par mois pour
    rester sous la limite offset=10000 de l'API Opendatasoft.
    """
    all_records = []
    curseur = date_debut.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while curseur <= date_fin:
        if curseur.month == 12:
            fin_mois = curseur.replace(year=curseur.year + 1, month=1, day=1)
        else:
            fin_mois = curseur.replace(month=curseur.month + 1, day=1)
        fin_mois = min(fin_mois - pd.Timedelta(seconds=1), date_fin)

        d1 = curseur.strftime("%Y-%m-%dT%H:%M:%SZ")
        d2 = fin_mois.strftime("%Y-%m-%dT%H:%M:%SZ")
        where = f"numer_sta='{numer_sta}' AND date >= '{d1}' AND date <= '{d2}'"
        all_records.extend(_fetch_tranche(where))

        if curseur.month == 12:
            curseur = curseur.replace(year=curseur.year + 1, month=1)
        else:
            curseur = curseur.replace(month=curseur.month + 1)
        time.sleep(0.1)
    return all_records


@st.cache_data(ttl=3600, show_spinner=False, max_entries=5)
def telecharger_synop_periode(ids_stations, date_debut, date_fin):
    """
    Télécharge et normalise les données horaires SYNOP pour une liste de
    stations sur une période. Colonnes de sortie compatibles avec les
    fonctions génériques de mf_client (NUM_POSTE, date_dt, t_celsius,
    rr1_mm, ff_ms, dd_deg, u_pct).

    Args:
        ids_stations : tuple de numer_sta (tuple plutôt que liste pour
                        être hashable et permettre le cache Streamlit)
        date_debut, date_fin : datetime

    Returns:
        DataFrame (vide si aucune donnée trouvée)
    """
    dfs = []
    for sid in ids_stations:
        records = _fetch_station_periode(sid, date_debut, date_fin)
        if records:
            df = pd.DataFrame(records)
            df["NUM_POSTE"] = sid
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["date_dt"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["date_dt"] = df["date_dt"].dt.tz_localize(None)
    df = df.dropna(subset=["date_dt"])

    if "t" in df.columns:
        df["t_celsius"] = pd.to_numeric(df["t"], errors="coerce") - 273.15
    if "u" in df.columns:
        df["u_pct"] = pd.to_numeric(df["u"], errors="coerce")
    if "ff" in df.columns:
        df["ff_ms"] = pd.to_numeric(df["ff"], errors="coerce")
    if "dd" in df.columns:
        df["dd_deg"] = pd.to_numeric(df["dd"], errors="coerce")
    # Les observations SYNOP sont tri-horaires : le cumul de la période est la
    # somme de rr3 (précipitations des 3 dernières heures). Sommer rr1 sous-estimerait
    # le total. La colonne garde le nom rr1_mm pour rester compatible avec le reste
    # de l'app, mais la valeur est un cumul par pas de temps (3 h). Valeur -0,1 = traces.
    rr3 = pd.to_numeric(df["rr3"], errors="coerce") if "rr3" in df.columns else None
    rr1 = pd.to_numeric(df["rr1"], errors="coerce") if "rr1" in df.columns else None
    if rr3 is not None and rr1 is not None:
        rr = rr3.where(rr3.notna(), rr1)
    else:
        rr = rr3 if rr3 is not None else rr1
    if rr is not None:
        df["rr1_mm"] = rr.clip(lower=0)

    return df
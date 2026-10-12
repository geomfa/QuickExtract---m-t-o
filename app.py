"""
QuickExtract Météo
Source : Météo-France (open data, aucun compte requis).

Flux :
  1. La zone d'étude est définie (code INSEE, carte ou coordonnées) puis
     "1 - Chercher les stations" télécharge les fichiers du département et
     liste les stations les plus proches.
  2. On coche les calculs voulus (vent, T° / précipitations, ETP) puis, dans le
     tableau, les stations à utiliser. Tous les calculs reposent sur les mêmes
     stations ; une station sans une variable ne contribue simplement pas à ce calcul.
  3. "2 - Lancer l'analyse" produit graphiques et exports.

Les observations viennent du fichier horaire départemental. Le réseau SYNOP
(Opendatasoft) n'est utilisé que si la période commence avant le début de ce
fichier : dans ce cas les stations SYNOP les plus proches remplacent les
stations cochées pour les observations, et les stations cochées servent aux
normales et à l'ETP.
"""

import io
import requests as _req
from datetime import datetime, timedelta, time, timezone

import numpy as np
import pandas as pd
import folium
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

from mf_client import (
    DonneesClimatoError,
    COLS_METEO,
    COLS_VENT,
    commune_depuis_insee,
    telecharger_horaire_departement,
    telecharger_quotidien_previous,
    telecharger_quotidien_etp,
    telecharger_quotidien_etp_latest,
    combiner_etp,
    plage_fichier_horaire,
    inspecter_stations,
    filtrer_periode,
    normaliser_variables,
    agreger_multi_stations,
    agreger_journalier,
    agreger_mensuel,
    agreger_annuel,
    calculer_normales,
    calculer_etp_officielle_mensuelle,
    calculer_etp_periode,
    recuperer_etp_journaliere,
    cumul_normale_jours,
    calculer_intensite_pluie,
    colonne_presente,
    debug_colonnes,
)
from synop_client import (
    stations_synop_proches,
    telecharger_synop_periode,
)
from graphiques import (
    graphique_temp_precip,
    graphique_rose_vents,
    graphique_thermopluviogramme,
    graphique_etp,
    graphique_histogramme_periode,
    graphique_temperatures_annuelles,
    graphique_etp_journalier,
    graphique_bilan_annuel_etp,
    graphique_intensite_pluie,
)

# ==============================================================================
# CONFIG
# ==============================================================================

st.set_page_config(page_title="QuickExtract Météo", layout="wide")
SOURCE_LABEL = "Source : Météo-France — Données climatologiques de base (data.gouv.fr)"

SEUIL_PERIODE_LONGUE = 31   # jours : au-delà, graphiques de type climatologie mensuelle
N_STATIONS_SYNOP = 3
N_STATIONS_DEFAUT = 3       # stations cochées par défaut (les plus proches)

COL_UTIL = "Utiliser"

# ==============================================================================
# SESSION STATE
# ==============================================================================

_DEFAULTS = {
    "carte_lat": None,
    "carte_lon": None,
    "clic_ignore": None,       # dernier clic écarté après "Réinitialiser le point"
    "zone": None,              # zone d'étude figée au clic sur le bouton 1
    "code_dept_actif": None,
    "df_brut": None,
    "df_quot": None,
    "df_etp": None,            # ETP historique + récente, déjà combinée
    "df_inspect": None,
    "n_recherche": 0,          # change à chaque recherche : réinitialise le tableau
    "demande_analyse": None,   # période demandée par le bouton 2
    "choix_actif": None,       # stations et variables figées au lancement
    "periode_active": None,
    "etape": 0,                # 0 accueil, 1 stations affichées, 2 résultats
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================================================================
# HELPERS
# ==============================================================================

@st.cache_data(ttl=86400, show_spinner=False, max_entries=200)
def _reverse_geocode(lat, lon):
    try:
        r = _req.get(
            "https://geo.api.gouv.fr/communes",
            params={"lat": lat, "lon": lon, "fields": "nom,codeDepartement",
                    "format": "json", "limit": 1},
            timeout=8,
        )
        if r.ok and r.json():
            d = r.json()[0]
            return d.get("nom", ""), d.get("codeDepartement", "")
    except Exception:
        pass
    return "", ""


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _afficher(fig, nom_fichier, label="Télécharger ce graphique"):
    st.pyplot(fig)
    st.download_button(label=label, data=_fig_png(fig), file_name=nom_fichier,
                       mime="image/png", key=f"dl_{nom_fichier}")
    plt.close(fig)


def _poids(df_sta):
    return {str(r["NUM_POSTE"]): 1.0 / max(float(r["distance_km"]), 0.1)
            for _, r in df_sta.iterrows()}


def _defaut(flags, n=N_STATIONS_DEFAUT):
    """Coche les n premières stations (déjà triées par distance) où flags est vrai."""
    flags = flags.astype(bool)
    return flags & (flags.cumsum() <= n)


def _oui_non(serie):
    return serie.map({True: "oui", False: "non"})


# ==============================================================================
# SIDEBAR : ZONE D'ETUDE
# ==============================================================================

st.sidebar.title("Configuration")
st.sidebar.caption("Données ouvertes — aucun compte requis")
st.sidebar.subheader("Zone d'étude")

mode_saisie = st.sidebar.radio(
    "Mode de saisie", ["Code INSEE", "Carte", "Coordonnées manuelles"], horizontal=True
)

lat_centre = lon_centre = code_dept = nom_commune = None

if mode_saisie == "Code INSEE":
    code_insee = st.sidebar.text_input("Code INSEE commune", value="", max_chars=5,
                                       placeholder="ex. 29232")
    if code_insee.strip():
        try:
            nom_commune, lat_centre, lon_centre, code_dept = commune_depuis_insee(code_insee)
            st.sidebar.caption(
                f"{nom_commune} — dept. {code_dept}\n"
                f"lat {lat_centre:.4f} / lon {lon_centre:.4f}"
            )
        except DonneesClimatoError as e:
            st.sidebar.error(str(e))

elif mode_saisie == "Carte":
    if st.session_state["carte_lat"] is not None:
        lat_centre = st.session_state["carte_lat"]
        lon_centre = st.session_state["carte_lon"]
        nom_commune, code_dept = _reverse_geocode(lat_centre, lon_centre)
        st.sidebar.caption(
            f"lat {lat_centre:.4f} / lon {lon_centre:.4f}"
            + (f"\n{nom_commune} — dept. {code_dept}" if nom_commune else "")
        )
        if not code_dept:
            code_dept = st.sidebar.text_input(
                "Code département (non trouvé automatiquement)", value="",
                placeholder="ex. 29", key="dept_carte",
            )
        if st.sidebar.button("Réinitialiser le point"):
            st.session_state["clic_ignore"] = (lat_centre, lon_centre)
            st.session_state["carte_lat"] = None
            st.session_state["carte_lon"] = None
            st.rerun()
    else:
        st.sidebar.info("Cliquez sur la carte pour choisir un point.")

else:
    lat_centre = st.sidebar.number_input("Latitude", value=None, format="%.4f",
                                         placeholder="ex. 48.3901")
    lon_centre = st.sidebar.number_input("Longitude", value=None, format="%.4f",
                                         placeholder="ex. -4.4860")
    code_dept = st.sidebar.text_input("Code département", value="", placeholder="ex. 29")
    if lat_centre is not None and lon_centre is not None:
        nom_commune, dept_auto = _reverse_geocode(lat_centre, lon_centre)
        if not code_dept.strip():
            code_dept = dept_auto
        if nom_commune:
            st.sidebar.caption(f"Commune la plus proche : {nom_commune} — dept. {code_dept}")

st.sidebar.divider()

# ==============================================================================
# SIDEBAR : BOUTON 1, CHERCHER LES STATIONS
# ==============================================================================

zone_definie = (lat_centre is not None and lon_centre is not None
                and bool(str(code_dept or "").strip()))

if st.sidebar.button("1 — Chercher les stations", type="primary",
                     width="stretch", disabled=not zone_definie):
    for cle in ("df_brut", "df_quot", "df_etp", "df_inspect", "demande_analyse",
                "choix_actif", "periode_active"):
        st.session_state[cle] = None
    st.session_state["n_recherche"] += 1
    st.session_state["code_dept_actif"] = str(code_dept).strip()
    st.session_state["zone"] = {
        "lat": float(lat_centre), "lon": float(lon_centre),
        "nom": nom_commune or "", "dept": str(code_dept).strip(),
    }
    st.session_state["etape"] = 1

    with st.spinner(f"Téléchargement des données — département {code_dept}..."):
        try:
            st.session_state["df_brut"] = telecharger_horaire_departement(code_dept)
        except DonneesClimatoError as e:
            st.session_state["etape"] = 0
            st.sidebar.error(str(e))
            st.stop()
    with st.spinner("Chargement des données historiques et de l'ETP..."):
        st.session_state["df_quot"] = telecharger_quotidien_previous(code_dept)
        st.session_state["df_etp"] = combiner_etp(
            telecharger_quotidien_etp(code_dept),
            telecharger_quotidien_etp_latest(code_dept),
        )
    with st.spinner("Inspection des stations..."):
        st.session_state["df_inspect"] = inspecter_stations(
            st.session_state["df_brut"], st.session_state["df_quot"],
            st.session_state["df_etp"], float(lat_centre), float(lon_centre), n=10,
        )
    st.rerun()

if not zone_definie:
    st.sidebar.caption("Renseignez la zone d'étude pour activer la recherche.")

# ==============================================================================
# SIDEBAR : PERIODE ET BOUTON 2
# ==============================================================================

if st.session_state["etape"] >= 1 and st.session_state["df_inspect"] is not None:
    st.sidebar.divider()
    st.sidebar.subheader("Fenêtre temporelle")
    fenetre = st.sidebar.radio(
        "Période",
        ["24 dernières heures", "7 derniers jours", "15 derniers jours", "1 an",
         "Personnalisée"],
        key="radio_fenetre",
    )

    now = _utcnow().replace(minute=0, second=0, microsecond=0)
    _dmin, _dmax = plage_fichier_horaire(st.session_state["df_brut"])
    # Les fenêtres glissantes s'arrêtent à la dernière mesure disponible (J-1 / J-2)
    ancre = min(now, _dmax) if _dmax is not None else now
    if fenetre == "24 dernières heures":
        date_debut, date_fin = ancre - timedelta(hours=24), ancre
    elif fenetre == "7 derniers jours":
        date_debut, date_fin = ancre - timedelta(days=7), ancre
    elif fenetre == "15 derniers jours":
        date_debut, date_fin = ancre - timedelta(days=15), ancre
    elif fenetre == "1 an":
        date_debut, date_fin = ancre - timedelta(days=365), ancre
    else:
        c1, c2 = st.sidebar.columns(2)
        d1 = c1.date_input("Du", value=(now - timedelta(days=30)).date(), key="d1")
        d2 = c2.date_input("Au", value=now.date(), max_value=now.date(), key="d2")
        h1 = st.sidebar.slider("Heure début (UTC)", 0, 23, 0, format="%dh", key="h1")
        h2 = st.sidebar.slider("Heure fin (UTC)", 0, 23, 23, format="%dh", key="h2")
        date_debut = datetime.combine(d1, time(h1, 0))
        date_fin = datetime.combine(d2, time(h2, 59))

    st.sidebar.caption(
        f"Du {date_debut.strftime('%d/%m/%Y %Hh')} au {date_fin.strftime('%d/%m/%Y %Hh')} (UTC)"
    )
    if _dmax is not None:
        st.sidebar.caption(
            f"Fichier horaire : du {_dmin.strftime('%d/%m/%Y')} au {_dmax.strftime('%d/%m/%Y %Hh')}. "
            "Les fenêtres glissantes se terminent à la dernière mesure."
        )
        if date_debut < _dmin:
            st.sidebar.caption("La période commence avant le fichier horaire : "
                               "observations SYNOP.")

    if date_fin <= date_debut:
        st.sidebar.warning("La date de fin doit suivre la date de début.")
    elif st.sidebar.button("2 — Lancer l'analyse", type="primary", width="stretch"):
        st.session_state["demande_analyse"] = {
            "fenetre": fenetre, "date_debut": date_debut, "date_fin": date_fin,
        }

with st.sidebar.expander("Mode debug"):
    debug_on = st.checkbox("Structure du fichier horaire")
    debug_etp = st.checkbox("Structure du fichier ETP")

# ==============================================================================
# CORPS : TITRE ET CARTE
# ==============================================================================

st.title("QuickExtract Météo")
st.caption(SOURCE_LABEL)

centre = [lat_centre, lon_centre] if lat_centre is not None else [46.5, 2.5]
zoom = 10 if lat_centre is not None else 5
m = folium.Map(location=centre, zoom_start=zoom, tiles="OpenStreetMap")
if lat_centre is not None and lon_centre is not None:
    folium.Marker(location=[lat_centre, lon_centre], popup=nom_commune or "Point d'étude",
                  icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

carte_result = st_folium(m, height=280, use_container_width=True,
                         returned_objects=["last_clicked"], key="carte_principale")

if mode_saisie == "Carte" and carte_result and carte_result.get("last_clicked"):
    clic = carte_result["last_clicked"]
    new_lat, new_lon = round(clic["lat"], 6), round(clic["lng"], 6)
    if (new_lat, new_lon) != st.session_state["clic_ignore"] \
            and new_lat != st.session_state["carte_lat"]:
        st.session_state["carte_lat"] = new_lat
        st.session_state["carte_lon"] = new_lon
        st.session_state["clic_ignore"] = None
        st.rerun()

if st.session_state["etape"] == 0 or st.session_state["df_brut"] is None:
    st.info("Renseignez la zone d'étude dans la barre latérale puis cliquez sur "
            "'1 — Chercher les stations'.")
    st.stop()

df_brut = st.session_state["df_brut"]
df_quot = st.session_state["df_quot"]
df_etp = st.session_state["df_etp"]
df_inspect = st.session_state["df_inspect"]
zone = st.session_state["zone"]

if df_inspect is None or df_inspect.empty:
    st.error("Aucune station trouvée pour ce département.")
    st.stop()

if debug_on:
    with st.expander("Structure du fichier horaire", expanded=True):
        debug_colonnes(df_brut)
if debug_etp:
    with st.expander("Structure du fichier ETP", expanded=True):
        if df_etp is None:
            st.warning("Aucun fichier ETP téléchargé (indisponible).")
        else:
            debug_colonnes(df_etp)

st.caption(
    f"Fichier horaire : {df_brut['_fichier_source'].iloc[0]} | "
    + (f"Historique : {df_quot['_fichier_source'].iloc[0]}"
       if df_quot is not None else "Historique : indisponible")
    + " | "
    + ("ETP : disponible" if df_etp is not None else "ETP : indisponible")
)

# ==============================================================================
# TABLEAU DES STATIONS AVEC CASES A COCHER
# ==============================================================================

st.subheader("Stations disponibles")
st.caption(
    "Choisissez les calculs, puis cochez les stations à utiliser. Tous les calculs "
    "reposent sur les mêmes stations : une station qui n'a pas une variable (vent, ETP...) "
    "ne contribue simplement pas à ce calcul. Les colonnes « dispo » aident à choisir."
)

k1, k2, k3 = st.columns(3)
veut_vent = k1.checkbox("Vent (rose des vents)", value=True, key="calc_vent")
veut_meteo = k2.checkbox("T° et précipitations", value=True, key="calc_meteo")
veut_etp = k3.checkbox("ETP", value=True, key="calc_etp")

di = df_inspect.reset_index(drop=True)
di["NUM_POSTE"] = di["NUM_POSTE"].astype(str)
if "NOM_USUEL" not in di.columns:
    di["NOM_USUEL"] = di["NUM_POSTE"]

base = pd.DataFrame({
    COL_UTIL: np.arange(len(di)) < N_STATIONS_DEFAUT,
    "Code": di["NUM_POSTE"],
    "Nom": di["NOM_USUEL"],
    "Dist. (km)": di["distance_km"],
    "Alt. (m)": di["ALTI"] if "ALTI" in di.columns else np.nan,
    "Dernière mesure": di["derniere_date"],
    "Fraîcheur (j)": di["fraicheur_jours"],
    "Nébulosité": _oui_non(di["nebulo_dispo"]),
    "Historique": _oui_non(di["historique_dispo"]),
    "Vent dispo": _oui_non(di["vent_dispo"]),
    "T°/précip. dispo": _oui_non(di["meteo_dispo"]),
    "ETP dispo": _oui_non(di["etp_dispo"]),
})
cols_info = [c for c in base.columns if c != COL_UTIL]

df_choix = st.data_editor(
    base,
    key=f"editeur_stations_{st.session_state['n_recherche']}",
    hide_index=True,
    width="stretch",
    disabled=cols_info,
    column_config={COL_UTIL: st.column_config.CheckboxColumn(COL_UTIL)},
)


def _extraire_choix(df_edit):
    """Stations utilisées ; pour chaque variable demandée, seules celles qui ont la donnée."""
    cols = [c for c in ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "distance_km"]
            if c in di.columns]
    utilise = df_edit[COL_UTIL].astype(bool).values
    sel = di.loc[utilise, cols].copy()
    sel["vent"] = di.loc[utilise, "vent_dispo"].astype(bool).values & veut_vent
    sel["meteo"] = di.loc[utilise, "meteo_dispo"].astype(bool).values & veut_meteo
    sel["etp"] = di.loc[utilise, "etp_dispo"].astype(bool).values & veut_etp
    sel = sel.reset_index(drop=True)
    manques = []
    if len(sel):
        for lib, veut, cle in (("Vent", veut_vent, "vent"), ("T° et précipitations", veut_meteo, "meteo"),
                               ("ETP", veut_etp, "etp")):
            if veut and not sel[cle].any():
                manques.append(lib)
    return sel, manques


choix_courant, manques = _extraire_choix(df_choix)
if manques:
    st.caption("Données non disponibles pour les stations cochées : " + ", ".join(manques) + ".")

# Instantané au clic sur le bouton 2 : les résultats restent figés tant qu'on ne relance pas
demande = st.session_state["demande_analyse"]
if demande is not None:
    st.session_state["demande_analyse"] = None
    if choix_courant.empty or not (veut_vent or veut_meteo or veut_etp):
        st.warning("Cochez au moins une station et au moins un calcul.")
    else:
        st.session_state["choix_actif"] = choix_courant.copy()
        st.session_state["periode_active"] = demande
        st.session_state["etape"] = 2

choix = st.session_state["choix_actif"]
if st.session_state["etape"] < 2 or st.session_state["periode_active"] is None or choix is None:
    st.info("Cochez les calculs et les stations, choisissez la période dans la barre "
            "latérale puis cliquez sur '2 — Lancer l'analyse'.")
    st.stop()

if not choix_courant.reset_index(drop=True).equals(choix.reset_index(drop=True)):
    st.info("La sélection a changé depuis la dernière analyse. "
            "Cliquez sur '2 — Lancer l'analyse' pour la prendre en compte.")

# ==============================================================================
# ETAPE 2 : TRAITEMENT
# ==============================================================================

periode = st.session_state["periode_active"]
fenetre = periode["fenetre"]
date_debut = periode["date_debut"]
date_fin = periode["date_fin"]

duree_jours = (date_fin - date_debut).days
duree_ans = duree_jours / 365.25
is_long_period = duree_jours > SEUIL_PERIODE_LONGUE

sta_vent = choix[choix["vent"]]
sta_meteo = choix[choix["meteo"]]
sta_etp = choix[choix["etp"]]
ids_vent = sta_vent["NUM_POSTE"].astype(str).tolist()
ids_meteo = sta_meteo["NUM_POSTE"].astype(str).tolist()
ids_etp = sta_etp["NUM_POSTE"].astype(str).tolist()
poids_meteo = _poids(sta_meteo) if len(sta_meteo) else None
poids_etp = _poids(sta_etp) if len(sta_etp) else None

titre_base = zone["nom"] or f"dept. {zone['dept']}"
prefixe_nom = titre_base.replace(" ", "_")
periode_nom = f"{date_debut.strftime('%Y%m%d%H')}_{date_fin.strftime('%Y%m%d%H')}"

# --- Observations ------------------------------------------------------------
dmin_fichier, _ = plage_fichier_horaire(df_brut)
has_obs = bool(ids_vent or ids_meteo)
use_synop = bool(has_obs and dmin_fichier is not None and date_debut < dmin_fichier)
source_donnees_label = "Météo-France — fichier horaire départemental"
pas_heures = 1

df_obs = pd.DataFrame()
df_agg = pd.DataFrame()
df_stations_synop = pd.DataFrame()

if has_obs and use_synop:
    pas_heures = 3
    source_donnees_label = "Météo-France — SYNOP (Opendatasoft)"
    with st.spinner("Recherche des stations SYNOP les plus proches..."):
        df_stations_synop = stations_synop_proches(zone["lat"], zone["lon"], n=N_STATIONS_SYNOP)
    if df_stations_synop.empty:
        st.error("Aucune station SYNOP trouvée à proximité.")
    else:
        with st.spinner(f"Téléchargement SYNOP ({duree_jours} jours, "
                        f"{len(df_stations_synop)} station(s))..."):
            df_obs = telecharger_synop_periode(
                tuple(df_stations_synop["NUM_POSTE"].tolist()), date_debut, date_fin
            )
        if not df_obs.empty:
            df_obs = df_obs.merge(df_stations_synop[["NUM_POSTE", "distance_km"]],
                                  on="NUM_POSTE", how="left")
            cols = (COLS_METEO if ids_meteo else []) + (COLS_VENT if ids_vent else [])
            df_agg = agreger_multi_stations(df_obs, df_stations_synop, cols)
elif has_obs:
    ids_union = sorted(set(ids_vent) | set(ids_meteo))
    df_f = df_brut[df_brut["NUM_POSTE"].astype(str).isin(ids_union)]
    df_obs = filtrer_periode(df_f, date_debut, date_fin)
    if not df_obs.empty:
        df_obs = normaliser_variables(df_obs)
        df_obs["NUM_POSTE"] = df_obs["NUM_POSTE"].astype(str)
        dist = choix[["NUM_POSTE", "distance_km"]].copy()
        dist["NUM_POSTE"] = dist["NUM_POSTE"].astype(str)
        df_obs = df_obs.merge(dist, on="NUM_POSTE", how="left")
        parts = []
        if ids_meteo:
            p = agreger_multi_stations(df_obs[df_obs["NUM_POSTE"].isin(ids_meteo)],
                                       sta_meteo, COLS_METEO)
            parts.append(p.rename(columns={"n_stations": "n_stations_meteo"}))
        if ids_vent:
            p = agreger_multi_stations(df_obs[df_obs["NUM_POSTE"].isin(ids_vent)],
                                       sta_vent, COLS_VENT)
            parts.append(p.rename(columns={"n_stations": "n_stations_vent"}))
        parts = [p for p in parts if not p.empty]
        if parts:
            df_agg = parts[0]
            for p in parts[1:]:
                df_agg = df_agg.merge(p, on="date_dt", how="outer")
            df_agg = df_agg.sort_values("date_dt").reset_index(drop=True)

if has_obs and df_agg.empty:
    st.warning(
        "Aucune observation sur cette période pour les stations cochées. "
        "Le fichier horaire va de janvier de l'année précédente à J-1 / J-2 : "
        "essayez une fenêtre plus large."
    )
    has_obs = False

# --- Normales ----------------------------------------------------------------
df_normales = (calculer_normales(df_quot, ids_meteo, n_annees=10, poids=poids_meteo)
               if ids_meteo else None)
if df_normales is not None:
    _an_min = df_normales.attrs.get("annee_min", "")
    _an_max = df_normales.attrs.get("annee_max", "")
    label_normale = f"Normale {_an_min}-{_an_max}" if _an_min != "" else "Normale (réf.)"
else:
    label_normale = "Normale (réf.)"

# --- ETP ---------------------------------------------------------------------
df_etp_normale = None
etp_jour = None
etp_periode_mens = None
if ids_etp:
    _norm = calculer_etp_officielle_mensuelle(df_etp, ids_etp, n_annees=10, poids=poids_etp)
    if _norm is not None:
        df_etp_normale = _norm.rename(columns={"etp_officielle_mm": "etp_normale_mm"})
    etp_jour = recuperer_etp_journaliere(df_etp, ids_etp, date_debut, date_fin, poids=poids_etp)
    etp_periode_mens = calculer_etp_periode(df_etp, ids_etp, date_debut, date_fin, poids=poids_etp)

# --- Agrégats ----------------------------------------------------------------
agg_j = agreger_journalier(df_agg, pas_heures) if has_obs else pd.DataFrame()
agg_m = agreger_mensuel(agg_j, date_debut, date_fin) if not agg_j.empty else pd.DataFrame()
agg_a = agreger_annuel(agg_j, date_debut, date_fin) if not agg_j.empty else pd.DataFrame()

# --- En-tête des résultats ---------------------------------------------------
st.subheader(f"Résultats — {titre_base}")
cap = f"{date_debut.strftime('%d/%m/%Y %Hh')} → {date_fin.strftime('%d/%m/%Y %Hh')} UTC"
if has_obs:
    cap = f"Observations : {source_donnees_label} | " + cap + f" | {len(df_agg)} pas de temps"
if df_normales is not None:
    cap += (f" | Normales : {df_normales.attrs.get('annee_min', '')}"
            f"–{df_normales.attrs.get('annee_max', '')}")
st.caption(cap)

if use_synop:
    st.info(
        "La période commence avant le début du fichier horaire départemental : les "
        "observations viennent des stations SYNOP les plus proches. Les stations cochées "
        "servent aux normales et à l'ETP ; la colonne Vent n'est pas utilisée dans ce cas."
    )


def _usage(r):
    u = []
    if r["meteo"]:
        u.append("T° / précip. : normales" + ("" if use_synop else " et observations"))
    if r["vent"]:
        u.append("vent : non utilisé (SYNOP)" if use_synop else "vent : observations")
    if r["etp"]:
        u.append("ETP")
    return " ; ".join(u) if u else "aucune donnée pour les calculs choisis"


df_recap = pd.DataFrame({
    "Code": choix["NUM_POSTE"].values,
    "Nom": choix["NOM_USUEL"].values,
    "Dist. (km)": choix["distance_km"].values,
    "Vent": _oui_non(choix["vent"]).values,
    "T° / précip.": _oui_non(choix["meteo"]).values,
    "ETP": _oui_non(choix["etp"]).values,
    "Utilisation": choix.apply(_usage, axis=1).values,
    "Réseau d'observation": "SYNOP" if use_synop else "Fichier horaire départemental",
})
with st.expander("Stations utilisées pour cette analyse", expanded=False):
    st.dataframe(df_recap, hide_index=True, width="stretch")
    if use_synop and not df_stations_synop.empty:
        st.caption("Stations SYNOP utilisées pour les observations :")
        st.dataframe(df_stations_synop[["NUM_POSTE", "NOM_USUEL", "distance_km"]],
                     hide_index=True, width="stretch")
    hist = di.set_index("NUM_POSTE")["historique_dispo"]
    sans_hist = [str(n) for c, n in zip(sta_meteo["NUM_POSTE"], sta_meteo["NOM_USUEL"])
                 if not bool(hist.get(str(c), False))]
    if sans_hist:
        st.caption("Sans historique (exclues des normales) : " + ", ".join(sans_hist))

# Indicateurs clés
if has_obs:
    c1, c2, c3, c4 = st.columns(4)
    if colonne_presente(df_agg, "t_celsius"):
        c1.metric("T° moyenne", f"{df_agg['t_celsius'].mean():.1f} °C")
        c2.metric("T° min / max",
                  f"{df_agg['t_celsius'].min():.1f} / {df_agg['t_celsius'].max():.1f} °C")
    if colonne_presente(df_agg, "rr1_mm"):
        c3.metric("Précip. cumulées", f"{df_agg['rr1_mm'].sum():.1f} mm")
    if colonne_presente(df_agg, "ff_ms"):
        c4.metric("Vent moyen", f"{df_agg['ff_ms'].mean():.1f} m/s")

has_t = has_obs and colonne_presente(df_agg, "t_celsius")
has_p = has_obs and colonne_presente(df_agg, "rr1_mm")
has_vent = has_obs and colonne_presente(df_agg, "ff_ms") and colonne_presente(df_agg, "dd_deg")
df_thermopluvio_export = None

# ==============================================================================
# GRAPHIQUES
# ==============================================================================

# 1. Température / précipitations, vue horaire
if has_t and (fenetre in ("24 dernières heures", "7 derniers jours")
              or (fenetre == "Personnalisée" and duree_jours <= 7)):
    st.subheader("Température et précipitations")
    fig = graphique_temp_precip(
        df_agg, titre=f"Évolution température / précipitations — {titre_base}",
        df_normales=df_normales, label_normale=label_normale,
    )
    if fig:
        _afficher(fig, f"temp_precip_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — T° / Précipitations")

# 2. Rose des vents : période + dernier mois
if has_vent:
    st.subheader("Rose des vents")
    date_max_dispo = df_agg["date_dt"].max()
    date_debut_dm = date_max_dispo - pd.Timedelta(days=30)
    df_dernier_mois = df_agg[df_agg["date_dt"] >= date_debut_dm]

    col_a, col_b = st.columns(2)
    with col_a:
        fig_p = graphique_rose_vents(
            df_agg,
            titre=(f"Période sélectionnée\n({df_agg['date_dt'].min().strftime('%d/%m/%Y')} "
                   f"— {date_max_dispo.strftime('%d/%m/%Y')})"),
        )
        if fig_p:
            _afficher(fig_p, f"rose_vents_periode_{prefixe_nom}_{periode_nom}.png",
                      "Télécharger — Rose des vents (période)")
        else:
            st.caption("Données de vent insuffisantes sur la période.")
    with col_b:
        fig_m = graphique_rose_vents(
            df_dernier_mois,
            titre=(f"Dernier mois (comparaison)\n({date_debut_dm.strftime('%d/%m/%Y')} "
                   f"— {date_max_dispo.strftime('%d/%m/%Y')})"),
        )
        if fig_m:
            _afficher(fig_m, f"rose_vents_dernier_mois_{prefixe_nom}_{periode_nom}.png",
                      "Télécharger — Rose des vents (dernier mois)")
        else:
            st.caption("Données de vent insuffisantes sur le dernier mois.")
elif ids_vent and has_obs and not use_synop:
    st.subheader("Rose des vents")
    st.caption("Données de vent non disponibles pour la période.")

# 3. Suivi journalier : fenêtres courtes
if has_t and not agg_j.empty and (
        fenetre in ("7 derniers jours", "15 derniers jours")
        or (fenetre == "Personnalisée" and duree_jours <= SEUIL_PERIODE_LONGUE)):
    st.subheader("Précipitations et températures — suivi journalier")
    fig = graphique_histogramme_periode(
        agg_j, titre=f"Précipitations et températures — suivi journalier — {titre_base}",
        df_normales=df_normales, label_normale=label_normale,
    )
    if fig:
        _afficher(fig, f"suivi_journalier_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Suivi journalier")

# 4. Thermopluviogramme : mois complets de la période
if is_long_period and has_t and not agg_m.empty:
    st.subheader("Thermopluviogramme — bilan mensuel")
    ret = agg_m[agg_m["retenu"]]
    if ret.empty:
        st.caption("Aucun mois complet sur la période : bilan mensuel non calculé.")
    else:
        spec = dict(t_moy=("t_moy", "mean"), t_min=("t_min", "mean"), t_max=("t_max", "mean"))
        if "precip_tot" in ret.columns:
            spec["precip_tot"] = ("precip_tot", "mean")
        mensuel_norm = ret.groupby("mois").agg(**spec).reset_index()

        etp_series = None
        if df_etp_normale is not None:
            _m = dict(zip(df_etp_normale["mois"], df_etp_normale["etp_normale_mm"]))
            etp_series = [_m.get(mm, float("nan")) for mm in mensuel_norm["mois"]]

        st.caption(f"{len(ret)} mois complets retenus sur {len(agg_m)} mois touchés par la "
                   "période (les mois partiels en début et fin de période sont écartés).")
        fig = graphique_thermopluviogramme(
            mensuel_norm, titre=f"Thermopluviogramme — bilan mensuel — {titre_base}",
            df_normales=df_normales, label_normale=label_normale,
            etp_normale=etp_series, label_etp=f"ETP Penman-Monteith - {label_normale}",
        )
        if fig:
            _afficher(fig, f"thermopluvio_{prefixe_nom}_{periode_nom}.png",
                      "Télécharger — Thermopluviogramme")
        df_thermopluvio_export = mensuel_norm.copy()
        if etp_series is not None:
            df_thermopluvio_export["etp_normale_mm"] = etp_series

# 5. ETP mensuelle et bilan hydrique (normales)
if ids_etp:
    st.subheader("Évapotranspiration et bilan hydrique")
    if df_etp_normale is not None:
        fig = graphique_etp(
            df_etp_normale, df_normales=df_normales,
            titre=f"Évapotranspiration et bilan hydrique — {titre_base}",
            label_normale=label_normale,
        )
        if fig:
            _afficher(fig, f"etp_{prefixe_nom}_{periode_nom}.png",
                      "Télécharger — ETP / bilan hydrique")
    else:
        st.caption("ETP : données non disponibles pour les stations cochées.")

# 6. Températures annuelles : années civiles complètes
if has_t and not agg_a.empty:
    ret_a = agg_a[agg_a["retenu"]]
    if not ret_a.empty:
        st.subheader("Évolution des températures annuelles")
        fig = graphique_temperatures_annuelles(
            ret_a, titre=f"Évolution des températures annuelles — {titre_base}",
            df_normales=df_normales, label_normale=label_normale,
        )
        if fig:
            _afficher(fig, f"temp_annuelles_{prefixe_nom}_{periode_nom}.png",
                      "Télécharger — Températures annuelles")
    elif duree_ans >= 1.0:
        st.caption("Évolution annuelle non tracée : aucune année civile complète dans la période.")

# 7. ETP journalière : périodes courtes
if ids_etp and not is_long_period:
    st.subheader("Évapotranspiration journalière")
    if etp_jour is not None:
        fig = graphique_etp_journalier(
            etp_jour, df_etp_normale=df_etp_normale,
            titre=f"Évapotranspiration journalière — {titre_base}",
            label_normale=label_normale,
        )
        if fig:
            _afficher(fig, f"etp_journalier_{prefixe_nom}_{periode_nom}.png",
                      "Télécharger — ETP journalière")
    else:
        st.caption("ETP journalière : données non disponibles sur la période "
                   "(le fichier ETP a quelques jours de retard).")

# 8. Bilan sur la période : normale et observé sur les mêmes jours
if is_long_period and (ids_etp or has_p):
    jours_p = (set(agg_j.loc[agg_j["couverture_precip"] >= 0.9, "date_dt"])
               if has_p and "couverture_precip" in agg_j.columns else set())
    jours_e = set(etp_jour["date_dt"]) if etp_jour is not None else set()
    if jours_p and jours_e:
        jours = sorted(jours_p & jours_e)
    else:
        jours = sorted(jours_p or jours_e)

    if jours:
        jdx = pd.DatetimeIndex(jours)
        precip_obs = (agg_j.loc[agg_j["date_dt"].isin(jdx), "precip_tot"].sum()
                      if jours_p else None)
        etp_obs = (etp_jour.loc[etp_jour["date_dt"].isin(jdx), "etp_mm"].sum()
                   if etp_jour is not None else None)
        precip_norm = cumul_normale_jours(df_normales, "precip_norm", jdx) if has_p else None
        etp_norm = cumul_normale_jours(df_etp_normale, "etp_normale_mm", jdx)

        st.subheader("Bilan sur la période — normale et observé")
        st.caption(
            f"Comparaison sur {len(jdx)} jours renseignés "
            f"({jdx.min().strftime('%d/%m/%Y')} au {jdx.max().strftime('%d/%m/%Y')}), "
            "la normale étant cumulée sur ces mêmes jours."
        )
        fig = graphique_bilan_annuel_etp(
            etp_norm, etp_obs, precip_norm, precip_obs,
            titre=f"Bilan sur la période — {titre_base}",
            label_normale=label_normale, label_periode="Période observée",
        )
        if fig:
            _afficher(fig, f"bilan_periode_{prefixe_nom}_{periode_nom}.png",
                      "Télécharger — Bilan sur la période")

# 9. Intensité des précipitations (pas horaire, hors SYNOP)
if has_p and not use_synop:
    intensite = calculer_intensite_pluie(df_agg)
    if intensite is not None:
        st.subheader("Intensité des précipitations")
        st.caption("Calculée sur la moyenne horaire des stations cochées : les pointes "
                   "locales sont lissées. Seuils indicatifs (2,5 / 7,6 / 15 mm/h).")
        ci1, ci2 = st.columns(2)
        ci1.metric("Intensité horaire max.", f"{intensite['intensite_max_mmh']:.1f} mm/h",
                   help=f"Le {intensite['date_max'].strftime('%d/%m/%Y %Hh')}")
        ci2.metric("Heures de pluie", f"{intensite['total_heures_pluie']} h")
        fig = graphique_intensite_pluie(
            intensite["repartition"],
            titre=f"Répartition des intensités de pluie — {titre_base}",
        )
        if fig:
            _afficher(fig, f"intensite_pluie_{prefixe_nom}_{periode_nom}.png",
                      "Télécharger — Intensité des pluies")
elif has_p and use_synop:
    st.caption("Intensité des pluies non calculée : les observations SYNOP sont tri-horaires.")

# ==============================================================================
# EXPORT CARTOGRAPHIQUE (carte statique A5 paysage)
# ==============================================================================

st.subheader("Export cartographique")
with st.expander("Paramètres de la carte", expanded=False):
    titre_carte = st.text_input("Titre", value=f"Météo — {titre_base}", key="titre_carte")
    auteur_carte = st.text_input("Auteur", value="", key="auteur_carte")
    fichier_zone = st.file_uploader(
        "Zone d'étude (GeoJSON, facultatif). Sans fichier, le point d'étude est affiché.",
        type=["geojson", "json"], key="upload_zone")
    fichier_logo = st.file_uploader("Logo (PNG)", type=["png"], key="upload_logo")

    if st.button("Générer la carte", key="btn_carte"):
        with st.spinner("Génération de la carte..."):
            try:
                from carte_statique import generer_carte_statique, charger_zone_geojson
                zone_gj = charger_zone_geojson(fichier_zone) if fichier_zone else None
                if fichier_zone and zone_gj is None:
                    st.warning("GeoJSON illisible : le point d'étude est affiché à la place.")
                png, pdf, fond_ok = generer_carte_statique(
                    df_stations=choix[["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "distance_km"]],
                    point=(zone["lat"], zone["lon"], zone["nom"]),
                    zone_geojson=zone_gj,
                    df_synop=df_stations_synop if (use_synop and not df_stations_synop.empty) else None,
                    titre=titre_carte,
                    logo_bytes=fichier_logo.read() if fichier_logo else None,
                    auteur=auteur_carte,
                )
                if not fond_ok:
                    st.warning("Fond OpenStreetMap indisponible (réseau) : carte sans fond.")
                st.session_state["carte_png"] = png
                st.session_state["carte_pdf"] = pdf
            except Exception as e:
                st.error(f"Erreur lors de la génération de la carte : {e}")

    if st.session_state.get("carte_png"):
        st.image(st.session_state["carte_png"])
        cc1, cc2 = st.columns(2)
        cc1.download_button("Télécharger la carte (PNG)", data=st.session_state["carte_png"],
                            file_name=f"carte_{prefixe_nom}.png", mime="image/png", key="dl_carte_png")
        cc2.download_button("Télécharger la carte (PDF)", data=st.session_state["carte_pdf"],
                            file_name=f"carte_{prefixe_nom}.pdf", mime="application/pdf", key="dl_carte_pdf")

# ==============================================================================
# EXPORT EXCEL
# ==============================================================================

st.subheader("Export Excel")

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df_recap.to_excel(writer, sheet_name="Stations", index=False)
    if use_synop and not df_stations_synop.empty:
        df_stations_synop.to_excel(writer, sheet_name="Stations_SYNOP", index=False)
    if has_obs:
        df_agg.to_excel(writer, sheet_name="Horaire_agrege", index=False)
        df_obs.drop(columns=["_fichier_source"], errors="ignore").to_excel(
            writer, sheet_name="Horaire_brut", index=False)
    if not agg_j.empty:
        agg_j.drop(columns=["date_seule"], errors="ignore").to_excel(
            writer, sheet_name="Journalier", index=False)
    if not agg_m.empty:
        agg_m.to_excel(writer, sheet_name="Mensuel", index=False)
    if not agg_a.empty:
        agg_a.to_excel(writer, sheet_name="Annuel", index=False)
    if df_normales is not None:
        df_normales.to_excel(writer, sheet_name="Normales", index=False)
    if df_etp_normale is not None:
        df_etp_normale.to_excel(writer, sheet_name="ETP_normale", index=False)
    if etp_periode_mens is not None:
        etp_periode_mens.to_excel(writer, sheet_name="ETP_periode", index=False)
    if etp_jour is not None:
        etp_jour.to_excel(writer, sheet_name="ETP_journaliere", index=False)
    if df_thermopluvio_export is not None:
        df_thermopluvio_export.to_excel(writer, sheet_name="Thermopluvio_mensuel", index=False)

st.download_button(
    "Télécharger l'Excel", data=buffer.getvalue(),
    file_name=f"meteo_{prefixe_nom}_{periode_nom}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="dl_excel",
)
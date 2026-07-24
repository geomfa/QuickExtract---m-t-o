"""
QuickExtract Météo — Données horaires récentes
Source : Météo-France / data.gouv.fr (open data, aucun compte requis).
"""

import io
import requests as _req
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
import folium
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

from mf_client import (
    DonneesClimatoError,
    commune_depuis_insee,
    telecharger_horaire_departement,
    telecharger_quotidien_previous,
    inspecter_stations,
    filtrer_periode,
    normaliser_variables,
    agreger_multi_stations,
    calculer_normales,
    calculer_normales_vent,
    colonne_presente,
    debug_colonnes,
)
from graphiques import (
    graphique_temp_precip,
    graphique_rose_vents,
    graphique_thermopluviogramme,
    graphique_histogramme_periode,
    graphique_temperatures_annuelles,
)

# ==============================================================================
# CONFIG
# ==============================================================================

st.set_page_config(page_title="QuickExtract Météo", layout="wide")
SOURCE_LABEL = "Source : Météo-France — Données climatologiques de base (data.gouv.fr)"

# ==============================================================================
# SESSION STATE
# ==============================================================================

_DEFAULTS = {
    "carte_lat":       None,
    "carte_lon":       None,
    "code_dept_actif": None,   # département chargé (détecte un changement de zone)
    "df_brut":         None,
    "df_quot":         None,
    "df_inspect":      None,
    "selection":       None,
    "periode_active":  None,
    "etape":           0,      # 0=accueil 1=stations affichées 2=résultats affichés
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================================================================
# HELPERS
# ==============================================================================

def _reverse_geocode(lat, lon):
    try:
        r = _req.get(
            "https://geo.api.gouv.fr/communes",
            params={"lat": lat, "lon": lon,
                    "fields": "nom,codeDepartement",
                    "format": "json", "limit": 1},
            timeout=8,
        )
        if r.ok and r.json():
            d = r.json()[0]
            return d.get("nom", ""), d.get("codeDepartement", "")
    except Exception:
        pass
    return "", ""


def _fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _afficher(fig, nom_fichier, label="Télécharger ce graphique"):
    st.pyplot(fig)
    st.download_button(
        label=label, data=_fig_png(fig),
        file_name=nom_fichier, mime="image/png",
        key=f"dl_{nom_fichier}",
    )
    plt.close(fig)


# ==============================================================================
# SIDEBAR — ZONE D'ETUDE
# ==============================================================================

st.sidebar.title("Configuration")
st.sidebar.caption("Données ouvertes — aucun compte requis")
st.sidebar.subheader("Zone d'étude")

mode_saisie = st.sidebar.radio(
    "Mode de saisie",
    ["Code INSEE", "Carte", "Coordonnées manuelles"],
    horizontal=True,
)

lat_centre = lon_centre = code_dept = nom_commune = None

if mode_saisie == "Code INSEE":
    code_insee = st.sidebar.text_input(
        "Code INSEE commune", value="", max_chars=5,
        placeholder="ex. 29232",
    )
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
        lat_centre  = st.session_state["carte_lat"]
        lon_centre  = st.session_state["carte_lon"]
        nom_commune, code_dept = _reverse_geocode(lat_centre, lon_centre)
        st.sidebar.caption(
            f"lat {lat_centre:.4f} / lon {lon_centre:.4f}"
            + (f"\n{nom_commune} — dept. {code_dept}" if nom_commune else "")
        )
        if st.sidebar.button("Réinitialiser le point"):
            st.session_state["carte_lat"] = None
            st.session_state["carte_lon"] = None
            st.rerun()
    else:
        st.sidebar.info("Cliquez sur la carte pour choisir un point.")

else:
    lat_centre = st.sidebar.number_input("Latitude",  value=None, format="%.4f",
                                          placeholder="ex. 48.3901")
    lon_centre = st.sidebar.number_input("Longitude", value=None, format="%.4f",
                                          placeholder="ex. -4.4860")
    code_dept  = st.sidebar.text_input("Code département", value="",
                                        placeholder="ex. 29")
    if lat_centre and lon_centre:
        nom_commune, dept_auto = _reverse_geocode(lat_centre, lon_centre)
        if not code_dept.strip():
            code_dept = dept_auto
        if nom_commune:
            st.sidebar.caption(f"Commune la plus proche : {nom_commune} — dept. {code_dept}")

st.sidebar.divider()

# ==============================================================================
# SIDEBAR — BOUTON 1 : CHERCHER LES STATIONS
# ==============================================================================

zone_definie = bool(lat_centre and lon_centre and code_dept)

if st.sidebar.button(
    "1 — Chercher les stations",
    type="primary",
    use_container_width=True,
    disabled=not zone_definie,
):
    # Réinitialisation de l'état
    st.session_state["code_dept_actif"] = code_dept
    st.session_state["df_brut"]         = None
    st.session_state["df_quot"]         = None
    st.session_state["df_inspect"]      = None
    st.session_state["selection"]       = None
    st.session_state["periode_active"]  = None
    st.session_state["etape"]           = 1

    # Téléchargement immédiat pour que la sidebar étape 2 soit disponible
    # au prochain run (sinon df_inspect reste None et les widgets n'apparaissent pas)
    with st.spinner(f"Téléchargement des données — département {code_dept}..."):
        try:
            _df_brut = telecharger_horaire_departement(code_dept)
            st.session_state["df_brut"] = _df_brut
        except DonneesClimatoError as e:
            st.sidebar.error(str(e))
            st.stop()
    with st.spinner("Chargement des données historiques..."):
        st.session_state["df_quot"] = telecharger_quotidien_previous(code_dept)
    _df_quot = st.session_state["df_quot"]
    with st.spinner("Inspection des stations..."):
        _df_inspect = inspecter_stations(
            st.session_state["df_brut"], _df_quot,
            lat_centre, lon_centre, n=10,
        )
        st.session_state["df_inspect"] = _df_inspect
    st.rerun()

if not zone_definie:
    st.sidebar.caption("Renseignez la zone d'étude pour activer la recherche.")

# ==============================================================================
# SIDEBAR — ETAPE 2 : SELECTION + PERIODE (visible seulement après étape 1)
# ==============================================================================

if st.session_state["etape"] >= 1 and st.session_state["df_inspect"] is not None:

    st.sidebar.divider()
    st.sidebar.subheader("Stations")

    df_inspect    = st.session_state["df_inspect"]
    col_nom_sta   = "NOM_USUEL" if "NOM_USUEL" in df_inspect.columns else "NUM_POSTE"
    noms_stations = df_inspect[col_nom_sta].tolist()

    default_sel = (
        [s for s in (st.session_state["selection"] or []) if s in noms_stations]
        or noms_stations[:3]
    )
    selection = st.sidebar.multiselect(
        "Stations à analyser",
        options=noms_stations,
        default=default_sel,
        key="multiselect_stations",
    )
    st.session_state["selection"] = selection

    st.sidebar.subheader("Fenêtre temporelle")
    fenetre = st.sidebar.radio(
        "Période",
        ["24 dernières heures", "7 derniers jours", "15 derniers jours", "Personnalisée"],
        key="radio_fenetre",
    )

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if fenetre == "24 dernières heures":
        date_debut, date_fin = now - timedelta(hours=24), now
    elif fenetre == "7 derniers jours":
        date_debut, date_fin = now - timedelta(days=7), now
    elif fenetre == "15 derniers jours":
        date_debut, date_fin = now - timedelta(days=15), now
    else:
        c1, c2 = st.sidebar.columns(2)
        d1 = c1.date_input("Du", value=(now - timedelta(days=30)).date(), key="d1")
        d2 = c2.date_input("Au", value=now.date(), key="d2")
        h1 = st.sidebar.slider("Heure début (UTC)", 0, 23, 0,  format="%dh", key="h1")
        h2 = st.sidebar.slider("Heure fin (UTC)",   0, 23, 23, format="%dh", key="h2")
        date_debut = datetime.combine(d1, time(h1, 0))
        date_fin   = datetime.combine(d2, time(h2, 59))

    st.sidebar.caption(
        f"Du {date_debut.strftime('%d/%m/%Y %Hh')} "
        f"au {date_fin.strftime('%d/%m/%Y %Hh')} (UTC)"
    )

    if st.sidebar.button(
        "2 — Lancer l'analyse",
        type="primary",
        use_container_width=True,
        disabled=not selection,
    ):
        st.session_state["periode_active"] = {
            "fenetre":    fenetre,
            "date_debut": date_debut,
            "date_fin":   date_fin,
        }
        st.session_state["etape"] = 2

    if not selection:
        st.sidebar.caption("Sélectionnez au moins une station.")

with st.sidebar.expander("Mode debug"):
    debug_on = st.checkbox("Afficher la structure brute du fichier")

# ==============================================================================
# CORPS — TITRE ET CARTE
# ==============================================================================

st.title("QuickExtract Météo — Données horaires récentes")
st.caption(SOURCE_LABEL)

# Carte Folium (toujours affichée)
centre = ([lat_centre, lon_centre] if lat_centre else [46.5, 2.5])
zoom   = 10 if lat_centre else 5
m = folium.Map(location=centre, zoom_start=zoom, tiles="OpenStreetMap")
if lat_centre and lon_centre:
    folium.Marker(
        location=[lat_centre, lon_centre],
        popup=nom_commune or "Point d'étude",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

carte_result = st_folium(
    m, height=260,
    use_container_width=True,
    returned_objects=["last_clicked"] if mode_saisie == "Carte" else [],
    key="carte_principale",
)

# Récupération du clic carte
if mode_saisie == "Carte" and carte_result and carte_result.get("last_clicked"):
    clic = carte_result["last_clicked"]
    new_lat = round(clic["lat"], 6)
    new_lon = round(clic["lng"], 6)
    if new_lat != st.session_state["carte_lat"]:
        st.session_state["carte_lat"] = new_lat
        st.session_state["carte_lon"] = new_lon
        st.rerun()

# Arrêt si aucune action n'a encore été faite
if st.session_state["etape"] == 0:
    st.info("Renseignez la zone d'étude dans la barre latérale puis cliquez sur "
            "'1 — Chercher les stations'.")
    st.stop()

# ==============================================================================
# TELECHARGEMENT (déclenché uniquement après clic sur bouton 1)
# ==============================================================================

# Données déjà téléchargées dans le callback du bouton 1
if st.session_state["df_brut"] is None:
    st.error("Données non disponibles. Cliquez sur '1 — Chercher les stations'.")
    st.stop()

df_brut = st.session_state["df_brut"]
df_quot = st.session_state["df_quot"]

if debug_on:
    with st.expander("Structure du fichier source", expanded=True):
        debug_colonnes(df_brut)

nom_fichier = df_brut["_fichier_source"].iloc[0] if "_fichier_source" in df_brut.columns else ""
st.caption(
    f"Fichier : {nom_fichier} — {df_brut['NUM_POSTE'].nunique()} stations | "
    + (f"Historique : {df_quot['_fichier_source'].iloc[0]}"
       if df_quot is not None else "Historique : indisponible")
)

# ==============================================================================
# ETAPE 1 — TABLEAU DES STATIONS
# ==============================================================================

st.subheader("Stations disponibles")

df_inspect = st.session_state["df_inspect"]
if df_inspect is None:
    st.error("Inspection des stations non disponible. Cliquez sur '1 — Chercher les stations'.")
    st.stop()

cols_affich = [c for c in [
    "NUM_POSTE", "NOM_USUEL", "distance_km", "ALTI",
    "derniere_date", "fraicheur_jours", "nebulo_dispo", "historique_dispo",
] if c in df_inspect.columns]

df_display = df_inspect[cols_affich].copy()
df_display.rename(columns={
    "NUM_POSTE":        "Code",
    "NOM_USUEL":        "Nom",
    "distance_km":      "Dist. (km)",
    "ALTI":             "Alt. (m)",
    "derniere_date":    "Dernière mesure",
    "fraicheur_jours":  "Fraîcheur (j)",
    "nebulo_dispo":     "Nébulosité",
    "historique_dispo": "Historique",
}, inplace=True)

for col in ["Nébulosité", "Historique"]:
    if col in df_display.columns:
        df_display[col] = df_display[col].map({True: "oui", False: "non"})

st.dataframe(df_display, use_container_width=True, hide_index=True)

# Arrêt si l'analyse n'a pas encore été lancée
if st.session_state["etape"] < 2 or st.session_state["periode_active"] is None:
    st.info("Sélectionnez les stations et la période dans la barre latérale, "
            "puis cliquez sur '2 — Lancer l'analyse'.")
    st.stop()

# ==============================================================================
# ETAPE 2 — TRAITEMENT
# ==============================================================================

periode    = st.session_state["periode_active"]
fenetre    = periode["fenetre"]
date_debut = periode["date_debut"]
date_fin   = periode["date_fin"]
selection  = st.session_state["selection"] or []

col_nom_sta = "NOM_USUEL" if "NOM_USUEL" in df_inspect.columns else "NUM_POSTE"
df_stations = df_inspect[df_inspect[col_nom_sta].isin(selection)].copy()
ids_selec   = df_stations["NUM_POSTE"].tolist()

if not ids_selec:
    st.warning("Aucune station sélectionnée.")
    st.stop()

df_filtre = df_brut[df_brut["NUM_POSTE"].astype(str).isin([str(i) for i in ids_selec])]

try:
    df_obs = filtrer_periode(df_filtre, date_debut, date_fin)
except DonneesClimatoError as e:
    st.error(str(e))
    st.stop()

if df_obs.empty:
    st.warning(
        "Aucune observation sur cette période. "
        "Le fichier 'latest' couvre depuis janvier de l'année précédente jusqu'à J-1/J-2. "
        "Essayez une fenêtre plus large."
    )
    st.stop()

df_obs = normaliser_variables(df_obs)
df_obs = df_obs.merge(df_stations[["NUM_POSTE", "distance_km"]], on="NUM_POSTE", how="left")
df_agg = agreger_multi_stations(df_obs, df_stations)

if df_agg.empty:
    st.error("Erreur lors de l'agrégation des données.")
    st.stop()

df_normales      = calculer_normales(df_quot, ids_selec, n_annees=10)
df_normales_vent = calculer_normales_vent(df_quot, ids_selec, n_annees=10)

# Libellés dynamiques des normales (années réellement couvertes)
if df_normales is not None:
    _an_min = df_normales.attrs.get("annee_min", "")
    _an_max = df_normales.attrs.get("annee_max", "")
    label_normale = f"Normale {_an_min}-{_an_max}" if _an_min else "Normale (réf.)"
else:
    label_normale = "Normale (réf.)"

if df_normales_vent is not None:
    _anv_min = df_normales_vent.attrs.get("annee_min", "")
    _anv_max = df_normales_vent.attrs.get("annee_max", "")
    label_normale_vent = (
        f"Normale dir. {_anv_min}-{_anv_max} (vent fort)" if _anv_min
        else "Normale directionnelle (réf.)"
    )
else:
    label_normale_vent = "Normale directionnelle (réf.)"

# Titres et variables de période
titre_base  = nom_commune or f"dept. {st.session_state['code_dept_actif']}"
prefixe_nom = titre_base.replace(" ", "_")
periode_nom = f"{date_debut.strftime('%Y%m%d%H')}_{date_fin.strftime('%Y%m%d%H')}"
duree_jours = (date_fin - date_debut).days
duree_ans   = duree_jours / 365.25

# Résumé
st.subheader(f"Résultats — {titre_base}")
cap = (
    f"Stations : {', '.join(selection)} | "
    f"{date_debut.strftime('%d/%m/%Y %Hh')} → {date_fin.strftime('%d/%m/%Y %Hh')} UTC | "
    f"{len(df_agg)} pas horaires"
)
if df_normales is not None:
    an_min = df_normales.attrs.get("annee_min", "")
    an_max = df_normales.attrs.get("annee_max", "")
    cap += f" | Normales : {an_min}–{an_max}"
st.caption(cap)

# Indicateurs clés
c1, c2, c3, c4 = st.columns(4)
if colonne_presente(df_agg, "t_celsius"):
    c1.metric("T° moyenne", f"{df_agg['t_celsius'].mean():.1f} °C")
    c2.metric("T° min / max",
              f"{df_agg['t_celsius'].min():.1f} / {df_agg['t_celsius'].max():.1f} °C")
if colonne_presente(df_agg, "rr1_mm"):
    c3.metric("Précip. cumulées", f"{df_agg['rr1_mm'].sum():.1f} mm")
if colonne_presente(df_agg, "ff_ms"):
    c4.metric("Vent moyen", f"{df_agg['ff_ms'].mean():.1f} m/s")

# ==============================================================================
# AGREGATS
# ==============================================================================

df_agg["date_seule"] = df_agg["date_dt"].dt.date
df_agg["mois"]       = df_agg["date_dt"].dt.month
df_agg["annee"]      = df_agg["date_dt"].dt.year

cols_agg = {}
if colonne_presente(df_agg, "t_celsius"):
    cols_agg.update({
        "t_moy": ("t_celsius", "mean"),
        "t_min": ("t_celsius", "min"),
        "t_max": ("t_celsius", "max"),
    })
if colonne_presente(df_agg, "rr1_mm"):
    cols_agg["precip_tot"] = ("rr1_mm", "sum")
if colonne_presente(df_agg, "ff_ms"):
    cols_agg["vent_moy_ms"] = ("ff_ms", "mean")
    cols_agg["vent_max_ms"] = ("ff_ms", "max")
if colonne_presente(df_agg, "u_pct"):
    cols_agg["humidite_moy"] = ("u_pct", "mean")

agg_journalier = (
    df_agg.groupby("date_seule").agg(**cols_agg).reset_index()
    if cols_agg else pd.DataFrame()
)
if not agg_journalier.empty:
    agg_journalier["date_dt"] = pd.to_datetime(agg_journalier["date_seule"])
    agg_journalier["mois"]    = agg_journalier["date_dt"].dt.month

agg_mensuel = pd.DataFrame()
if not agg_journalier.empty:
    agg_mensuel = df_agg.groupby(["annee", "mois"]).agg(**cols_agg).reset_index()
    agg_mensuel["date_dt"] = pd.to_datetime(
        agg_mensuel["annee"].astype(str) + "-"
        + agg_mensuel["mois"].astype(str).str.zfill(2) + "-01"
    )

agg_annuel = (
    df_agg.groupby("annee").agg(**cols_agg).reset_index()
    if not agg_journalier.empty else pd.DataFrame()
)

# ==============================================================================
# GRAPHIQUES
# ==============================================================================

# 1. Courbe T° / précip horaire — 24h, 7j, personnalisée <= 7j
if (
    fenetre in ("24 dernières heures", "7 derniers jours")
    or (fenetre == "Personnalisée" and duree_jours <= 7)
) and colonne_presente(df_agg, "t_celsius"):
    st.subheader("Température et précipitations")
    fig = graphique_temp_precip(
        df_agg,
        titre=f"Évolution température / précipitations — {titre_base}",
        df_normales=df_normales,
        label_normale=label_normale,
    )
    _afficher(fig, f"temp_precip_{prefixe_nom}_{periode_nom}.png",
              "Télécharger — T° / Précipitations")

# 2. Rose des vents (toujours, avec normales directionnelles si disponibles)
if colonne_presente(df_agg, "ff_ms") and colonne_presente(df_agg, "dd_deg"):
    st.subheader("Rose des vents")
    fig = graphique_rose_vents(
        df_agg,
        titre=(
            f"Rose des vents — {titre_base}\n"
            f"({df_agg['date_dt'].min().strftime('%d/%m/%Y')} "
            f"— {df_agg['date_dt'].max().strftime('%d/%m/%Y')})"
        ),
        df_normales_vent=df_normales_vent,
        label_normale=label_normale_vent,
    )
    if fig:
        _afficher(fig, f"rose_vents_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Rose des vents")
        if df_normales_vent is None:
            st.caption(
                "Normale directionnelle non disponible pour ces stations "
                "(direction du vent absente ou insuffisante dans l'historique)."
            )

# 3. Suivi journalier — fenêtres courtes uniquement (7j, 15j, perso <= 31j)
# Distinct du thermopluviogramme (climatologie mensuelle) pour éviter la
# redondance sur les longues périodes.
SEUIL_MENSUEL_JOURS = 31

if (
    fenetre in ("7 derniers jours", "15 derniers jours")
    or (fenetre == "Personnalisée" and duree_jours <= SEUIL_MENSUEL_JOURS)
) and not agg_journalier.empty and colonne_presente(df_agg, "t_celsius"):

    st.subheader("Précipitations et températures — suivi journalier")

    if fenetre == "7 derniers jours":
        df_histo, label_p = agg_journalier.tail(7).reset_index(drop=True), "7 derniers jours"
    elif fenetre == "15 derniers jours":
        df_histo, label_p = agg_journalier.tail(15).reset_index(drop=True), "15 derniers jours"
    else:
        df_histo, label_p = agg_journalier.copy(), f"{duree_jours} jours"

    fig = graphique_histogramme_periode(
        df_histo,
        titre=f"Précipitations et températures — suivi journalier ({label_p}) — {titre_base}",
        df_normales=df_normales,
        label_normale=label_normale,
    )
    if fig:
        _afficher(fig, f"suivi_journalier_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Suivi journalier")

# 4. Thermopluviogramme — climatologie mensuelle (perso > 31j uniquement)
# Seul graphique mensuel affiché : pas de doublon avec le suivi journalier.
if (
    fenetre == "Personnalisée" and duree_jours > SEUIL_MENSUEL_JOURS
    and not agg_mensuel.empty
    and colonne_presente(df_agg, "t_celsius")
):
    st.subheader("Thermopluviogramme — bilan mensuel")
    mensuel_norm = agg_mensuel.groupby("mois").agg(
        t_moy=("t_moy", "mean"),
        t_min=("t_min", "mean"),
        t_max=("t_max", "mean"),
    ).reset_index()
    if "precip_tot" in agg_mensuel.columns:
        p_norm       = agg_mensuel.groupby("mois")["precip_tot"].mean().reset_index(name="precip_tot")
        mensuel_norm = mensuel_norm.merge(p_norm, on="mois")
    fig = graphique_thermopluviogramme(
        mensuel_norm,
        titre=f"Thermopluviogramme — bilan mensuel — {titre_base}",
        df_normales=df_normales,
        label_normale=label_normale,
    )
    if fig:
        _afficher(fig, f"thermopluvio_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Thermopluviogramme")

# 5. Évolution T° annuelles — personnalisée >= 1 an
if (
    fenetre == "Personnalisée" and duree_ans >= 1.0
    and not agg_annuel.empty
    and "t_min" in agg_annuel.columns
):
    st.subheader("Évolution des températures annuelles")
    fig = graphique_temperatures_annuelles(
        agg_annuel,
        titre=f"Évolution des températures annuelles — {titre_base}",
        df_normales=df_normales,
        label_normale=label_normale,
    )
    if fig:
        _afficher(fig, f"temp_annuelles_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Températures annuelles")

# ==============================================================================
# EXPORT CARTOGRAPHIQUE
# ==============================================================================

st.subheader("Export cartographique")
with st.expander("Paramètres de la carte", expanded=False):
    titre_carte  = st.text_input("Titre", value=f"Météo — {titre_base}", key="titre_carte")
    auteur_carte = st.text_input("Auteur", value="", key="auteur_carte")
    fichier_zone = st.file_uploader(
        "Zone d'étude (GeoJSON)", type=["geojson"], key="upload_zone"
    )
    fichier_logo = st.file_uploader("Logo (PNG)", type=["png"], key="upload_logo")

    if st.button("Générer la carte", key="btn_carte"):
        with st.spinner("Génération de la carte..."):
            try:
                from carte_folium import generer_carte_folium, charger_zone_geojson
                gdf_zone   = charger_zone_geojson(fichier_zone) if fichier_zone else None
                logo_bytes = fichier_logo.read() if fichier_logo else None
                html_carte = generer_carte_folium(
                    df_stations=df_stations,
                    titre=titre_carte,
                    gdf_zone=gdf_zone,
                    logo_bytes=logo_bytes,
                    auteur=auteur_carte,
                    sources="Météo-France, OpenStreetMap contributors",
                )
                components.html(html_carte, height=500, scrolling=False)
                st.download_button(
                    "Télécharger la carte (HTML)",
                    data=html_carte.encode(),
                    file_name=f"carte_{prefixe_nom}_{periode_nom}.html",
                    mime="text/html",
                    key="dl_carte",
                )
            except Exception as e:
                st.error(f"Erreur lors de la génération de la carte : {e}")

# ==============================================================================
# EXPORT EXCEL
# ==============================================================================

st.subheader("Export Excel")

df_export_agg = df_agg.drop(columns=["date_seule", "mois", "annee"], errors="ignore")
df_export_obs = df_obs.drop(columns=["_fichier_source"], errors="ignore")

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df_stations.to_excel(writer,    sheet_name="Stations",       index=False)
    df_export_agg.to_excel(writer,  sheet_name="Horaire_agrege", index=False)
    df_export_obs.to_excel(writer,  sheet_name="Horaire_brut",   index=False)
    if not agg_journalier.empty:
        agg_journalier.to_excel(writer, sheet_name="Journalier", index=False)
    if not agg_mensuel.empty:
        agg_mensuel.to_excel(writer,    sheet_name="Mensuel",    index=False)
    if not agg_annuel.empty:
        agg_annuel.to_excel(writer,     sheet_name="Annuel",     index=False)
    if df_normales is not None:
        df_normales.to_excel(writer,    sheet_name="Normales",   index=False)

st.download_button(
    "Télécharger l'Excel",
    data=buffer.getvalue(),
    file_name=f"meteo_{prefixe_nom}_{periode_nom}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="dl_excel",
)

"""
QuickExtract Météo — Données horaires récentes
================================================
Source : Météo-France / data.gouv.fr (open data, aucun compte requis).
Fraicheur : J-1 / J-2.

Déploiement Streamlit Community Cloud :
  1. Pousser app.py, mf_client.py, graphiques.py, requirements.txt sur GitHub.
  2. Sur share.streamlit.io : New app -> repo -> app.py.
  Aucun secret à configurer.
"""

import io
import requests as _req
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

from mf_client import (
    DonneesClimatoError,
    commune_depuis_insee,
    telecharger_horaire_departement,
    stations_proches,
    filtrer_periode,
    normaliser_variables,
    agreger_multi_stations,
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

SOURCE_LABEL = (
    "Source : Météo-France — Données climatologiques de base horaires (data.gouv.fr)"
)


# ==============================================================================
# SIDEBAR — ZONE D'ETUDE
# ==============================================================================

st.sidebar.title("Configuration")
st.sidebar.caption("Données ouvertes — aucun compte requis")

st.sidebar.subheader("Zone d'étude")

mode_saisie = st.sidebar.radio(
    "Mode de saisie",
    ["Code INSEE", "Coordonnées manuelles"],
    horizontal=True,
)

lat_centre  = None
lon_centre  = None
code_dept   = None
nom_commune = None

if mode_saisie == "Code INSEE":
    code_insee = st.sidebar.text_input(
        "Code INSEE commune", value="29232", max_chars=5,
        help="5 chiffres, ex. 29232 pour Quimper, 75056 pour Paris.",
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

else:
    lat_centre = st.sidebar.number_input("Latitude",    value=48.39, format="%.4f")
    lon_centre = st.sidebar.number_input("Longitude",   value=-4.49, format="%.4f")
    code_dept  = st.sidebar.text_input(
        "Code département", value="29",
        help="2 chiffres, ex. 29 pour le Finistère."
    )
    # Résolution du nom de commune le plus proche pour les titres
    if lat_centre and lon_centre:
        try:
            r = _req.get(
                "https://geo.api.gouv.fr/communes",
                params={
                    "lat": lat_centre, "lon": lon_centre,
                    "fields": "nom,codeDepartement",
                    "format": "json", "limit": 1,
                },
                timeout=8,
            )
            if r.ok and r.json():
                data = r.json()[0]
                nom_commune = data.get("nom", "")
                if not code_dept.strip():
                    code_dept = data.get("codeDepartement", "")
            st.sidebar.caption(
                f"Commune la plus proche : {nom_commune} — dept. {code_dept}"
            )
        except Exception:
            pass

st.sidebar.divider()

# ==============================================================================
# SIDEBAR — PARAMETRES
# ==============================================================================

n_stations = st.sidebar.slider("Nombre de stations à agréger", 1, 8, 3)

st.sidebar.subheader("Fenêtre temporelle")

fenetre = st.sidebar.radio(
    "Période",
    ["24 dernières heures", "7 derniers jours", "15 derniers jours", "Personnalisée"],
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
    d1 = c1.date_input("Du", value=(now - timedelta(days=30)).date())
    d2 = c2.date_input("Au", value=now.date())
    date_debut = datetime.combine(d1, datetime.min.time())
    date_fin   = datetime.combine(d2, datetime.max.time())

st.sidebar.caption(
    f"Du {date_debut.strftime('%d/%m/%Y %Hh')} "
    f"au {date_fin.strftime('%d/%m/%Y %Hh')} (UTC)"
)

lancer = st.sidebar.button("Lancer l'analyse", type="primary", use_container_width=True)

with st.sidebar.expander("Mode debug"):
    debug_on = st.checkbox("Afficher la structure brute du fichier")


# ==============================================================================
# CORPS PRINCIPAL
# ==============================================================================

st.title("QuickExtract Météo — Données horaires récentes")
st.caption(SOURCE_LABEL)

# Carte de visualisation du point (lecture seule, toujours visible si coords connues)
if lat_centre and lon_centre:
    popup = (nom_commune or "Point d'étude").replace("'", "\\'")
    carte_html = f"""<!DOCTYPE html><html><head>
      <meta charset="utf-8"/>
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <style>body{{margin:0;padding:0;}} #map{{width:100%;height:260px;}}</style>
    </head><body><div id="map"></div><script>
      var map = L.map('map').setView([{lat_centre},{lon_centre}],10);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{
        attribution:'OpenStreetMap contributors'}}).addTo(map);
      L.marker([{lat_centre},{lon_centre}]).addTo(map).bindPopup('{popup}').openPopup();
    </script></body></html>"""
    components.html(carte_html, height=265, scrolling=False)

if not lancer:
    st.info(
        "Renseignez les paramètres dans la barre latérale "
        "puis cliquez sur 'Lancer l'analyse'."
    )
    st.stop()

if not lat_centre or not lon_centre or not code_dept:
    st.error(
        "Zone d'étude non définie. "
        "Saisissez un code INSEE valide ou renseignez les coordonnées manuellement."
    )
    st.stop()


# ==============================================================================
# TELECHARGEMENT ET TRAITEMENT
# ==============================================================================

with st.spinner(f"Téléchargement des données — département {code_dept}..."):
    try:
        df_brut = telecharger_horaire_departement(code_dept)
    except DonneesClimatoError as e:
        st.error(str(e))
        st.stop()

if debug_on:
    with st.expander("Structure du fichier source", expanded=True):
        debug_colonnes(df_brut)

nom_fichier = (
    df_brut["_fichier_source"].iloc[0]
    if "_fichier_source" in df_brut.columns else ""
)
n_total = df_brut["NUM_POSTE"].nunique() if "NUM_POSTE" in df_brut.columns else "?"
st.caption(f"Fichier : {nom_fichier} — {n_total} stations dans le département")

# Stations proches
df_stations = stations_proches(df_brut, lat_centre, lon_centre, n=n_stations)
if df_stations.empty:
    st.error("Aucune station identifiée. Activez le mode debug.")
    st.stop()

st.subheader("Stations sélectionnées")
st.dataframe(
    df_stations[["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "distance_km"]],
    use_container_width=True,
    hide_index=True,
)

# Filtrage, normalisation, agrégation
ids = [str(i) for i in df_stations["NUM_POSTE"].tolist()]
df_filtre = df_brut[df_brut["NUM_POSTE"].astype(str).isin(ids)]

try:
    df_obs = filtrer_periode(df_filtre, date_debut, date_fin)
except DonneesClimatoError as e:
    st.error(str(e))
    st.stop()

if df_obs.empty:
    st.warning(
        "Aucune observation sur cette période. "
        "Le fichier 'latest' couvre depuis janvier de l'année précédente jusqu'à J-1/J-2. "
        "Essayez une fenêtre plus large ou une période moins récente."
    )
    st.stop()

df_obs = normaliser_variables(df_obs)
df_obs = df_obs.merge(
    df_stations[["NUM_POSTE", "distance_km"]], on="NUM_POSTE", how="left"
)
df_agg = agreger_multi_stations(df_obs, df_stations)

if df_agg.empty:
    st.error("Erreur lors de l'agrégation. Activez le mode debug.")
    st.stop()

st.caption(
    f"{len(df_agg)} pas de temps horaires — "
    f"{df_stations.shape[0]} station(s) — "
    f"{df_agg['date_dt'].min().strftime('%d/%m/%Y %Hh')} "
    f"à {df_agg['date_dt'].max().strftime('%d/%m/%Y %Hh')} UTC"
)


# ==============================================================================
# INDICATEURS CLES
# ==============================================================================

st.subheader("Indicateurs clés")
c1, c2, c3, c4 = st.columns(4)

if colonne_presente(df_agg, "t_celsius"):
    c1.metric("T° moyenne", f"{df_agg['t_celsius'].mean():.1f} °C")
    c2.metric(
        "T° min / max",
        f"{df_agg['t_celsius'].min():.1f} / {df_agg['t_celsius'].max():.1f} °C",
    )
if colonne_presente(df_agg, "rr1_mm"):
    c3.metric("Précip. cumulées", f"{df_agg['rr1_mm'].sum():.1f} mm")
if colonne_presente(df_agg, "ff_ms"):
    c4.metric("Vent moyen", f"{df_agg['ff_ms'].mean():.1f} m/s")


# ==============================================================================
# PREPARATION DES AGREGATS
# ==============================================================================

titre_base  = nom_commune or f"dept. {code_dept}"
prefixe_nom = (nom_commune or code_dept).replace(" ", "_")
periode_nom = f"{date_debut.strftime('%Y%m%d')}_{date_fin.strftime('%Y%m%d')}"
duree_jours = (date_fin - date_debut).days
duree_ans   = duree_jours / 365.25

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

agg_mensuel = (
    df_agg.groupby(["annee", "mois"]).agg(**cols_agg).reset_index()
    if not agg_journalier.empty else pd.DataFrame()
)
if not agg_mensuel.empty:
    agg_mensuel["date_dt"] = pd.to_datetime(
        agg_mensuel["annee"].astype(str) + "-"
        + agg_mensuel["mois"].astype(str).str.zfill(2) + "-01"
    )

agg_annuel = (
    df_agg.groupby("annee").agg(**cols_agg).reset_index()
    if not agg_journalier.empty else pd.DataFrame()
)


# ==============================================================================
# HELPERS GRAPHIQUES
# ==============================================================================

def _fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _afficher(fig, nom_fichier, label_dl="Télécharger ce graphique"):
    st.pyplot(fig)
    st.download_button(
        label=label_dl,
        data=_fig_png(fig),
        file_name=nom_fichier,
        mime="image/png",
        key=nom_fichier,
    )
    plt.close(fig)


# ==============================================================================
# GRAPHIQUES SELON LA FENETRE
# ==============================================================================

# 1. Courbe horaire T° / précip — 24h et 7j (et personnalisée <= 7j)
if (
    fenetre in ("24 dernières heures", "7 derniers jours")
    or (fenetre == "Personnalisée" and duree_jours <= 7)
) and colonne_presente(df_agg, "t_celsius"):
    st.subheader("Température et précipitations")
    fig = graphique_temp_precip(
        df_agg,
        titre=f"Évolution température / précipitations — {titre_base}",
    )
    _afficher(fig, f"temp_precip_{prefixe_nom}_{periode_nom}.png",
              "Télécharger — T° / Précipitations")

# 2. Rose des vents — toujours
if colonne_presente(df_agg, "ff_ms") and colonne_presente(df_agg, "dd_deg"):
    st.subheader("Rose des vents")
    fig = graphique_rose_vents(
        df_agg,
        titre=(
            f"Rose des vents — {titre_base}\n"
            f"({df_agg['date_dt'].min().strftime('%d/%m/%Y')} "
            f"— {df_agg['date_dt'].max().strftime('%d/%m/%Y')})"
        ),
    )
    if fig:
        _afficher(fig, f"rose_vents_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Rose des vents")

# 3. Histogramme — tout sauf 24h
if (
    fenetre != "24 dernières heures"
    and not agg_journalier.empty
    and colonne_presente(df_agg, "t_celsius")
):
    st.subheader("Précipitations et températures — bilan")

    if fenetre == "7 derniers jours":
        df_histo      = agg_journalier.tail(7).reset_index(drop=True)
        label_periode = "7 derniers jours"
    elif fenetre == "15 derniers jours":
        df_histo      = agg_journalier.tail(15).reset_index(drop=True)
        label_periode = "15 derniers jours"
    else:
        # Personnalisée : mensuel si > 31j, journalier sinon
        if duree_jours > 31 and not agg_mensuel.empty:
            df_histo      = agg_mensuel.copy()
            label_periode = "bilan mensuel"
        else:
            df_histo      = agg_journalier.copy()
            label_periode = f"{duree_jours} jours"

    fig = graphique_histogramme_periode(
        df_histo,
        titre=f"Précipitations et températures ({label_periode}) — {titre_base}",
    )
    if fig:
        _afficher(fig, f"histogramme_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Histogramme")

# 4. Thermopluviogramme — personnalisée >= 30j
if (
    fenetre == "Personnalisée"
    and duree_jours >= 30
    and not agg_mensuel.empty
    and colonne_presente(df_agg, "t_celsius")
):
    st.subheader("Thermopluviogramme — normales mensuelles")

    mensuel_norm = agg_mensuel.groupby("mois").agg(
        t_moy=("t_moy", "mean"),
        t_min=("t_min", "mean"),
        t_max=("t_max", "mean"),
    ).reset_index()
    if "precip_tot" in agg_mensuel.columns:
        p_norm     = agg_mensuel.groupby("mois")["precip_tot"].mean().reset_index(name="precip_tot")
        mensuel_norm = mensuel_norm.merge(p_norm, on="mois")

    fig = graphique_thermopluviogramme(
        mensuel_norm,
        titre=f"Thermopluviogramme — {titre_base}",
    )
    if fig:
        _afficher(fig, f"thermopluvio_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Thermopluviogramme")

# 5. Evolution températures annuelles — personnalisée >= 1 an
if (
    fenetre == "Personnalisée"
    and duree_ans >= 1.0
    and not agg_annuel.empty
    and "t_min" in agg_annuel.columns
):
    st.subheader("Évolution des températures annuelles")
    fig = graphique_temperatures_annuelles(
        agg_annuel,
        titre=f"Évolution des températures annuelles — {titre_base}",
    )
    if fig:
        _afficher(fig, f"temp_annuelles_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Températures annuelles")


# ==============================================================================
# EXPORT EXCEL
# ==============================================================================

st.subheader("Export Excel")

df_export_obs = df_obs.drop(columns=["_fichier_source"], errors="ignore")
df_export_agg = df_agg.drop(columns=["date_seule", "mois", "annee"], errors="ignore")

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df_stations.to_excel(writer,     sheet_name="Stations",       index=False)
    df_export_agg.to_excel(writer,   sheet_name="Horaire_agrege", index=False)
    df_export_obs.to_excel(writer,   sheet_name="Horaire_brut",   index=False)
    if not agg_journalier.empty:
        agg_journalier.to_excel(writer, sheet_name="Journalier",  index=False)
    if not agg_mensuel.empty:
        agg_mensuel.to_excel(writer,    sheet_name="Mensuel",     index=False)
    if not agg_annuel.empty:
        agg_annuel.to_excel(writer,     sheet_name="Annuel",      index=False)

st.download_button(
    label="Télécharger l'Excel (stations / horaire / journalier / mensuel / annuel)",
    data=buffer.getvalue(),
    file_name=f"meteo_{prefixe_nom}_{periode_nom}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

<<<<<<< Updated upstream
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
import streamlit.components.v1 as _components_v1
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
# HELPERS
# ==============================================================================

def _reverse_geocode(lat, lon):
    try:
        r = _req.get(
            "https://geo.api.gouv.fr/communes",
            params={"lat": lat, "lon": lon,
                    "fields": "nom,codeDepartement", "format": "json", "limit": 1},
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
    st.download_button(label=label, data=_fig_png(fig),
                       file_name=nom_fichier, mime="image/png", key=nom_fichier)
    plt.close(fig)


# ==============================================================================
# SIDEBAR — ZONE D'ETUDE
# ==============================================================================

st.sidebar.title("Configuration")
st.sidebar.caption("Données ouvertes — aucun compte requis")
st.sidebar.subheader("Zone d'étude")

mode_saisie = st.sidebar.radio(
    "Mode de saisie", ["Code INSEE", "Carte", "Coordonnées manuelles"],
    horizontal=True,
)

if "carte_lat" not in st.session_state:
    st.session_state["carte_lat"] = None
if "carte_lon" not in st.session_state:
    st.session_state["carte_lon"] = None
if "df_inspect" not in st.session_state:
    st.session_state["df_inspect"] = None
if "selection" not in st.session_state:
    st.session_state["selection"] = None
if "analyse_active" not in st.session_state:
    st.session_state["analyse_active"] = False
if "lancer_analyse" not in st.session_state:
    st.session_state["lancer_analyse"] = False

lat_centre = lon_centre = code_dept = nom_commune = None

if mode_saisie == "Code INSEE":
    code_insee = st.sidebar.text_input("Code INSEE commune", value="29232",
                                        max_chars=5)
    if code_insee.strip():
        try:
            nom_commune, lat_centre, lon_centre, code_dept = commune_depuis_insee(code_insee)
            st.sidebar.caption(f"{nom_commune} — dept. {code_dept}\n"
                               f"lat {lat_centre:.4f} / lon {lon_centre:.4f}")
        except DonneesClimatoError as e:
            st.sidebar.error(str(e))

elif mode_saisie == "Carte":
    if st.session_state["carte_lat"] is not None:
        lat_centre = st.session_state["carte_lat"]
        lon_centre = st.session_state["carte_lon"]
        nom_commune, code_dept = _reverse_geocode(lat_centre, lon_centre)
        st.sidebar.caption(f"lat {lat_centre:.4f} / lon {lon_centre:.4f}\n"
                           + (f"{nom_commune} — dept. {code_dept}" if nom_commune else ""))
        if st.sidebar.button("Réinitialiser le point"):
            st.session_state["carte_lat"] = st.session_state["carte_lon"] = None
            st.rerun()
    else:
        st.sidebar.info("Cliquez sur la carte pour choisir un point.")

else:
    lat_centre = st.sidebar.number_input("Latitude",   value=48.39, format="%.4f")
    lon_centre = st.sidebar.number_input("Longitude",  value=-4.49, format="%.4f")
    code_dept  = st.sidebar.text_input("Code département", value="29")
    if lat_centre and lon_centre:
        nom_commune, dept_auto = _reverse_geocode(lat_centre, lon_centre)
        if not code_dept.strip():
            code_dept = dept_auto
        st.sidebar.caption(f"Commune la plus proche : {nom_commune} — dept. {code_dept}")

st.sidebar.divider()

# ==============================================================================
# SIDEBAR — FENETRE TEMPORELLE
# ==============================================================================

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
    h1 = st.sidebar.slider("Heure de début (UTC)", 0, 23,
                            value=0, format="%dh")
    h2 = st.sidebar.slider("Heure de fin (UTC)",   0, 23,
                            value=23, format="%dh")
    date_debut = datetime.combine(d1, time(h1, 0))
    date_fin   = datetime.combine(d2, time(h2, 59))

st.sidebar.caption(f"Du {date_debut.strftime('%d/%m/%Y %Hh')} "
                   f"au {date_fin.strftime('%d/%m/%Y %Hh')} (UTC)")

st.sidebar.divider()

if st.sidebar.button("1 — Chercher les stations", type="primary",
                      use_container_width=True):
    st.session_state["analyse_active"] = True
    st.session_state["df_inspect"]     = None
    st.session_state["selection"]      = None
    st.session_state["lancer_analyse"] = False

with st.sidebar.expander("Mode debug"):
    debug_on = st.checkbox("Afficher la structure brute du fichier")


# ==============================================================================
# CORPS PRINCIPAL — TITRE + CARTE
# ==============================================================================

st.title("QuickExtract Météo — Données horaires récentes")
st.caption(SOURCE_LABEL)

# Carte Folium
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
    m, width="100%", height=280,
    returned_objects=["last_clicked"] if mode_saisie == "Carte" else [],
    key="carte_principale",
)

if mode_saisie == "Carte" and carte_result and carte_result.get("last_clicked"):
    clic = carte_result["last_clicked"]
    new_lat, new_lon = round(clic["lat"], 6), round(clic["lng"], 6)
    if new_lat != st.session_state.get("carte_lat"):
        st.session_state["carte_lat"] = new_lat
        st.session_state["carte_lon"] = new_lon
        st.rerun()

if not st.session_state.get("analyse_active"):
    st.info("Renseignez les paramètres dans la barre latérale puis cliquez sur 'Lancer l\'analyse'.")
    st.stop()

if not lat_centre or not lon_centre or not code_dept:
    st.error("Zone d'étude non définie.")
    st.stop()

# ==============================================================================
# TELECHARGEMENT
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

# Fichier quotidien historique pour les normales (téléchargé une fois, mis en cache 24h)
with st.spinner("Chargement des normales (données quotidiennes historiques)..."):
    df_quot = telecharger_quotidien_previous(code_dept)
if df_quot is not None:
    st.caption(
        f"Normales : fichier quotidien {df_quot['_fichier_source'].iloc[0]} chargé."
    )
else:
    st.caption("Normales non disponibles pour ce département (fichier quotidien introuvable).")

nom_fichier = df_brut["_fichier_source"].iloc[0] if "_fichier_source" in df_brut.columns else ""
st.caption(f"Fichier : {nom_fichier} — "
           f"{df_brut['NUM_POSTE'].nunique()} stations dans le département")



# ==============================================================================
# ETAPE 1 : SELECTION DES STATIONS
# ==============================================================================

st.subheader("Étape 1 — Sélection des stations")

# Inspection mise en cache dans session_state pour éviter le recalcul
if st.session_state.get("df_inspect") is None:
    with st.spinner("Inspection des stations..."):
        df_inspect = inspecter_stations(df_brut, lat_centre, lon_centre, n=10)
        st.session_state["df_inspect"] = df_inspect
df_inspect = st.session_state["df_inspect"]

# Affichage du tableau d'inspection
cols_affich = [c for c in [
    "NUM_POSTE", "NOM_USUEL", "distance_km", "ALTI",
    "derniere_date", "fraicheur_jours", "nebulo_dispo"
] if c in df_inspect.columns]

df_display = df_inspect[cols_affich].copy()
df_display.rename(columns={
    "NUM_POSTE":         "Code",
    "NOM_USUEL":         "Nom",
    "distance_km":       "Dist. (km)",
    "ALTI":              "Alt. (m)",
    "derniere_date":     "Dernière mesure",
    "fraicheur_jours":   "Fraîcheur (j)",
    "nebulo_dispo":      "Nébulosité",
}, inplace=True)

for col in ["Nébulosité"]:
    if col in df_display.columns:
        df_display[col] = df_display[col].map({True: "oui", False: "non"})

st.dataframe(df_display, use_container_width=True, hide_index=True)

# Sélection des stations — stockée en session pour ne pas perdre la sélection
noms_stations = (df_inspect["NOM_USUEL"].tolist()
                 if "NOM_USUEL" in df_inspect.columns
                 else df_inspect["NUM_POSTE"].tolist())

# Valeur par défaut : 3 premières stations, ou sélection précédente si compatible
default_sel = (
    [s for s in (st.session_state.get("selection") or []) if s in noms_stations]
    or noms_stations[:3]
)

selection = st.multiselect(
    "Choisissez les stations à utiliser pour l'analyse :",
    options=noms_stations,
    default=default_sel,
    key="multiselect_stations",
)
st.session_state["selection"] = selection

if not selection:
    st.warning("Sélectionnez au moins une station.")
    st.stop()

col_nom = "NOM_USUEL" if "NOM_USUEL" in df_inspect.columns else "NUM_POSTE"
df_stations = df_inspect[df_inspect[col_nom].isin(selection)].copy()
ids_selec   = df_stations["NUM_POSTE"].tolist()

if st.button("2 — Lancer l'analyse sur les stations sélectionnées",
              type="primary"):
    st.session_state["lancer_analyse"] = True

if not st.session_state.get("lancer_analyse"):
    st.info("Sélectionnez les stations puis cliquez sur '2 — Lancer l\'analyse'.")
    st.stop()

# ==============================================================================
# ETAPE 2 : TRAITEMENT
# ==============================================================================

st.subheader("Étape 2 — Analyse météo")

df_filtre = df_brut[df_brut["NUM_POSTE"].astype(str).isin([str(i) for i in ids_selec])]

try:
    df_obs = filtrer_periode(df_filtre, date_debut, date_fin)
except DonneesClimatoError as e:
    st.error(str(e))
    st.stop()

if df_obs.empty:
    st.warning("Aucune observation sur cette période. Essayez une fenêtre plus large.")
    st.stop()

df_obs     = normaliser_variables(df_obs)
df_obs     = df_obs.merge(df_stations[["NUM_POSTE", "distance_km"]],
                           on="NUM_POSTE", how="left")
df_agg     = agreger_multi_stations(df_obs, df_stations)

if df_agg.empty:
    st.error("Erreur lors de l'agrégation.")
    st.stop()

# Normales sur les 10 dernières années (depuis le fichier quotidien historique)
df_normales = calculer_normales(df_quot, ids_selec, n_annees=10)
if df_normales is not None:
    an_min = df_normales.attrs.get("annee_min", "")
    an_max = df_normales.attrs.get("annee_max", "")
    st.caption(f"Normales calculées sur {an_min}–{an_max} — superposées sur les graphiques.")
else:
    st.caption("Pas assez de données historiques pour calculer des normales.")

st.caption(
    f"{len(df_agg)} pas de temps horaires — "
    f"{df_agg['date_dt'].min().strftime('%d/%m/%Y %Hh')} "
    f"à {df_agg['date_dt'].max().strftime('%d/%m/%Y %Hh')} UTC"
)

# Indicateurs clés
titre_base  = nom_commune or f"dept. {code_dept}"
prefixe_nom = (nom_commune or code_dept).replace(" ", "_")
periode_nom = f"{date_debut.strftime('%Y%m%d%H')}_{date_fin.strftime('%Y%m%d%H')}"
duree_jours = (date_fin - date_debut).days
duree_ans   = duree_jours / 365.25

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
    cols_agg.update({"t_moy": ("t_celsius", "mean"),
                     "t_min": ("t_celsius", "min"),
                     "t_max": ("t_celsius", "max")})
if colonne_presente(df_agg, "rr1_mm"):
    cols_agg["precip_tot"] = ("rr1_mm", "sum")
if colonne_presente(df_agg, "ff_ms"):
    cols_agg["vent_moy_ms"] = ("ff_ms", "mean")
    cols_agg["vent_max_ms"] = ("ff_ms", "max")
if colonne_presente(df_agg, "u_pct"):
    cols_agg["humidite_moy"] = ("u_pct", "mean")

agg_journalier = (df_agg.groupby("date_seule").agg(**cols_agg).reset_index()
                  if cols_agg else pd.DataFrame())
if not agg_journalier.empty:
    agg_journalier["date_dt"] = pd.to_datetime(agg_journalier["date_seule"])
    agg_journalier["mois"]    = agg_journalier["date_dt"].dt.month

agg_mensuel = pd.DataFrame()
if not agg_journalier.empty:
    agg_mensuel = df_agg.groupby(["annee", "mois"]).agg(**cols_agg).reset_index()
    agg_mensuel["date_dt"] = pd.to_datetime(
        agg_mensuel["annee"].astype(str) + "-" +
        agg_mensuel["mois"].astype(str).str.zfill(2) + "-01"
    )

agg_annuel = (df_agg.groupby("annee").agg(**cols_agg).reset_index()
              if not agg_journalier.empty else pd.DataFrame())


# ==============================================================================
# GRAPHIQUES
# ==============================================================================

# 1. T° / précip horaire — 24h, 7j, perso <= 7j
if (fenetre in ("24 dernières heures", "7 derniers jours")
        or (fenetre == "Personnalisée" and duree_jours <= 7)) \
        and colonne_presente(df_agg, "t_celsius"):
    st.subheader("Température et précipitations")
    fig = graphique_temp_precip(
        df_agg,
        titre=f"Évolution température / précipitations — {titre_base}",
        df_normales=df_normales,
    )
    _afficher(fig, f"temp_precip_{prefixe_nom}_{periode_nom}.png",
              "Télécharger — T° / Précipitations")

# 2. Rose des vents — toujours
if colonne_presente(df_agg, "ff_ms") and colonne_presente(df_agg, "dd_deg"):
    st.subheader("Rose des vents")
    fig = graphique_rose_vents(
        df_agg,
        titre=(f"Rose des vents — {titre_base}\n"
               f"({df_agg['date_dt'].min().strftime('%d/%m/%Y')} "
               f"— {df_agg['date_dt'].max().strftime('%d/%m/%Y')})"),
    )
    if fig:
        _afficher(fig, f"rose_vents_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Rose des vents")

# 3. Histogramme — tout sauf 24h
if (fenetre != "24 dernières heures"
        and not agg_journalier.empty
        and colonne_presente(df_agg, "t_celsius")):
    st.subheader("Précipitations et températures — bilan")

    if fenetre == "7 derniers jours":
        df_histo, label_p = agg_journalier.tail(7).reset_index(drop=True), "7 derniers jours"
    elif fenetre == "15 derniers jours":
        df_histo, label_p = agg_journalier.tail(15).reset_index(drop=True), "15 derniers jours"
    elif duree_jours > 31 and not agg_mensuel.empty:
        df_histo, label_p = agg_mensuel.copy(), "bilan mensuel"
    else:
        df_histo, label_p = agg_journalier.copy(), f"{duree_jours} jours"

    fig = graphique_histogramme_periode(
        df_histo,
        titre=f"Précipitations et températures ({label_p}) — {titre_base}",
        df_normales=df_normales,
    )
    if fig:
        _afficher(fig, f"histogramme_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Histogramme")

# 4. Thermopluviogramme — perso >= 30j
if (fenetre == "Personnalisée" and duree_jours >= 30
        and not agg_mensuel.empty
        and colonne_presente(df_agg, "t_celsius")):
    st.subheader("Thermopluviogramme — normales mensuelles")

    mensuel_norm = agg_mensuel.groupby("mois").agg(
        t_moy=("t_moy", "mean"), t_min=("t_min", "mean"), t_max=("t_max", "mean")
    ).reset_index()
    if "precip_tot" in agg_mensuel.columns:
        p_norm       = agg_mensuel.groupby("mois")["precip_tot"].mean().reset_index(name="precip_tot")
        mensuel_norm = mensuel_norm.merge(p_norm, on="mois")

    fig = graphique_thermopluviogramme(
        mensuel_norm,
        titre=f"Thermopluviogramme — {titre_base}",
        df_normales=df_normales,
    )
    if fig:
        _afficher(fig, f"thermopluvio_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Thermopluviogramme")

# 5. Évolution T° annuelles — perso >= 1 an
if (fenetre == "Personnalisée" and duree_ans >= 1.0
        and not agg_annuel.empty
        and "t_min" in agg_annuel.columns):
    st.subheader("Évolution des températures annuelles")
    fig = graphique_temperatures_annuelles(
        agg_annuel,
        titre=f"Évolution des températures annuelles — {titre_base}",
        df_normales=df_normales,
    )
    if fig:
        _afficher(fig, f"temp_annuelles_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Températures annuelles")


# ==============================================================================
# EXPORT CARTOGRAPHIQUE
# ==============================================================================

st.subheader("Export cartographique")

with st.expander("Paramètres de la carte", expanded=False):
    titre_carte  = st.text_input("Titre", value=f"Météo — {titre_base}")
    auteur_carte = st.text_input("Auteur", value="")
    fichier_zone = st.file_uploader(
        "Zone d'étude (geojson)", type=["geojson"], key="upload_zone"
    )
    fichier_logo = st.file_uploader("Logo (PNG)", type=["png"], key="upload_logo")

    if st.button("Générer la carte"):
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
                )
                st.components.v1.html(html_carte, height=500, scrolling=False)
                st.download_button(
                    "Télécharger la carte (HTML)",
                    data=html_carte.encode(),
                    file_name=f"carte_{prefixe_nom}_{periode_nom}.html",
                    mime="text/html",
                )
            except ImportError:
                st.error("Module carte_folium manquant.")
            except Exception as e:
                st.error(f"Erreur : {e}")


# ==============================================================================
# EXPORT EXCEL
# ==============================================================================

st.subheader("Export Excel")

df_export_agg = df_agg.drop(columns=["date_seule", "mois", "annee"], errors="ignore")
df_export_obs = df_obs.drop(columns=["_fichier_source"], errors="ignore")

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df_stations.to_excel(writer,   sheet_name="Stations",       index=False)
    df_export_agg.to_excel(writer, sheet_name="Horaire_agrege", index=False)
    df_export_obs.to_excel(writer, sheet_name="Horaire_brut",   index=False)
    if not agg_journalier.empty:
        agg_journalier.to_excel(writer, sheet_name="Journalier", index=False)
    if not agg_mensuel.empty:
        agg_mensuel.to_excel(writer,    sheet_name="Mensuel",    index=False)
    if not agg_annuel.empty:
        agg_annuel.to_excel(writer,     sheet_name="Annuel",     index=False)
    if df_normales is not None:
        df_normales.to_excel(writer,    sheet_name="Normales_1991_2020", index=False)

st.download_button(
    "Télécharger l'Excel",
    data=buffer.getvalue(),
    file_name=f"meteo_{prefixe_nom}_{periode_nom}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
=======
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
# HELPERS
# ==============================================================================

def _reverse_geocode(lat, lon):
    try:
        r = _req.get(
            "https://geo.api.gouv.fr/communes",
            params={"lat": lat, "lon": lon,
                    "fields": "nom,codeDepartement", "format": "json", "limit": 1},
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
    st.download_button(label=label, data=_fig_png(fig),
                       file_name=nom_fichier, mime="image/png", key=nom_fichier)
    plt.close(fig)


# ==============================================================================
# SESSION STATE — initialisation
# ==============================================================================

for k, v in {
    "carte_lat":      None,
    "carte_lon":      None,
    "df_inspect":     None,
    "df_brut":        None,
    "df_quot":        None,
    "selection":      None,
    "etape":          1,       # 1 = chercher stations, 2 = lancer analyse
    "periode_active": None,    # dict avec date_debut/date_fin/fenetre
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==============================================================================
# SIDEBAR — ZONE D'ETUDE
# ==============================================================================

st.sidebar.title("Configuration")
st.sidebar.caption("Données ouvertes — aucun compte requis")
st.sidebar.subheader("Zone d'étude")

mode_saisie = st.sidebar.radio(
    "Mode de saisie", ["Code INSEE", "Carte", "Coordonnées manuelles"],
    horizontal=True,
)

lat_centre = lon_centre = code_dept = nom_commune = None

if mode_saisie == "Code INSEE":
    code_insee = st.sidebar.text_input("Code INSEE commune", value="29232", max_chars=5)
    if code_insee.strip():
        try:
            nom_commune, lat_centre, lon_centre, code_dept = commune_depuis_insee(code_insee)
            st.sidebar.caption(f"{nom_commune} — dept. {code_dept}\n"
                               f"lat {lat_centre:.4f} / lon {lon_centre:.4f}")
        except DonneesClimatoError as e:
            st.sidebar.error(str(e))

elif mode_saisie == "Carte":
    if st.session_state["carte_lat"] is not None:
        lat_centre  = st.session_state["carte_lat"]
        lon_centre  = st.session_state["carte_lon"]
        nom_commune, code_dept = _reverse_geocode(lat_centre, lon_centre)
        st.sidebar.caption(f"lat {lat_centre:.4f} / lon {lon_centre:.4f}\n"
                           + (f"{nom_commune} — dept. {code_dept}" if nom_commune else ""))
        if st.sidebar.button("Réinitialiser le point"):
            st.session_state["carte_lat"] = st.session_state["carte_lon"] = None
            st.rerun()
    else:
        st.sidebar.info("Cliquez sur la carte pour choisir un point.")

else:
    lat_centre = st.sidebar.number_input("Latitude",  value=48.39, format="%.4f")
    lon_centre = st.sidebar.number_input("Longitude", value=-4.49, format="%.4f")
    code_dept  = st.sidebar.text_input("Code département", value="29")
    if lat_centre and lon_centre:
        nom_commune, dept_auto = _reverse_geocode(lat_centre, lon_centre)
        if not code_dept.strip():
            code_dept = dept_auto
        st.sidebar.caption(f"Commune la plus proche : {nom_commune} — dept. {code_dept}")

st.sidebar.divider()

# ==============================================================================
# SIDEBAR — ETAPE 1 : CHERCHER LES STATIONS
# ==============================================================================

if st.sidebar.button("1 — Chercher les stations", type="primary",
                      use_container_width=True):
    if lat_centre and lon_centre and code_dept:
        st.session_state["etape"]      = 1
        st.session_state["df_inspect"] = None
        st.session_state["df_brut"]    = None
        st.session_state["df_quot"]    = None
        st.session_state["selection"]  = None
        st.session_state["periode_active"] = None
        st.rerun()
    else:
        st.sidebar.error("Zone d'étude non définie.")

# ==============================================================================
# SIDEBAR — ETAPE 2 : SELECTION STATIONS + PERIODE (visible après étape 1)
# ==============================================================================

if st.session_state["etape"] >= 1 and st.session_state.get("df_inspect") is not None:
    st.sidebar.divider()
    st.sidebar.subheader("Stations")

    df_inspect = st.session_state["df_inspect"]
    noms_stations = (df_inspect["NOM_USUEL"].tolist()
                     if "NOM_USUEL" in df_inspect.columns
                     else df_inspect["NUM_POSTE"].tolist())

    default_sel = (
        [s for s in (st.session_state.get("selection") or []) if s in noms_stations]
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
        h1 = st.sidebar.slider("Heure début (UTC)", 0, 23, 0, format="%dh", key="h1")
        h2 = st.sidebar.slider("Heure fin (UTC)",   0, 23, 23, format="%dh", key="h2")
        date_debut = datetime.combine(d1, time(h1, 0))
        date_fin   = datetime.combine(d2, time(h2, 59))

    st.sidebar.caption(f"Du {date_debut.strftime('%d/%m/%Y %Hh')} "
                       f"au {date_fin.strftime('%d/%m/%Y %Hh')} (UTC)")

    if st.sidebar.button("2 — Lancer l'analyse", type="primary",
                          use_container_width=True):
        if selection:
            st.session_state["etape"] = 2
            st.session_state["periode_active"] = {
                "fenetre":    fenetre,
                "date_debut": date_debut,
                "date_fin":   date_fin,
            }
        else:
            st.sidebar.warning("Sélectionnez au moins une station.")

with st.sidebar.expander("Mode debug"):
    debug_on = st.checkbox("Afficher la structure brute du fichier")

# ==============================================================================
# CORPS PRINCIPAL — CARTE
# ==============================================================================

st.title("QuickExtract Météo — Données horaires récentes")
st.caption(SOURCE_LABEL)

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
    m, width="100%", height=260,
    returned_objects=["last_clicked"] if mode_saisie == "Carte" else [],
    key="carte_principale",
)

if mode_saisie == "Carte" and carte_result and carte_result.get("last_clicked"):
    clic = carte_result["last_clicked"]
    new_lat, new_lon = round(clic["lat"], 6), round(clic["lng"], 6)
    if new_lat != st.session_state.get("carte_lat"):
        st.session_state["carte_lat"] = new_lat
        st.session_state["carte_lon"] = new_lon
        st.rerun()

if st.session_state["etape"] < 1 or not lat_centre:
    st.info("Renseignez la zone d'étude puis cliquez sur '1 — Chercher les stations'.")
    st.stop()

# ==============================================================================
# TELECHARGEMENT (une seule fois, mis en cache session)
# ==============================================================================

if st.session_state["df_brut"] is None:
    with st.spinner(f"Téléchargement des données — département {code_dept}..."):
        try:
            df_brut = telecharger_horaire_departement(code_dept)
            st.session_state["df_brut"] = df_brut
        except DonneesClimatoError as e:
            st.error(str(e))
            st.stop()
    with st.spinner("Chargement des données historiques..."):
        st.session_state["df_quot"] = telecharger_quotidien_previous(code_dept)

df_brut = st.session_state["df_brut"]
df_quot = st.session_state["df_quot"]

if debug_on:
    with st.expander("Structure du fichier source", expanded=True):
        debug_colonnes(df_brut)

nom_fichier = df_brut["_fichier_source"].iloc[0] if "_fichier_source" in df_brut.columns else ""
st.caption(f"Fichier horaire : {nom_fichier} | "
           + (f"Historique : {df_quot['_fichier_source'].iloc[0]}" if df_quot is not None else "Historique : indisponible"))

# ==============================================================================
# ETAPE 1 — TABLEAU DES STATIONS
# ==============================================================================

st.subheader("Stations disponibles")

if st.session_state["df_inspect"] is None:
    with st.spinner("Inspection des stations..."):
        df_inspect = inspecter_stations(
            df_brut, df_quot, lat_centre, lon_centre, n=10
        )
        st.session_state["df_inspect"] = df_inspect

df_inspect = st.session_state["df_inspect"]

cols_affich = [c for c in [
    "NUM_POSTE", "NOM_USUEL", "distance_km", "ALTI",
    "derniere_date", "fraicheur_jours", "nebulo_dispo", "historique_dispo"
] if c in df_inspect.columns]

df_display = df_inspect[cols_affich].copy()
df_display.rename(columns={
    "NUM_POSTE":         "Code",
    "NOM_USUEL":         "Nom",
    "distance_km":       "Dist. (km)",
    "ALTI":              "Alt. (m)",
    "derniere_date":     "Dernière mesure",
    "fraicheur_jours":   "Fraîcheur (j)",
    "nebulo_dispo":      "Nébulosité",
    "historique_dispo":  "Historique",
}, inplace=True)

for col in ["Nébulosité", "Historique"]:
    if col in df_display.columns:
        df_display[col] = df_display[col].map({True: "oui", False: "non"})

st.dataframe(df_display, use_container_width=True, hide_index=True)

if st.session_state["etape"] < 2 or st.session_state.get("periode_active") is None:
    st.info("Sélectionnez les stations et la période dans la barre latérale, "
            "puis cliquez sur '2 — Lancer l'analyse'.")
    st.stop()

# ==============================================================================
# ETAPE 2 — ANALYSE
# ==============================================================================

periode   = st.session_state["periode_active"]
fenetre   = periode["fenetre"]
date_debut = periode["date_debut"]
date_fin   = periode["date_fin"]
selection  = st.session_state["selection"] or []

col_nom    = "NOM_USUEL" if "NOM_USUEL" in df_inspect.columns else "NUM_POSTE"
df_stations = df_inspect[df_inspect[col_nom].isin(selection)].copy()
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
    st.warning("Aucune observation sur cette période. Essayez une fenêtre plus large.")
    st.stop()

df_obs  = normaliser_variables(df_obs)
df_obs  = df_obs.merge(df_stations[["NUM_POSTE", "distance_km"]], on="NUM_POSTE", how="left")
df_agg  = agreger_multi_stations(df_obs, df_stations)

if df_agg.empty:
    st.error("Erreur lors de l'agrégation.")
    st.stop()

df_normales      = calculer_normales(df_quot, ids_selec, n_annees=10)
df_normales_vent = calculer_normales_vent(df_quot, ids_selec, n_annees=10)

titre_base  = nom_commune or f"dept. {code_dept}"
prefixe_nom = (nom_commune or code_dept).replace(" ", "_")
periode_nom = f"{date_debut.strftime('%Y%m%d%H')}_{date_fin.strftime('%Y%m%d%H')}"
duree_jours = (date_fin - date_debut).days
duree_ans   = duree_jours / 365.25

st.subheader(f"Résultats — {titre_base}")
st.caption(
    f"Stations : {', '.join(selection)} | "
    f"{date_debut.strftime('%d/%m/%Y %Hh')} → {date_fin.strftime('%d/%m/%Y %Hh')} UTC | "
    f"{len(df_agg)} pas horaires"
)
if df_normales is not None:
    an_min = df_normales.attrs.get("annee_min", "")
    an_max = df_normales.attrs.get("annee_max", "")
    st.caption(f"Normales de référence : {an_min}–{an_max}")

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
    cols_agg.update({"t_moy": ("t_celsius", "mean"),
                     "t_min": ("t_celsius", "min"),
                     "t_max": ("t_celsius", "max")})
if colonne_presente(df_agg, "rr1_mm"):
    cols_agg["precip_tot"] = ("rr1_mm", "sum")
if colonne_presente(df_agg, "ff_ms"):
    cols_agg["vent_moy_ms"] = ("ff_ms", "mean")
    cols_agg["vent_max_ms"] = ("ff_ms", "max")
if colonne_presente(df_agg, "u_pct"):
    cols_agg["humidite_moy"] = ("u_pct", "mean")

agg_journalier = (df_agg.groupby("date_seule").agg(**cols_agg).reset_index()
                  if cols_agg else pd.DataFrame())
if not agg_journalier.empty:
    agg_journalier["date_dt"] = pd.to_datetime(agg_journalier["date_seule"])
    agg_journalier["mois"]    = agg_journalier["date_dt"].dt.month

agg_mensuel = pd.DataFrame()
if not agg_journalier.empty:
    agg_mensuel = df_agg.groupby(["annee", "mois"]).agg(**cols_agg).reset_index()
    agg_mensuel["date_dt"] = pd.to_datetime(
        agg_mensuel["annee"].astype(str) + "-" +
        agg_mensuel["mois"].astype(str).str.zfill(2) + "-01"
    )

agg_annuel = (df_agg.groupby("annee").agg(**cols_agg).reset_index()
              if not agg_journalier.empty else pd.DataFrame())

# ==============================================================================
# GRAPHIQUES
# ==============================================================================

# 1. T° / précip — 24h, 7j, perso <= 7j
if (fenetre in ("24 dernières heures", "7 derniers jours")
        or (fenetre == "Personnalisée" and duree_jours <= 7)) \
        and colonne_presente(df_agg, "t_celsius"):
    st.subheader("Température et précipitations")
    fig = graphique_temp_precip(
        df_agg,
        titre=f"Évolution température / précipitations — {titre_base}",
        df_normales=df_normales,
    )
    _afficher(fig, f"temp_precip_{prefixe_nom}_{periode_nom}.png",
              "Télécharger — T° / Précipitations")

# 2. Rose des vents avec normales superposées
if colonne_presente(df_agg, "ff_ms") and colonne_presente(df_agg, "dd_deg"):
    st.subheader("Rose des vents")
    fig = graphique_rose_vents(
        df_agg,
        titre=(f"Rose des vents — {titre_base}\n"
               f"({df_agg['date_dt'].min().strftime('%d/%m/%Y')} "
               f"— {df_agg['date_dt'].max().strftime('%d/%m/%Y')})"),
        df_normales_vent=df_normales_vent,
    )
    if fig:
        _afficher(fig, f"rose_vents_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Rose des vents")

# 3. Histogramme — tout sauf 24h
if (fenetre != "24 dernières heures"
        and not agg_journalier.empty
        and colonne_presente(df_agg, "t_celsius")):
    st.subheader("Précipitations et températures — bilan")

    if fenetre == "7 derniers jours":
        df_histo, label_p = agg_journalier.tail(7).reset_index(drop=True), "7 derniers jours"
    elif fenetre == "15 derniers jours":
        df_histo, label_p = agg_journalier.tail(15).reset_index(drop=True), "15 derniers jours"
    elif duree_jours > 31 and not agg_mensuel.empty:
        df_histo, label_p = agg_mensuel.copy(), "bilan mensuel"
    else:
        df_histo, label_p = agg_journalier.copy(), f"{duree_jours} jours"

    fig = graphique_histogramme_periode(
        df_histo,
        titre=f"Précipitations et températures ({label_p}) — {titre_base}",
        df_normales=df_normales,
    )
    if fig:
        _afficher(fig, f"histogramme_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Histogramme")

# 4. Thermopluviogramme — perso >= 30j
if (fenetre == "Personnalisée" and duree_jours >= 30
        and not agg_mensuel.empty
        and colonne_presente(df_agg, "t_celsius")):
    st.subheader("Thermopluviogramme — normales mensuelles")
    mensuel_norm = agg_mensuel.groupby("mois").agg(
        t_moy=("t_moy", "mean"), t_min=("t_min", "mean"), t_max=("t_max", "mean")
    ).reset_index()
    if "precip_tot" in agg_mensuel.columns:
        p_norm       = agg_mensuel.groupby("mois")["precip_tot"].mean().reset_index(name="precip_tot")
        mensuel_norm = mensuel_norm.merge(p_norm, on="mois")
    fig = graphique_thermopluviogramme(
        mensuel_norm,
        titre=f"Thermopluviogramme — {titre_base}",
        df_normales=df_normales,
    )
    if fig:
        _afficher(fig, f"thermopluvio_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Thermopluviogramme")

# 5. Évolution T° annuelles — perso >= 1 an
if (fenetre == "Personnalisée" and duree_ans >= 1.0
        and not agg_annuel.empty and "t_min" in agg_annuel.columns):
    st.subheader("Évolution des températures annuelles")
    fig = graphique_temperatures_annuelles(
        agg_annuel,
        titre=f"Évolution des températures annuelles — {titre_base}",
        df_normales=df_normales,
    )
    if fig:
        _afficher(fig, f"temp_annuelles_{prefixe_nom}_{periode_nom}.png",
                  "Télécharger — Températures annuelles")

# ==============================================================================
# EXPORT CARTOGRAPHIQUE
# ==============================================================================

st.subheader("Export cartographique")
with st.expander("Paramètres de la carte", expanded=False):
    titre_carte  = st.text_input("Titre", value=f"Météo — {titre_base}")
    auteur_carte = st.text_input("Auteur", value="")
    fichier_zone = st.file_uploader("Zone d'étude (geojson)", type=["geojson"], key="upload_zone")
    fichier_logo = st.file_uploader("Logo (PNG)", type=["png"], key="upload_logo")
    if st.button("Générer la carte"):
        with st.spinner("Génération de la carte..."):
            try:
                from carte_folium import generer_carte_folium, charger_zone_geojson
                gdf_zone   = charger_zone_geojson(fichier_zone) if fichier_zone else None
                logo_bytes = fichier_logo.read() if fichier_logo else None
                html_carte = generer_carte_folium(
                    df_stations=df_stations, titre=titre_carte,
                    gdf_zone=gdf_zone, logo_bytes=logo_bytes, auteur=auteur_carte,
                )
                components.html(html_carte, height=500, scrolling=False)
                st.download_button(
                    "Télécharger la carte (HTML)", data=html_carte.encode(),
                    file_name=f"carte_{prefixe_nom}_{periode_nom}.html", mime="text/html",
                )
            except Exception as e:
                st.error(f"Erreur : {e}")

# ==============================================================================
# EXPORT EXCEL
# ==============================================================================

st.subheader("Export Excel")
df_export_agg = df_agg.drop(columns=["date_seule", "mois", "annee"], errors="ignore")
df_export_obs = df_obs.drop(columns=["_fichier_source"], errors="ignore")

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df_stations.to_excel(writer,   sheet_name="Stations",       index=False)
    df_export_agg.to_excel(writer, sheet_name="Horaire_agrege", index=False)
    df_export_obs.to_excel(writer, sheet_name="Horaire_brut",   index=False)
    if not agg_journalier.empty:
        agg_journalier.to_excel(writer, sheet_name="Journalier", index=False)
    if not agg_mensuel.empty:
        agg_mensuel.to_excel(writer, sheet_name="Mensuel", index=False)
    if not agg_annuel.empty:
        agg_annuel.to_excel(writer, sheet_name="Annuel",   index=False)
    if df_normales is not None:
        df_normales.to_excel(writer, sheet_name="Normales", index=False)

st.download_button(
    "Télécharger l'Excel",
    data=buffer.getvalue(),
    file_name=f"meteo_{prefixe_nom}_{periode_nom}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
>>>>>>> Stashed changes

"""
QuickExtract Météo — Données horaires récentes
================================================
Source : Météo-France / data.gouv.fr (open data, aucun compte requis).
Fraicheur : J-1 / J-2.

Déploiement Streamlit Community Cloud :
  1. Pousser app.py, mf_client.py, graphiques.py, requirements.txt sur GitHub.
  2. Sur share.streamlit.io : New app -> sélectionner le repo -> app.py.
  Aucun secret à configurer.
"""

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

from mf_client import (
    DonneesClimatoError,
    commune_depuis_insee,
    dept_depuis_insee,
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

st.set_page_config(
    page_title="QuickExtract Météo",
    layout="wide",
)

SOURCE_LABEL = "Source : Météo-France — Données climatologiques de base horaires (data.gouv.fr)"


# ==============================================================================
# SIDEBAR
# ==============================================================================

st.sidebar.title("Configuration")
st.sidebar.caption("Données ouvertes data.gouv.fr — aucun compte requis")

st.sidebar.subheader("Zone d'étude")

mode_saisie = st.sidebar.radio(
    "Mode de saisie",
    ["Code INSEE", "Carte (clic)"],
    horizontal=True,
)

lat_centre  = None
lon_centre  = None
code_dept   = None
nom_commune = None

# --- Mode INSEE ---
if mode_saisie == "Code INSEE":
    code_insee = st.sidebar.text_input(
        "Code INSEE commune",
        value="29232",
        max_chars=5,
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

# --- Mode Carte ---
else:
    st.sidebar.caption(
        "Cliquez sur la carte dans le corps principal pour définir le point d'intérêt."
    )
    # Les coordonnées sont transmises via st.session_state après clic sur la carte.
    if "carte_lat" in st.session_state and "carte_lon" in st.session_state:
        lat_centre  = st.session_state["carte_lat"]
        lon_centre  = st.session_state["carte_lon"]
        # Département déduit des coordonnées via reverse geocoding simplifié :
        # on demande le code commune le plus proche et on en extrait le département.
        try:
            import requests as _req
            r = _req.get(
                "https://geo.api.gouv.fr/communes",
                params={
                    "lat": lat_centre, "lon": lon_centre,
                    "fields": "nom,codeDepartement", "format": "json", "limit": 1
                },
                timeout=8,
            )
            if r.ok and r.json():
                data = r.json()[0]
                nom_commune = data.get("nom", "")
                code_dept   = data.get("codeDepartement", "")
        except Exception:
            pass
        st.sidebar.caption(
            f"Point selectionne : lat {lat_centre:.4f} / lon {lon_centre:.4f}"
            + (f"\n{nom_commune} — dept. {code_dept}" if nom_commune else "")
        )
    else:
        st.sidebar.info("Cliquez sur la carte ci-dessous pour choisir un point.")

st.sidebar.divider()
n_stations = st.sidebar.slider("Nombre de stations à agréger", 1, 8, 3)

st.sidebar.subheader("Fenetre temporelle")

fenetre = st.sidebar.radio(
    "Periode",
    ["24 dernieres heures", "7 derniers jours", "15 derniers jours", "Personnalisee"],
)

now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

if fenetre == "24 dernieres heures":
    date_debut, date_fin = now - timedelta(hours=24), now
elif fenetre == "7 derniers jours":
    date_debut, date_fin = now - timedelta(days=7), now
elif fenetre == "15 derniers jours":
    date_debut, date_fin = now - timedelta(days=15), now
else:
    c1, c2 = st.sidebar.columns(2)
    d1 = c1.date_input("Du", value=(now - timedelta(days=7)).date())
    d2 = c2.date_input("Au", value=now.date())
    date_debut = datetime.combine(d1, datetime.min.time())
    date_fin   = datetime.combine(d2, datetime.max.time())

st.sidebar.caption(
    f"Du {date_debut.strftime('%d/%m/%Y %Hh')} au {date_fin.strftime('%d/%m/%Y %Hh')} (UTC)"
)

lancer = st.sidebar.button("Lancer l'analyse", type="primary", use_container_width=True)

with st.sidebar.expander("Mode debug"):
    debug_on = st.checkbox("Afficher la structure brute du fichier")


# ==============================================================================
# CORPS PRINCIPAL
# ==============================================================================

st.title("QuickExtract Météo — Données horaires recentes")
st.caption(SOURCE_LABEL)

# --- Carte Leaflet (mode carte ou toujours affichée si coordonnées connues) ---

if mode_saisie == "Carte (clic)" or (lat_centre is not None and lon_centre is not None):
    _lat  = lat_centre  or 46.5
    _lon  = lon_centre  or 2.5
    _zoom = 10 if lat_centre else 5

    carte_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ width: 100%; height: 340px; }}
        #coords {{ font-family: monospace; font-size: 12px;
                   padding: 4px 8px; background: #f5f5f5; }}
      </style>
    </head>
    <body>
      <div id="coords">Cliquez sur la carte pour selectionner un point</div>
      <div id="map"></div>
      <script>
        var map = L.map('map').setView([{_lat}, {_lon}], {_zoom});
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
          attribution: 'OpenStreetMap contributors'
        }}).addTo(map);

        var marker = null;
        {"var initMarker = L.marker([" + str(_lat) + "," + str(_lon) + "]).addTo(map); marker = initMarker;" if lat_centre else ""}

        map.on('click', function(e) {{
          var lat = e.latlng.lat.toFixed(6);
          var lon = e.latlng.lng.toFixed(6);
          document.getElementById('coords').textContent =
            'Point selectionne : lat ' + lat + ' / lon ' + lon;
          if (marker) {{ map.removeLayer(marker); }}
          marker = L.marker(e.latlng).addTo(map);
          // Transmission des coordonnees à Streamlit via query params
          window.parent.postMessage(
            {{type: 'streamlit:setComponentValue', value: {{lat: parseFloat(lat), lon: parseFloat(lon)}}}},
            '*'
          );
        }});
      </script>
    </body>
    </html>
    """

    result = components.html(carte_html, height=360, scrolling=False)

    # Récupération du clic (Streamlit Community ne supporte pas encore les
    # messages postMessage natifs sans composant custom ; on utilise
    # st.query_params comme canal de secours via un lien).
    # Solution pragmatique : afficher lat/lon dans un champ texte copiable
    # et laisser l'utilisateur les coller en mode INSEE si nécessaire.
    # La carte reste utile pour la visualisation du point et des stations.

if mode_saisie == "Carte (clic)" and lat_centre is None:
    st.info("Utilisez le mode 'Code INSEE' pour saisir votre zone, ou entrez manuellement les coordonnées ci-dessous.")
    col_lat, col_lon = st.columns(2)
    lat_saisie = col_lat.number_input("Latitude", value=46.5, format="%.4f", key="lat_man")
    lon_saisie = col_lon.number_input("Longitude", value=2.5,  format="%.4f", key="lon_man")
    dept_saisie = st.text_input("Code département", value="29", key="dept_man")
    if st.button("Valider ces coordonnées"):
        st.session_state["carte_lat"]  = lat_saisie
        st.session_state["carte_lon"]  = lon_saisie
        lat_centre  = lat_saisie
        lon_centre  = lon_saisie
        code_dept   = dept_saisie
        st.rerun()

if not lancer:
    st.info("Renseignez les paramètres dans la barre latérale puis cliquez sur 'Lancer l'analyse'.")
    st.stop()

if lat_centre is None or lon_centre is None or not code_dept:
    st.error(
        "Zone d'étude non définie. "
        "Saisissez un code INSEE valide ou renseignez les coordonnées manuellement."
    )
    st.stop()


# ==============================================================================
# TELECHARGEMENT ET TRAITEMENT
# ==============================================================================

with st.spinner(f"Téléchargement des données horaires — département {code_dept}..."):
    try:
        df_brut = telecharger_horaire_departement(code_dept)
    except DonneesClimatoError as e:
        st.error(str(e))
        st.stop()

if debug_on:
    with st.expander("Structure du fichier source", expanded=True):
        debug_colonnes(df_brut)

nom_fichier = df_brut["_fichier_source"].iloc[0] if "_fichier_source" in df_brut.columns else ""
n_total     = df_brut["NUM_POSTE"].nunique() if "NUM_POSTE" in df_brut.columns else "?"
st.caption(f"Fichier : {nom_fichier} — {n_total} stations dans le département")

# Stations proches
df_stations = stations_proches(df_brut, lat_centre, lon_centre, n=n_stations)

if df_stations.empty:
    st.error("Aucune station identifiée dans le fichier. Activez le mode debug.")
    st.stop()

st.subheader("Stations selectionnees")
st.dataframe(
    df_stations[["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "distance_km"]],
    use_container_width=True,
    hide_index=True,
)

# Filtrage et normalisation
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
        "Essayez une fenêtre plus large."
    )
    st.stop()

df_obs = normaliser_variables(df_obs)
df_obs = df_obs.merge(df_stations[["NUM_POSTE", "distance_km"]], on="NUM_POSTE", how="left")

# Agrégation pondérée multi-stations
df_agg = agreger_multi_stations(df_obs, df_stations)

if df_agg.empty:
    st.error("Erreur lors de l'agrégation. Activez le mode debug.")
    st.stop()

st.caption(
    f"{len(df_agg)} pas de temps horaires — "
    f"{df_stations.shape[0]} station(s) — "
    f"période : {df_agg['date_dt'].min().strftime('%d/%m/%Y %Hh')} "
    f"à {df_agg['date_dt'].max().strftime('%d/%m/%Y %Hh')} UTC"
)


# ==============================================================================
# INDICATEURS CLES
# ==============================================================================

st.subheader("Indicateurs cles")
cols_m = st.columns(4)

if colonne_presente(df_agg, "t_celsius"):
    cols_m[0].metric("T° moyenne", f"{df_agg['t_celsius'].mean():.1f} °C")
    cols_m[1].metric(
        "T° min / max",
        f"{df_agg['t_celsius'].min():.1f} / {df_agg['t_celsius'].max():.1f} °C",
    )

if colonne_presente(df_agg, "rr1_mm"):
    cols_m[2].metric("Précip. cumulées", f"{df_agg['rr1_mm'].sum():.1f} mm")

if colonne_presente(df_agg, "ff_ms"):
    cols_m[3].metric("Vent moyen", f"{df_agg['ff_ms'].mean():.1f} m/s")


# ==============================================================================
# PREPARATION COMMUNE
# ==============================================================================

titre_base   = nom_commune or f"dept. {code_dept}"
prefixe_nom  = (nom_commune or code_dept).replace(" ", "_")
periode_nom  = f"{date_debut.strftime('%Y%m%d')}_{date_fin.strftime('%Y%m%d')}"
duree_jours  = (date_fin - date_debut).days
duree_ans    = duree_jours / 365.25

df_agg["date_seule"] = df_agg["date_dt"].dt.date
df_agg["mois"]       = df_agg["date_dt"].dt.month
df_agg["annee"]      = df_agg["date_dt"].dt.year

# Agregation journaliere (toujours calculée, sert à plusieurs graphiques)
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

agg_journalier = df_agg.groupby("date_seule").agg(**cols_agg).reset_index() if cols_agg else pd.DataFrame()
if not agg_journalier.empty:
    agg_journalier["date_dt"] = pd.to_datetime(agg_journalier["date_seule"])

# Agregation mensuelle
agg_mensuel = pd.DataFrame()
if not agg_journalier.empty:
    agg_mensuel = df_agg.groupby(["annee", "mois"]).agg(**cols_agg).reset_index()

# Agregation annuelle
agg_annuel = pd.DataFrame()
if not agg_journalier.empty:
    agg_annuel = df_agg.groupby("annee").agg(**cols_agg).reset_index()

# Helper : convertit une figure en bytes PNG
def _fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# Helper : affiche figure + bouton download sous la figure
def _afficher_avec_export(fig, nom_fichier, titre_bouton="Télécharger ce graphique"):
    st.pyplot(fig)
    st.download_button(
        label=titre_bouton,
        data=_fig_png(fig),
        file_name=nom_fichier,
        mime="image/png",
        key=nom_fichier,
    )
    plt.close(fig)


# ==============================================================================
# GRAPHIQUES — SELECTION SELON LA FENETRE TEMPORELLE
# ==============================================================================

# ------------------------------------------------------------------
# 1. Courbe T° / précipitations horaire  (24h et 7j uniquement)
# ------------------------------------------------------------------
afficher_courbe_horaire = (
    fenetre in ("24 dernieres heures", "7 derniers jours")
    or (fenetre == "Personnalisee" and duree_jours <= 7)
)

if afficher_courbe_horaire and colonne_presente(df_agg, "t_celsius"):
    st.subheader("Temperature et precipitations")
    fig = graphique_temp_precip(
        df_agg,
        titre=f"Evolution temperature / precipitations — {titre_base}",
    )
    _afficher_avec_export(
        fig,
        f"temp_precip_{prefixe_nom}_{periode_nom}.png",
        "Télécharger — T° / Précipitations",
    )

# ------------------------------------------------------------------
# 2. Rose des vents  (toujours)
# ------------------------------------------------------------------
if colonne_presente(df_agg, "ff_ms") and colonne_presente(df_agg, "dd_deg"):
    st.subheader("Rose des vents")
    fig = graphique_rose_vents(
        df_agg,
        titre=(
            f"Rose des vents — {titre_base}\n"
            f"({df_agg['date_dt'].min().strftime('%d/%m/%Y')} "
            f"- {df_agg['date_dt'].max().strftime('%d/%m/%Y')})"
        ),
    )
    if fig:
        _afficher_avec_export(
            fig,
            f"rose_vents_{prefixe_nom}_{periode_nom}.png",
            "Télécharger — Rose des vents",
        )

# ------------------------------------------------------------------
# 3. Histogramme journalier sur la période  (7j, 15j, personnalisée)
# ------------------------------------------------------------------
afficher_histogramme = fenetre != "24 dernieres heures"

if afficher_histogramme and not agg_journalier.empty and colonne_presente(df_agg, "t_celsius"):
    st.subheader("Précipitations et températures — bilan journalier")

    # Titre de l'axe x adapté
    if fenetre == "7 derniers jours":
        label_periode = "7 derniers jours"
        df_histo = agg_journalier.tail(7).reset_index(drop=True)
    elif fenetre == "15 derniers jours":
        label_periode = "15 derniers jours"
        df_histo = agg_journalier.tail(15).reset_index(drop=True)
    else:
        # Personnalisée : agrégation mensuelle si > 31j, journalière sinon
        if duree_jours > 31:
            label_periode = "bilan mensuel"
            df_histo = agg_mensuel.copy()
            df_histo["date_dt"] = pd.to_datetime(
                df_histo["annee"].astype(str) + "-" + df_histo["mois"].astype(str).str.zfill(2) + "-01"
            )
        else:
            label_periode = f"{duree_jours} derniers jours"
            df_histo = agg_journalier.copy()

    from graphiques import graphique_histogramme_periode
    fig = graphique_histogramme_periode(
        df_histo,
        titre=f"Précipitations et températures ({label_periode}) — {titre_base}",
    )
    if fig:
        _afficher_avec_export(
            fig,
            f"histogramme_{prefixe_nom}_{periode_nom}.png",
            "Télécharger — Histogramme",
        )

# ------------------------------------------------------------------
# 4. Thermopluviogramme mensuel  (personnalisée uniquement)
# ------------------------------------------------------------------
afficher_thermo = fenetre == "Personnalisee" and duree_jours >= 30

if afficher_thermo and not agg_mensuel.empty and colonne_presente(df_agg, "t_celsius"):
    st.subheader("Thermopluviogramme — normales mensuelles")

    mensuel_norm = agg_mensuel.groupby("mois").agg(
        t_moy=("t_moy", "mean"),
        t_min=("t_min", "mean"),
        t_max=("t_max", "mean"),
    ).reset_index()
    if "precip_tot" in agg_mensuel.columns:
        mensuel_norm_p = agg_mensuel.groupby("mois")["precip_tot"].mean().reset_index(name="precip_tot")
        mensuel_norm   = mensuel_norm.merge(mensuel_norm_p, on="mois")

    fig = graphique_thermopluviogramme(
        mensuel_norm,
        titre=f"Thermopluviogramme — {titre_base}",
    )
    if fig:
        _afficher_avec_export(
            fig,
            f"thermopluvio_{prefixe_nom}_{periode_nom}.png",
            "Télécharger — Thermopluviogramme",
        )

# ------------------------------------------------------------------
# 5. Evolution des temperatures annuelles  (personnalisée >= 1 an uniquement)
# ------------------------------------------------------------------
afficher_annuel = fenetre == "Personnalisee" and duree_ans >= 1.0

if afficher_annuel and not agg_annuel.empty and "t_min" in agg_annuel.columns:
    st.subheader("Evolution des temperatures annuelles")
    fig = graphique_temperatures_annuelles(
        agg_annuel,
        titre=f"Evolution des temperatures annuelles — {titre_base}",
    )
    if fig:
        _afficher_avec_export(
            fig,
            f"temp_annuelles_{prefixe_nom}_{periode_nom}.png",
            "Télécharger — Températures annuelles",
        )


# ==============================================================================
# EXPORT EXCEL
# ==============================================================================

st.subheader("Export Excel")

df_export_obs = df_obs.drop(columns=["_fichier_source"], errors="ignore")
df_export_agg = df_agg.drop(columns=["date_seule", "mois", "annee"], errors="ignore")

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df_stations.to_excel(writer, sheet_name="Stations", index=False)
    df_export_agg.to_excel(writer, sheet_name="Horaire_agrege", index=False)
    df_export_obs.to_excel(writer, sheet_name="Horaire_brut", index=False)
    if not agg_journalier.empty:
        agg_journalier.to_excel(writer, sheet_name="Journalier", index=False)
    if not agg_mensuel.empty:
        agg_mensuel.to_excel(writer, sheet_name="Mensuel", index=False)
    if not agg_annuel.empty:
        agg_annuel.to_excel(writer, sheet_name="Annuel", index=False)

nom_export = (
    f"meteo_{prefixe_nom}_{periode_nom}.xlsx"
)

st.download_button(
    label="Télécharger l'Excel (stations / horaire / journalier / mensuel / annuel)",
    data=buffer.getvalue(),
    file_name=nom_export,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

"""
Export cartographique léger via Folium (HTML interactif).
Aucune dépendance lourde (pas de geopandas, contextily, GDAL).
Retourne du HTML prêt à être affiché avec st.components.v1.html()
ou téléchargé directement.
"""

import io
import json
import base64
import math
from datetime import datetime

import folium

# Charte graphique SCE
C_ORANGE = "#E07020"
C_MARINE = "#1A3A4A"
C_GRIS   = "#6D7274"


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def charger_zone_geojson(fichier_upload):
    """
    Charge un GeoJSON uploadé via st.file_uploader.
    Retourne le dict GeoJSON ou None en cas d'erreur.
    """
    if fichier_upload is None:
        return None
    try:
        return json.loads(fichier_upload.read().decode("utf-8"))
    except Exception:
        return None


def generer_carte_folium(
    df_stations,
    titre="Carte météo",
    gdf_zone=None,
    logo_bytes=None,
    auteur="",
    sources="Météo-France, OpenStreetMap contributors",
):
    """
    Génère une carte Folium (HTML) avec :
      - Fond OpenStreetMap
      - Zone d'étude en contour orange (GeoJSON)
      - Stations météo en marqueurs rouges avec popup
      - Titre, auteur, sources en cartouche HTML injecté dans la carte

    Args:
        df_stations  : DataFrame avec NUM_POSTE, NOM_USUEL, LAT, LON, distance_km
        titre        : titre affiché dans le cartouche
        gdf_zone     : dict GeoJSON de la zone d'étude (optionnel)
        logo_bytes   : bytes PNG du logo (optionnel, intégré en base64)
        auteur       : nom de l'auteur
        sources      : texte des sources

    Returns:
        str HTML complet de la carte Folium
    """
    # Centre de la carte
    lats = [float(r["LAT"]) for _, r in df_stations.iterrows() if r.get("LAT")]
    lons = [float(r["LON"]) for _, r in df_stations.iterrows() if r.get("LON")]

    if gdf_zone is not None:
        try:
            coords = _extraire_coords_geojson(gdf_zone)
            lats += [c[1] for c in coords]
            lons += [c[0] for c in coords]
        except Exception:
            pass

    if not lats:
        lats, lons = [46.5], [2.5]

    lat_c = sum(lats) / len(lats)
    lon_c = sum(lons) / len(lons)

    # Niveau de zoom adapté à l'étendue
    if len(lats) > 1:
        dlat = max(lats) - min(lats)
        dlon = max(lons) - min(lons)
        span = max(dlat, dlon)
        zoom = 12 if span < 0.1 else 11 if span < 0.3 else 10 if span < 1 else 9 if span < 3 else 7
    else:
        zoom = 11

    # Carte Folium
    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        prefer_canvas=True,
    )

    # Zone d'étude
    if gdf_zone is not None:
        try:
            folium.GeoJson(
                gdf_zone,
                name="Zone d'étude",
                style_function=lambda _: {
                    "fillColor": C_ORANGE,
                    "color":     C_ORANGE,
                    "weight":    2.5,
                    "fillOpacity": 0.08,
                },
            ).add_to(m)
        except Exception:
            pass

    # Stations
    for _, row in df_stations.iterrows():
        if not row.get("LAT") or not row.get("LON"):
            continue
        nom  = str(row.get("NOM_USUEL", row.get("NUM_POSTE", "")))
        dist = row.get("distance_km", "")
        alti = row.get("ALTI", "")
        popup_html = f"""
        <div style="font-family:sans-serif;font-size:13px;min-width:160px">
            <b style="color:{C_MARINE}">{nom}</b><br>
            <span style="color:{C_GRIS}">N° {row.get('NUM_POSTE','')}</span><br>
            {"Distance : " + str(dist) + " km<br>" if dist else ""}
            {"Altitude : " + str(alti) + " m" if alti else ""}
        </div>"""

        folium.CircleMarker(
            location=[float(row["LAT"]), float(row["LON"])],
            radius=7,
            color="white",
            weight=1.5,
            fill=True,
            fill_color="#D62728",
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=nom,
        ).add_to(m)

    # Cartouche HTML injecté en bas de carte
    date_str = datetime.today().strftime("%d/%m/%Y")
    logo_html = ""
    if logo_bytes:
        try:
            b64 = base64.b64encode(logo_bytes).decode()
            logo_html = f'<img src="data:image/png;base64,{b64}" style="height:32px;float:right;margin-left:10px"/>'
        except Exception:
            pass

    cartouche = f"""
    <div style="
        position:fixed; bottom:0; left:0; right:0; z-index:9999;
        background:white; border-top:3px solid {C_ORANGE};
        padding:6px 12px; font-family:sans-serif;
        display:flex; align-items:center; gap:12px;
    ">
        {logo_html}
        <div style="flex:1">
            <span style="font-weight:bold;color:{C_MARINE};font-size:13px">{titre}</span>
            <span style="color:{C_GRIS};font-size:10px;margin-left:12px">
                Sources : {sources}
            </span>
        </div>
        <div style="color:{C_GRIS};font-size:10px;text-align:right">
            {("Auteur : " + auteur + " — ") if auteur else ""}{date_str}
        </div>
    </div>"""

    m.get_root().html.add_child(folium.Element(cartouche))

    return m.get_root().render()


def _extraire_coords_geojson(geojson):
    """Extrait toutes les coordonnées [lon, lat] d'un GeoJSON."""
    coords = []

    def _walk(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "Point":
                coords.append(obj["coordinates"])
            elif obj.get("type") in ("LineString", "MultiPoint"):
                coords.extend(obj["coordinates"])
            elif obj.get("type") in ("Polygon", "MultiLineString"):
                for ring in obj["coordinates"]:
                    coords.extend(ring)
            elif obj.get("type") == "MultiPolygon":
                for poly in obj["coordinates"]:
                    for ring in poly:
                        coords.extend(ring)
            elif obj.get("type") in ("Feature", "FeatureCollection"):
                for v in obj.values():
                    _walk(v)
            else:
                for v in obj.values():
                    _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(geojson)
    return coords

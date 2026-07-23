"""
Export cartographique — mise en page paysage A5.
Fond OpenStreetMap via contextily. Charte graphique proche SCE.

Dépendances supplémentaires : contextily, geopandas, shapely, pyproj, Pillow
"""

import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime

try:
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import Point
    from pyproj import Transformer
    CTX_OK = True
except ImportError:
    CTX_OK = False

# Charte SCE
C_ORANGE  = "#E07020"
C_MARINE  = "#1A3A4A"
C_GRIS    = "#6D7274"
C_FOND    = "#F5F5F5"
C_ZONE    = "#E07020"    # contour zone d'étude
C_STATION = "#D62728"    # symbole station

# A5 paysage à 150 dpi
FIG_W_IN = 8.27
FIG_H_IN = 5.83


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _bbox_avec_marge(geoms_wgs84, marge_km=2.0):
    """Retourne (minx, miny, maxx, maxy) en WGS84 avec une marge en km."""
    all_x, all_y = [], []
    for g in geoms_wgs84:
        if g is not None:
            b = g.bounds
            all_x += [b[0], b[2]]
            all_y += [b[1], b[3]]
    if not all_x:
        return None
    deg_lat = marge_km / 111.0
    deg_lon = marge_km / (111.0 * math.cos(math.radians(sum(all_y)/len(all_y))))
    return (min(all_x)-deg_lon, min(all_y)-deg_lat,
            max(all_x)+deg_lon, max(all_y)+deg_lat)


def _barre_echelle(ax, ax_crs, lon_c, lat_c, longueur_km=1):
    """Dessine une barre d'échelle simple en bas à droite."""
    transformer = Transformer.from_crs("EPSG:4326", ax_crs, always_xy=True)
    x0, y0 = transformer.transform(lon_c, lat_c)
    deg_lon = longueur_km / (111.0 * math.cos(math.radians(lat_c)))
    x1, _ = transformer.transform(lon_c + deg_lon, lat_c)
    dx = x1 - x0

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xr = xlim[1] - 0.02*(xlim[1]-xlim[0])
    xl = xr - dx
    yb = ylim[0] + 0.04*(ylim[1]-ylim[0])

    ax.plot([xl, xr], [yb, yb], color="black", linewidth=2.5, solid_capstyle="butt")
    ax.plot([xl, xl], [yb-dx*0.02, yb+dx*0.02], color="black", linewidth=1.5)
    ax.plot([xr, xr], [yb-dx*0.02, yb+dx*0.02], color="black", linewidth=1.5)
    ax.text((xl+xr)/2, yb + dx*0.04, f"{longueur_km} km",
            ha="center", va="bottom", fontsize=7, color="black",
            fontweight="bold")


def _fleche_nord(ax):
    """Flèche nord en haut à droite de la carte."""
    ax.annotate("N", xy=(0.96, 0.92), xycoords="axes fraction",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=C_MARINE)
    ax.annotate("", xy=(0.96, 0.96), xytext=(0.96, 0.88),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=C_MARINE,
                                lw=1.5, mutation_scale=10))


def generer_carte(
    df_stations,
    titre="",
    gdf_zone=None,
    logo_bytes=None,
    sources="IGN-BD TOPO, Météo-France, OpenStreetMap contributors",
    auteur="",
    date_str=None,
    zoom=None,
):
    """
    Génère la carte de mise en page A5 paysage.

    Args:
        df_stations : DataFrame avec colonnes NUM_POSTE, NOM_USUEL, LAT, LON
        titre       : titre affiché en haut de la carte
        gdf_zone    : GeoDataFrame zone d'étude (optionnel, CRS quelconque)
        logo_bytes  : bytes du logo PNG (optionnel)
        sources     : texte de la ligne sources
        auteur      : nom de l'auteur pour le cartouche
        date_str    : date du cartouche (défaut : aujourd'hui)
        zoom        : niveau de zoom contextily (None = auto)

    Returns:
        bytes PNG
    """
    if not CTX_OK:
        raise ImportError(
            "Librairies manquantes pour la carte : contextily, geopandas, shapely, pyproj. "
            "Ajoutez-les à requirements.txt."
        )

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")

    # Layout : carte principale + cartouche bas
    ax_carte  = fig.add_axes([0.01, 0.12, 0.98, 0.82])   # zone carte
    ax_bas    = fig.add_axes([0.0,  0.0,  1.0,  0.12])   # cartouche bas
    ax_bas.axis("off")

    # --- Titre ---
    fig.text(0.5, 0.96, titre or "Carte météo",
             ha="center", va="top", fontsize=13, fontweight="bold",
             color=C_MARINE, fontfamily="sans-serif")
    fig.add_artist(plt.Line2D([0.01, 0.99], [0.945, 0.945],
                              color=C_ORANGE, linewidth=2.5,
                              transform=fig.transFigure))

    # --- Reprojection Web Mercator (EPSG:3857) ---
    transformer_to = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    geoms_wgs = []

    # Stations
    sta_pts = []
    for _, r in df_stations.iterrows():
        if pd.notna(r.get("LAT")) and pd.notna(r.get("LON")):
            sta_pts.append((float(r["LON"]), float(r["LAT"]),
                            str(r.get("NOM_USUEL", r.get("NUM_POSTE", "")))))
            from shapely.geometry import Point as _Pt
            geoms_wgs.append(_Pt(float(r["LON"]), float(r["LAT"])))

    # Zone d'étude
    gdf_zone_3857 = None
    if gdf_zone is not None and not gdf_zone.empty:
        gdf_zone_3857 = gdf_zone.to_crs("EPSG:3857")
        for geom in gdf_zone.to_crs("EPSG:4326").geometry:
            geoms_wgs.append(geom)

    # Extent
    bbox = _bbox_avec_marge(geoms_wgs, marge_km=2.0)
    if bbox is None:
        ax_carte.text(0.5, 0.5, "Aucune donnée spatiale disponible",
                      ha="center", va="center", transform=ax_carte.transAxes)
    else:
        xmin, ymin, xmax, ymax = [
            transformer_to.transform(bbox[0], bbox[1]),
            transformer_to.transform(bbox[2], bbox[3]),
        ][0] + [transformer_to.transform(bbox[0], bbox[1]),
                transformer_to.transform(bbox[2], bbox[3])][1]

        x0, y0 = transformer_to.transform(bbox[0], bbox[1])
        x1, y1 = transformer_to.transform(bbox[2], bbox[3])
        ax_carte.set_xlim(x0, x1)
        ax_carte.set_ylim(y0, y1)

        # Fond OSM
        try:
            ctx.add_basemap(
                ax_carte, crs="EPSG:3857",
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom=zoom, attribution=False,
            )
        except Exception:
            ax_carte.set_facecolor(C_FOND)

        # Zone d'étude
        if gdf_zone_3857 is not None:
            gdf_zone_3857.plot(
                ax=ax_carte, facecolor="none",
                edgecolor=C_ZONE, linewidth=2.2, zorder=4,
            )

        # Stations
        for lon, lat, nom in sta_pts:
            xp, yp = transformer_to.transform(lon, lat)
            ax_carte.plot(xp, yp, marker="o", color=C_STATION,
                          markersize=7, zorder=5,
                          markeredgecolor="white", markeredgewidth=0.8)
            ax_carte.text(xp, yp, f"  {nom}", fontsize=7, color=C_MARINE,
                          va="center", zorder=6,
                          path_effects=[pe.withStroke(linewidth=2, foreground="white")])

        # Légende
        handles = []
        if gdf_zone_3857 is not None:
            handles.append(mpatches.Patch(facecolor="none", edgecolor=C_ZONE,
                                          linewidth=2, label="Zone d'étude"))
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=C_STATION, markersize=7,
                                   label="Station météo"))
        ax_carte.legend(handles=handles, loc="lower left", fontsize=7,
                        framealpha=0.85, edgecolor=C_GRIS)

        # Éléments cartographiques
        lon_c = (bbox[0] + bbox[2]) / 2
        lat_c = (bbox[1] + bbox[3]) / 2
        span_km = _haversine_km(bbox[1], bbox[0], bbox[1], bbox[2])
        ech_km = max(1, round(span_km / 5))
        _barre_echelle(ax_carte, "EPSG:3857", lon_c, lat_c, longueur_km=ech_km)
        _fleche_nord(ax_carte)

    ax_carte.axis("off")

    # --- Cartouche bas ---
    date_str = date_str or datetime.today().strftime("%d/%m/%Y")

    # Ligne orange de séparation
    fig.add_artist(plt.Line2D([0.0, 1.0], [0.12, 0.12],
                              color=C_ORANGE, linewidth=1.5,
                              transform=fig.transFigure))

    # Sources
    fig.text(0.01, 0.07, f"Sources : {sources}",
             va="center", fontsize=6.5, color=C_GRIS)
    # Auteur / date
    fig.text(0.01, 0.035, f"Auteur : {auteur or 'XXX/YYY'}",
             va="center", fontsize=6.5, color=C_GRIS)
    fig.text(0.18, 0.035, f"Date : {date_str}",
             va="center", fontsize=6.5, color=C_GRIS)

    # Logo
    if logo_bytes:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")
            arr = np.array(img)
            ax_logo = fig.add_axes([0.85, 0.0, 0.14, 0.12])
            ax_logo.imshow(arr)
            ax_logo.axis("off")
        except Exception:
            pass

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def charger_zone_etude(fichier_upload):
    """
    Charge un fichier géographique uploadé (shp zippé, gpkg, geojson).
    Retourne un GeoDataFrame ou None.
    """
    if not CTX_OK:
        return None

    import tempfile, os, zipfile

    nom = fichier_upload.name.lower()
    contenu = fichier_upload.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        chemin = os.path.join(tmpdir, fichier_upload.name)
        with open(chemin, "wb") as f:
            f.write(contenu)

        # Shapefile zippé
        if nom.endswith(".zip"):
            with zipfile.ZipFile(chemin, "r") as z:
                z.extractall(tmpdir)
            shps = [os.path.join(tmpdir, f)
                    for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shps:
                return None
            chemin = shps[0]

        try:
            gdf = gpd.read_file(chemin)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            return gdf
        except Exception:
            return None

"""
Export cartographique statique (PNG et PDF), mise en page A5 paysage.

Fond OpenStreetMap assemblé à partir des tuiles (requests + matplotlib seulement :
ni geopandas ni contextily). Contenu : zone d'étude (GeoJSON) ou, à défaut, le
point d'étude ; stations avec nom et distance ; titre, échelle, nord, sources,
auteur, date, logo.
"""

import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import requests
from PIL import Image

# Charte SCE
C_ORANGE = "#E07020"
C_MARINE = "#1A3A4A"
C_GRIS = "#6D7274"
C_FOND = "#EDEDED"
C_STATION = "#D62728"
C_SYNOP = "#2F6FDB"

FIG_W_IN, FIG_H_IN = 8.27, 5.83          # A5 paysage
AX_RECT = [0.01, 0.12, 0.98, 0.82]       # zone carte (fraction de la figure)
RATIO_CARTE = (AX_RECT[2] * FIG_W_IN) / (AX_RECT[3] * FIG_H_IN)

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
HEADERS = {"User-Agent": "QuickExtractMeteo/1.0 (export cartographique interne)"}
TUILE = 256
LARGEUR_MAX_PX = 1500                     # borne le nombre de tuiles à télécharger


# ==============================================================================
# PROJECTION WEB MERCATOR
# ==============================================================================

def _monde_px(lon, lat, z):
    """Coordonnées pixel dans le monde Web Mercator au zoom z."""
    n = TUILE * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    lat = max(min(lat, 85.0511), -85.0511)
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def _metres_par_px(lat, z):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)


# ==============================================================================
# GEOJSON
# ==============================================================================

def charger_zone_geojson(fichier_upload):
    """GeoJSON uploadé via st.file_uploader -> dict, ou None."""
    if fichier_upload is None:
        return None
    try:
        return json.loads(fichier_upload.read().decode("utf-8"))
    except Exception:
        return None


def _geometries(geojson):
    """Liste de (type, coordonnées) de toutes les géométries d'un GeoJSON."""
    out = []

    def _walk(o):
        if isinstance(o, dict):
            t = o.get("type")
            if t == "FeatureCollection":
                for f in o.get("features", []):
                    _walk(f)
            elif t == "Feature":
                _walk(o.get("geometry"))
            elif t == "GeometryCollection":
                for g in o.get("geometries", []):
                    _walk(g)
            elif t in ("Point", "MultiPoint", "LineString", "MultiLineString",
                       "Polygon", "MultiPolygon"):
                out.append((t, o.get("coordinates")))

    _walk(geojson)
    return out


def _lignes_et_polygones(geojson):
    """Retourne (polygones, lignes, points) ; chaque anneau/ligne = liste de (lon, lat)."""
    polys, lignes, points = [], [], []
    for t, c in _geometries(geojson):
        if t == "Polygon":
            polys.append([[(p[0], p[1]) for p in ring] for ring in c])
        elif t == "MultiPolygon":
            for poly in c:
                polys.append([[(p[0], p[1]) for p in ring] for ring in poly])
        elif t == "LineString":
            lignes.append([(p[0], p[1]) for p in c])
        elif t == "MultiLineString":
            for ln in c:
                lignes.append([(p[0], p[1]) for p in ln])
        elif t == "Point":
            points.append((c[0], c[1]))
        elif t == "MultiPoint":
            points.extend((p[0], p[1]) for p in c)
    return polys, lignes, points


# ==============================================================================
# FOND DE CARTE
# ==============================================================================

def _telecharger_tuile(z, x, y):
    try:
        r = requests.get(TILE_URL.format(z=z, x=x, y=y), headers=HEADERS, timeout=8)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    return None


def _fond_osm(x0, y0, x1, y1, z):
    """Mosaïque de tuiles couvrant [x0,x1]x[y0,y1] (pixels monde). Retourne (image, origine)."""
    tx0, tx1 = int(x0 // TUILE), int(x1 // TUILE)
    ty0, ty1 = int(y0 // TUILE), int(y1 // TUILE)
    n = 2 ** z
    coords = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)
              if 0 <= ty < n]
    mosaique = Image.new("RGB", ((tx1 - tx0 + 1) * TUILE, (ty1 - ty0 + 1) * TUILE),
                         (237, 237, 237))
    with ThreadPoolExecutor(max_workers=8) as ex:
        tuiles = list(ex.map(lambda c: _telecharger_tuile(z, c[0] % n, c[1]), coords))
    ok = 0
    for (tx, ty), img in zip(coords, tuiles):
        if img is not None:
            mosaique.paste(img, ((tx - tx0) * TUILE, (ty - ty0) * TUILE))
            ok += 1
    return (mosaique if ok else None), (tx0 * TUILE, ty0 * TUILE)


# ==============================================================================
# ELEMENTS CARTOGRAPHIQUES
# ==============================================================================

def _echelle(ax, lat, z, x_lim, y_lim):
    """Barre d'échelle en bas à droite, longueur ronde."""
    m_px = _metres_par_px(lat, z)
    largeur_m = (x_lim[1] - x_lim[0]) * m_px
    cible = largeur_m / 5
    base = 10 ** math.floor(math.log10(cible))
    longueur_m = next((k * base for k in (1, 2, 5, 10) if k * base >= cible), base)
    px = longueur_m / m_px
    xr = x_lim[1] - 0.025 * (x_lim[1] - x_lim[0])
    xl = xr - px
    yb = y_lim[0] - 0.05 * (y_lim[0] - y_lim[1])      # y inversé
    h = 0.012 * abs(y_lim[0] - y_lim[1])
    ax.plot([xl, xr], [yb, yb], color="black", linewidth=2.6, solid_capstyle="butt", zorder=9)
    for xx in (xl, xr):
        ax.plot([xx, xx], [yb - h, yb + h], color="black", linewidth=1.4, zorder=9)
    txt = f"{longueur_m / 1000:g} km" if longueur_m >= 1000 else f"{longueur_m:g} m"
    ax.text((xl + xr) / 2, yb - 2.2 * h, txt, ha="center", va="bottom", fontsize=7,
            fontweight="bold", zorder=9,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def _fleche_nord(ax):
    ax.annotate("N", xy=(0.965, 0.90), xycoords="axes fraction", ha="center", va="center",
                fontsize=9, fontweight="bold", color=C_MARINE, zorder=9,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax.annotate("", xy=(0.965, 0.96), xytext=(0.965, 0.925), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=C_MARINE, lw=1.6, mutation_scale=11),
                zorder=9)


# ==============================================================================
# CARTE
# ==============================================================================

def generer_carte_statique(
    df_stations,
    point=None,
    zone_geojson=None,
    df_synop=None,
    titre="",
    logo_bytes=None,
    auteur="",
    sources="Météo-France, OpenStreetMap contributors",
):
    """
    df_stations : NUM_POSTE, NOM_USUEL, LAT, LON, distance_km
    point       : (lat, lon, nom) du site d'étude, dessiné si aucune zone n'est fournie
    zone_geojson: dict GeoJSON (WGS84) de la zone d'étude
    df_synop    : stations SYNOP éventuelles (marqueur d'une autre couleur)

    Retourne (png_bytes, pdf_bytes, fond_ok).
    """
    polys, lignes, pts_zone = ([], [], [])
    if zone_geojson is not None:
        polys, lignes, pts_zone = _lignes_et_polygones(zone_geojson)
    a_zone = bool(polys or lignes or pts_zone)

    stations = []
    for df, couleur in ((df_stations, C_STATION), (df_synop, C_SYNOP)):
        if df is None or len(df) == 0:
            continue
        for _, r in df.iterrows():
            try:
                stations.append((float(r["LON"]), float(r["LAT"]),
                                 str(r.get("NOM_USUEL", r.get("NUM_POSTE", ""))),
                                 r.get("distance_km"), couleur))
            except (TypeError, ValueError, KeyError):
                continue

    lonlat = [(s[0], s[1]) for s in stations]
    if point is not None:
        lonlat.append((point[1], point[0]))
    for poly in polys:
        for ring in poly:
            lonlat.extend(ring)
    for ln in lignes:
        lonlat.extend(ln)
    lonlat.extend(pts_zone)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
    ax = fig.add_axes(AX_RECT)

    fig.text(0.5, 0.99, titre or "Carte météo", ha="center", va="top", fontsize=13,
             fontweight="bold", color=C_MARINE)
    fig.add_artist(plt.Line2D([0.01, 0.99], [0.945, 0.945], color=C_ORANGE, linewidth=2.5,
                              transform=fig.transFigure))

    fond_ok = False
    if not lonlat:
        ax.text(0.5, 0.5, "Aucune donnée spatiale disponible", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
    else:
        lons = [p[0] for p in lonlat]
        lats = [p[1] for p in lonlat]
        lat_c = (min(lats) + max(lats)) / 2
        lon_c = (min(lons) + max(lons)) / 2

        # Étendue en pixels au zoom de référence 18, puis ajustée au ratio de la carte
        zr = 18
        xs, ys = zip(*[_monde_px(lo, la, zr) for lo, la in lonlat])
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        w = max(max(xs) - min(xs), 1.0) * 1.35
        h = max(max(ys) - min(ys), 1.0) * 1.35
        m_ref = _metres_par_px(lat_c, zr)
        w = max(w, 1500 / m_ref)                      # au moins 1,5 km de large
        if w / h < RATIO_CARTE:
            w = h * RATIO_CARTE
        else:
            h = w / RATIO_CARTE

        # Zoom : le plus fin dont la largeur reste sous LARGEUR_MAX_PX
        z = zr
        while z > 3 and w / (2 ** (zr - z)) > LARGEUR_MAX_PX:
            z -= 1
        z = min(z, 17)
        k = 2 ** (z - zr)
        x0, x1 = (cx - w / 2) * k, (cx + w / 2) * k
        y0, y1 = (cy - h / 2) * k, (cy + h / 2) * k

        fond, (ox, oy) = _fond_osm(x0, y0, x1, y1, z)
        if fond is not None:
            fond_ok = True
            ax.imshow(np.asarray(fond), extent=(ox, ox + fond.width, oy + fond.height, oy),
                      zorder=0, interpolation="bilinear")
        else:
            ax.set_facecolor(C_FOND)
            ax.text(0.5, 0.5, "Fond de carte indisponible", ha="center", va="center",
                    transform=ax.transAxes, color=C_GRIS, fontsize=9)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_aspect("equal", adjustable="box")

        def px(lon, lat):
            return _monde_px(lon, lat, z)

        # Zone d'étude
        for poly in polys:
            for i, ring in enumerate(poly):
                xy = np.array([px(lo, la) for lo, la in ring])
                if i == 0:
                    ax.fill(xy[:, 0], xy[:, 1], facecolor=C_ORANGE, alpha=0.12, zorder=3)
                ax.plot(xy[:, 0], xy[:, 1], color=C_ORANGE, linewidth=2.2, zorder=4)
        for ln in lignes:
            xy = np.array([px(lo, la) for lo, la in ln])
            ax.plot(xy[:, 0], xy[:, 1], color=C_ORANGE, linewidth=2.2, zorder=4)
        for lo, la in pts_zone:
            xp, yp = px(lo, la)
            ax.plot(xp, yp, marker="D", color=C_ORANGE, markersize=8, zorder=5,
                    markeredgecolor="white", markeredgewidth=1)

        # Point d'étude (seulement sans zone)
        if point is not None and not a_zone:
            xp, yp = px(point[1], point[0])
            ax.plot(xp, yp, marker="*", color=C_MARINE, markersize=17, zorder=7,
                    markeredgecolor="white", markeredgewidth=1.2)
            if len(point) > 2 and point[2]:
                ax.annotate(str(point[2]), (xp, yp), xytext=(0, 12), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold",
                            color=C_MARINE, zorder=8,
                            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

        # Stations : nom et distance en étiquette
        decalages = [(9, 7, "left", "bottom"), (9, -7, "left", "top"),
                     (-9, 7, "right", "bottom"), (-9, -7, "right", "top")]
        for i, (lo, la, nom, dist, couleur) in enumerate(stations):
            xp, yp = px(lo, la)
            ax.plot(xp, yp, marker="o", color=couleur, markersize=8, zorder=6,
                    markeredgecolor="white", markeredgewidth=1)
            etiquette = nom
            try:
                if dist is not None and not np.isnan(float(dist)):
                    etiquette += f"\n{float(dist):.1f} km".replace(".", ",")
            except (TypeError, ValueError):
                pass
            dx, dy, ha, va = decalages[i % 4]
            ax.annotate(etiquette, (xp, yp), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=7, color=C_MARINE, zorder=8, linespacing=1.15,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

        # Légende
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = []
        if a_zone:
            handles.append(Patch(facecolor=C_ORANGE, alpha=0.3, edgecolor=C_ORANGE,
                                 label="Zone d'étude"))
        elif point is not None:
            handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor=C_MARINE,
                                  markersize=12, label="Point d'étude"))
        if df_stations is not None and len(df_stations):
            handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=C_STATION,
                                  markersize=8, label="Station météo"))
        if df_synop is not None and len(df_synop):
            handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SYNOP,
                                  markersize=8, label="Station SYNOP"))
        if handles:
            ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9,
                      edgecolor=C_GRIS).set_zorder(10)

        _echelle(ax, lat_c, z, (x0, x1), (y1, y0))
        _fleche_nord(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C_GRIS)
            sp.set_linewidth(0.8)

    # Cartouche
    fig.add_artist(plt.Line2D([0.0, 1.0], [0.115, 0.115], color=C_ORANGE, linewidth=1.5,
                              transform=fig.transFigure))
    fig.text(0.01, 0.075, f"Sources : {sources}", va="center", fontsize=6.5, color=C_GRIS)
    fig.text(0.01, 0.04, f"Auteur : {auteur or 'XXX/YYY'}", va="center", fontsize=6.5,
             color=C_GRIS)
    fig.text(0.22, 0.04, f"Date : {datetime.today().strftime('%d/%m/%Y')}", va="center",
             fontsize=6.5, color=C_GRIS)
    if logo_bytes:
        try:
            img = np.array(Image.open(io.BytesIO(logo_bytes)).convert("RGBA"))
            axl = fig.add_axes([0.84, 0.005, 0.15, 0.105])
            axl.imshow(img)
            axl.axis("off")
        except Exception:
            pass

    png, pdf = io.BytesIO(), io.BytesIO()
    fig.savefig(png, format="png", dpi=200, facecolor="white")
    fig.savefig(pdf, format="pdf", facecolor="white")
    plt.close(fig)
    return png.getvalue(), pdf.getvalue(), fond_ok

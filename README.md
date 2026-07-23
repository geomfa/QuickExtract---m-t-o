# QuickExtract Météo

Données météo horaires Météo-France, sans compte. Fraîcheur J-1/J-2.

Complète le bloc SYNOP du pipeline QuickExtract pour les analyses courtes (< 1 an).
Pour les analyses longues (tendances, normales climatiques), utiliser le bloc SYNOP.

---

## Source des données

| Fichier | Source | Fréquence | Usage |
|---------|--------|-----------|-------|
| `H_{dept}_latest-*.csv.gz` | `BASE/HOR/` | Horaire | Observations récentes |
| `Q_{dept}_previous-1950-*_RR-T-Vent.csv.gz` | `BASE/QUOT/` | Quotidien | Normales de référence (10 ans) |

Bucket source : `meteofrance.s3.sbg.io.cloud.ovh.net` — open data, mise à jour quotidienne.

Variables horaires disponibles : température (°C), humidité (%), vitesse et direction
du vent (m/s, °), rafales (m/s), précipitations (mm/h), pression (hPa), nébulosité
(octas), visibilité (m).

---

## Utilisation

### Étape 1 — Zone d'étude

Trois modes de saisie dans la barre latérale :

- **Code INSEE** : résolution automatique des coordonnées et du département
- **Carte** : clic sur la carte Folium, reverse geocoding automatique
- **Coordonnées manuelles** : latitude, longitude, code département

Cliquer sur **1 — Chercher les stations** pour télécharger les données et afficher
le tableau des stations disponibles dans le département.

### Tableau des stations

Pour chaque station (10 plus proches) :

| Colonne | Description |
|---------|-------------|
| Dist. (km) | Distance au point d'étude |
| Dernière mesure | Date et heure de la dernière observation disponible |
| Fraîcheur (j) | Nombre de jours depuis la dernière mesure |
| Nébulosité | Colonne N disponible dans le fichier (oui/non) |
| Historique | Station présente dans le fichier quotidien historique (oui/non) |

> Les stations sans historique (`non`) ne produiront pas de normales de référence
> sur les graphiques.

### Étape 2 — Analyse

Sélectionner les stations et la période dans la barre latérale, puis cliquer sur
**2 — Lancer l'analyse**. La période peut être modifiée et relancée sans
retélécharger les données.

**Fenêtres disponibles** : 24h · 7j · 15j · Personnalisée (avec sélection de l'heure)

### Graphiques produits

| Graphique | 24h | 7j | 15j | < 1 an | ≥ 1 an |
|-----------|:---:|:--:|:---:|:------:|:------:|
| T° / précip (horaire) | oui | oui | — | si ≤ 7j | — |
| Rose des vents + normales | oui | oui | oui | oui | oui |
| Histogramme journalier/mensuel | — | oui | oui | oui | oui |
| Thermopluviogramme mensuel | — | — | — | oui | oui |
| Évolution T° annuelles | — | — | — | — | oui |

Les normales de référence (10 dernières années disponibles) sont superposées sur
chaque graphique quand les données historiques sont disponibles :

- Plage grisée (min–max) + courbe pointillée (moyenne) sur les graphiques de température
- Contour pointillé sur la rose des vents (fréquence directionnelle de référence)

Chaque graphique dispose d'un bouton de téléchargement PNG individuel.

### Export Excel

Onglets selon la période : `Stations` · `Horaire_agrege` · `Horaire_brut` ·
`Journalier` · `Mensuel` · `Annuel` · `Normales`

### Export cartographique

Section en bas de page. Génère une carte HTML interactive (Folium) avec :

- Fond OpenStreetMap
- Zone d'étude en contour orange (upload GeoJSON optionnel)
- Stations en marqueurs avec popup (nom, distance, altitude)
- Cartouche titre / auteur / sources / logo

---

## Installation locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pour changer le port : `streamlit run app.py --server.port 8502`

## Déploiement Streamlit Cloud

```bash
git add app.py mf_client.py graphiques.py carte_folium.py requirements.txt README.md
git commit -m "deploy"
git push
```

Sur [share.streamlit.io](https://share.streamlit.io) : *New app* → repo → `app.py` → *Deploy*.
Aucun secret à configurer.

---

## Fichiers

```
app.py            interface Streamlit (sidebar 2 étapes, graphiques, exports)
mf_client.py      données (téléchargement, inspection stations, agrégation, normales)
graphiques.py     graphiques matplotlib
carte_folium.py   export cartographique HTML
requirements.txt
```

## Dépendances

```
streamlit, pandas, numpy, matplotlib, requests, openpyxl, folium, streamlit-folium
```

Aucune librairie géospatiale lourde (geopandas, contextily, GDAL) — l'app démarre
rapidement sur Streamlit Cloud.

---

*Pour les analyses > 1 an : utiliser le bloc `bloc_meteo_synop.py` du pipeline
QuickExtract (données SYNOP via Opendatasoft).*

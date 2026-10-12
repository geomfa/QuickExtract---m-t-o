# QuickExtract Météo

Application Streamlit d'analyse météo à partir des données ouvertes de Météo-France (aucun compte requis). Elle complète le bloc SYNOP du pipeline QuickExtract : données horaires récentes, normales sur 10 ans, évapotranspiration et intensité des pluies.

## Lancer

```bash
pip install -r requirements.txt
streamlit run app.py
```

Autre port : `streamlit run app.py --server.port 8600`

Déploiement Streamlit Community Cloud : pousser le dossier sur GitHub, créer l'app avec `app.py` comme fichier principal. Aucun secret à configurer.

## Utilisation

1. Définir la zone d'étude dans la barre latérale : code INSEE, clic sur la carte ou coordonnées.
2. Cliquer sur **1 - Chercher les stations**. Les fichiers du département sont téléchargés et les 10 stations les plus proches sont listées (fraîcheur, nébulosité, historique, vent, T° / précipitations, ETP).
3. Cocher les calculs voulus (**Vent**, **T° et précipitations**, **ETP**), puis, dans le tableau, la colonne **Utiliser** pour les stations retenues (les 3 plus proches sont cochées par défaut). Tous les calculs reposent sur les mêmes stations ; une station qui n'a pas une variable ne contribue simplement pas à ce calcul. Les colonnes « dispo » aident à choisir.
4. Choisir la période (24 h, 7 j, 15 j, 1 an, personnalisée avec heures) puis cliquer sur **2 - Lancer l'analyse**.

Les résultats restent figés jusqu'au prochain clic sur le bouton 2. Les fenêtres glissantes se terminent à la dernière mesure du fichier horaire (J-1 / J-2).

## Sorties

| Période | Graphiques |
|---|---|
| 24 h, 7 j, personnalisée jusqu'à 7 j | T° et précipitations horaires |
| 7 j, 15 j, personnalisée jusqu'à 31 j | suivi journalier, ETP journalière |
| Plus de 31 j | thermopluviogramme (mois complets), bilan normale / observé |
| Toutes | rose des vents (période et dernier mois), ETP et bilan hydrique mensuels, intensité des pluies |
| Au moins une année civile complète | évolution des températures annuelles |

Exports : PNG par graphique, Excel (stations utilisées, données horaires, journalières, mensuelles, normales, ETP), carte HTML avec zone d'étude, logo, titre et auteur.

## Sources et règles de calcul

- Observations : `BASE/HOR` (fichier horaire départemental).
- Normales : `BASE/QUOT` `RR-T-Vent`, 10 dernières années.
- ETP Penman-Monteith : `BASE/QUOT` `autres-parametres` (historique et récent). Si la donnée manque, l'app affiche « données non disponibles » ; aucun calcul de repli.
- SYNOP (Opendatasoft) : utilisé uniquement si la période commence avant le début du fichier horaire. Les trois stations SYNOP les plus proches remplacent alors les stations cochées pour les observations ; les stations cochées servent aux normales et à l'ETP (message affiché dans l'app).
- Un mois n'entre dans une normale ou un bilan que s'il est renseigné à 90 % au moins. Les mois partiels en bord de période sont écartés.
- Précipitations et ETP sont sommées ; températures moyennées. Les normales sont calculées station par station puis moyennées, pondérées par l'inverse de la distance.
- La direction du vent est moyennée vectoriellement. Il n'y a pas de normale de vent.
- Intensité des pluies : indicateurs descriptifs sur la moyenne horaire des stations, seuils indicatifs 2,5 / 7,6 / 15 mm/h. Non calculée avec SYNOP (pas de 3 h).

## Mode debug

Dans la barre latérale, afficher la structure des fichiers horaire et ETP pour vérifier les noms de colonnes réels (la colonne ETP est recherchée parmi `ETPMON`, `ETP`, `ETPP`, `ETPPM`, `ETPGRILLE`, puis toute colonne commençant par `ETP`).

## Fichiers

| Fichier | Rôle |
|---|---|
| `app.py` | interface et enchaînement des calculs |
| `mf_client.py` | téléchargements, stations, agrégations, normales, ETP, intensité |
| `synop_client.py` | stations et observations SYNOP |
| `graphiques.py` | graphiques matplotlib |
| `carte_folium.py` | export cartographique |
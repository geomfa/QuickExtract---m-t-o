"""
Graphiques météo statiques (matplotlib).
Chaque fonction retourne une figure matplotlib prête pour st.pyplot().

Les normales de référence (n dernières années disponibles) sont superposées
quand fournies :
  - plage grisée (min-max) + courbe pointillée (moyenne) sur les températures
  - contour pointillé sur la rose des vents
Le texte exact de la légende ("Normale 2015-2024" par ex.) est calculé
dynamiquement en amont (app.py) à partir des années réellement couvertes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

SOURCE_LABEL = "Source : Météo-France — Données climatologiques de base (data.gouv.fr)"
MOIS_LABELS  = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                 "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

# Palette — décalée des teintes matplotlib/IA par défaut tout en restant
# lumineuse et saturée (évite le style "diagramme entreprise" terne).
C_MOY     = "#FF7A1A"   # orange vif — T° moyenne / accents principaux
C_MAX     = "#E63950"   # corail vif — T° max
C_MIN     = "#0FA3A3"   # teal vif — T° min
C_PRECIP  = "#2F8FEF"   # bleu lumineux — précipitations

DEFAULT_LABEL_NORMALE      = "Normale (réf.)"
DEFAULT_LABEL_NORMALE_VENT = "Normale directionnelle (réf.)"


def _source(fig):
    fig.text(0.5, 0.01, SOURCE_LABEL, ha="center", fontsize=8, style="italic",
             color="#555555")


def _fmt_axe_dates(ax, n_points):
    if n_points <= 48:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    elif n_points <= 360:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


# ==============================================================================
# 1. EVOLUTION TEMPERATURE / PRECIPITATIONS (vue horaire courte)
# ==============================================================================

def graphique_temp_precip(df_agg, titre="Évolution température / précipitations",
                           df_normales=None, label_normale=DEFAULT_LABEL_NORMALE):
    """
    Courbe T° (axe gauche) + barres précip (axe droit), vue horaire.
    df_normales : DataFrame avec mois, t_moy_norm, t_min_norm, t_max_norm (optionnel).
    """
    fig, ax1 = plt.subplots(figsize=(13, 5))

    ax1.plot(df_agg["date_dt"], df_agg["t_celsius"],
             color=C_MAX, linewidth=1.6, label="T° observée (°C)")
    ax1.set_ylabel("Température (°C)", color=C_MAX, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=C_MAX)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    if df_normales is not None and not df_normales.empty \
            and "t_moy_norm" in df_normales.columns:
        df_plot = df_agg.copy()
        df_plot["mois"] = df_plot["date_dt"].dt.month
        df_plot = df_plot.merge(df_normales[
            ["mois"] + [c for c in ["t_moy_norm", "t_min_norm", "t_max_norm"]
                        if c in df_normales.columns]
        ], on="mois", how="left")

        ax1.plot(df_plot["date_dt"], df_plot["t_moy_norm"],
                 color=C_NORM, linewidth=1.2, linestyle="--",
                 label=f"T° moy. {label_normale}", zorder=2)
        if "t_min_norm" in df_plot.columns and "t_max_norm" in df_plot.columns:
            ax1.fill_between(df_plot["date_dt"],
                             df_plot["t_min_norm"], df_plot["t_max_norm"],
                             color=C_NORM, alpha=0.15, zorder=1,
                             label=f"Plage {label_normale}")

    if "rr1_mm" in df_agg.columns and df_agg["rr1_mm"].notna().any():
        ax2 = ax1.twinx()
        ax2.bar(df_agg["date_dt"], df_agg["rr1_mm"],
                width=0.03, color=C_PRECIP, alpha=0.55, label="Précip. (mm/h)")
        ax2.set_ylabel("Précipitations (mm/h)", color=C_PRECIP, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=C_PRECIP)
        ax2.set_ylim(bottom=0)
        lines1, lbl1 = ax1.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="upper left", fontsize=9)
    else:
        ax1.legend(loc="upper left", fontsize=9)

    ax1.set_title(titre, fontsize=12, fontweight="bold")
    _fmt_axe_dates(ax1, len(df_agg))
    _source(fig)
    plt.tight_layout()
    return fig


# ==============================================================================
# 2. ROSE DES VENTS
# ==============================================================================

def graphique_rose_vents(df_agg, titre="Rose des vents", n_secteurs=16,
                         df_normales_vent=None,
                         label_normale=DEFAULT_LABEL_NORMALE_VENT):
    """
    Rose des vents polaire colorée par classe de vitesse (dégradé teal).
    Si df_normales_vent est fourni (colonnes secteur, freq_norm),
    superpose la fréquence directionnelle de référence en contour pointillé.
    """
    df = df_agg[["ff_ms", "dd_deg"]].dropna()
    if df.empty or len(df) < 3:
        return None

    bins_v   = [0, 2, 5, 8, 11, 17, np.inf]
    labels_v = ["0-2 m/s", "2-5 m/s", "5-8 m/s", "8-11 m/s", "11-17 m/s", ">17 m/s"]
    # Dégradé teal lumineux — se démarque du bleu générique habituel
    couleurs = ["#D6F5EC", "#8FE3CF", "#4FCDAE", "#1FB08C", "#0E8F6F", "#066352"]

    sec_deg = 360 / n_secteurs
    df = df.copy()
    df["secteur"] = ((df["dd_deg"] + sec_deg / 2) % 360 // sec_deg).astype(int).clip(0, n_secteurs - 1)
    df["cat_v"]   = pd.cut(df["ff_ms"], bins=bins_v, labels=labels_v, right=False)
    theta = np.linspace(0, 2 * np.pi, n_secteurs, endpoint=False)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    bottom = np.zeros(n_secteurs)

    for cat, couleur in zip(labels_v, couleurs):
        counts = (
            df[df["cat_v"] == cat].groupby("secteur").size()
            .reindex(range(n_secteurs), fill_value=0).values
        )
        freqs = counts / len(df) * 100
        ax.bar(theta, freqs, width=np.radians(sec_deg) * 0.88,
               bottom=bottom, color=couleur, label=cat, alpha=0.95)
        bottom += freqs

    if df_normales_vent is not None and not df_normales_vent.empty:
        freq_norm = (
            df_normales_vent
            .set_index("secteur")["freq_norm"]
            .reindex(range(n_secteurs), fill_value=0)
            .values
        )
        theta_closed = np.append(theta, theta[0])
        freq_closed  = np.append(freq_norm, freq_norm[0])
        ax.plot(theta_closed, freq_closed,
                color=C_NORM, linewidth=1.8, linestyle="--",
                label=label_normale, zorder=5)

    labels_dir = (
        ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
         "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
        if n_secteurs == 16 else
        ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    )
    ax.set_xticks(theta)
    ax.set_xticklabels(labels_dir[:n_secteurs], fontsize=9)
    ax.set_ylabel("Fréquence (%)", labelpad=30, fontsize=9)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.18, -0.16),
              title="Vitesse", fontsize=8, title_fontsize=9)
    ax.set_title(titre, fontsize=11, fontweight="bold", pad=18)
    fig.text(0.5, 0.01, SOURCE_LABEL, ha="center", fontsize=8, style="italic")
    plt.tight_layout()
    return fig


# ==============================================================================
# 3. THERMOPLUVIOGRAMME — bilan mensuel (climatologie de la période sélectionnée)
# ==============================================================================

def graphique_thermopluviogramme(df_mensuel, titre="Thermopluviogramme — bilan mensuel",
                                  df_normales=None,
                                  label_normale=DEFAULT_LABEL_NORMALE):
    """
    Diagramme ombrothermique mensuel : moyenne des mois observés sur la
    période sélectionnée (barres pleines + courbes), comparée à la normale
    de référence (barres et courbe grisées en retrait) si disponible.
    """
    if df_mensuel.empty:
        return None

    x = np.arange(len(df_mensuel))
    fig, ax1 = plt.subplots(figsize=(13, 6.5))
    ax2 = ax1.twinx()

    if df_normales is not None and not df_normales.empty \
            and "t_moy_norm" in df_normales.columns:
        norm = df_mensuel.merge(
            df_normales[["mois"] + [c for c in
                ["t_moy_norm", "t_min_norm", "t_max_norm", "precip_norm"]
                if c in df_normales.columns]],
            on="mois", how="left"
        )
        if "t_min_norm" in norm.columns and "t_max_norm" in norm.columns:
            ax1.fill_between(x, norm["t_min_norm"], norm["t_max_norm"],
                             color=C_NORM, alpha=0.13, zorder=1)
        ax1.plot(x, norm["t_moy_norm"], color=C_NORM, linewidth=1.6,
                 linestyle="--", marker="o", markersize=3,
                 label=f"T° moy. {label_normale}", zorder=2)
        if "precip_norm" in norm.columns and norm["precip_norm"].notna().any():
            ax2.bar(x - 0.16, norm["precip_norm"].fillna(0),
                    color=C_NORM, alpha=0.35, width=0.30,
                    label=f"Précip. {label_normale}", zorder=2)

    p = df_mensuel["precip_tot"].fillna(0).values if "precip_tot" in df_mensuel.columns \
        else np.zeros(len(df_mensuel))
    ax2.bar(x + 0.16, p, color=C_PRECIP, alpha=0.65, width=0.30,
            label="Précip. observées (mm)", zorder=3)
    ax2.set_ylabel("Précipitations (mm/mois)", color=C_PRECIP, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=C_PRECIP)
    ax2.set_ylim(0, max(p.max() if len(p) else 1, 1) * 2.6)

    ax1.plot(x, df_mensuel["t_max"], color=C_MAX, linewidth=2,
             marker="o", markersize=5, label="T° max observée", zorder=4)
    ax1.plot(x, df_mensuel["t_moy"], color=C_MOY, linewidth=2.5,
             marker="o", markersize=5, label="T° moy. observée", zorder=4)
    ax1.plot(x, df_mensuel["t_min"], color=C_MIN, linewidth=2,
             marker="o", markersize=5, label="T° min observée", zorder=4)
    ax1.fill_between(x, df_mensuel["t_min"], df_mensuel["t_max"],
                     alpha=0.08, color=C_MOY, zorder=1)

    ax1.set_ylabel("Température (°C)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MOIS_LABELS[:len(df_mensuel)], fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2,
               loc="upper left", fontsize=9, framealpha=0.92, ncol=2)
    ax1.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig


# ==============================================================================
# 4. SUIVI JOURNALIER — précipitations et températures (fenêtres courtes)
# ==============================================================================

def graphique_histogramme_periode(df, titre="Précipitations et températures — suivi journalier",
                                   df_normales=None,
                                   label_normale=DEFAULT_LABEL_NORMALE):
    """
    Vue journalière (7j, 15j, ou période personnalisée courte) : une barre/
    point par jour. Distincte du thermopluviogramme qui montre une
    climatologie mensuelle sur des périodes plus longues.
    """
    if df.empty or "t_moy" not in df.columns:
        return None

    n = len(df)
    fig, ax1 = plt.subplots(figsize=(13, 5))
    x = np.arange(n)

    if df_normales is not None and not df_normales.empty \
            and "mois" in df.columns and "t_moy_norm" in df_normales.columns:
        df_m = df.merge(
            df_normales[["mois"] + [c for c in
                ["t_moy_norm", "t_min_norm", "t_max_norm"]
                if c in df_normales.columns]],
            on="mois", how="left"
        )
        if "t_min_norm" in df_m.columns and "t_max_norm" in df_m.columns:
            ax1.fill_between(x, df_m["t_min_norm"], df_m["t_max_norm"],
                             color=C_NORM, alpha=0.15, zorder=1,
                             label=f"Plage {label_normale}")
        if "t_moy_norm" in df_m.columns:
            ax1.plot(x, df_m["t_moy_norm"], color=C_NORM, linewidth=1.2,
                     linestyle="--", label=f"T° moy. {label_normale}", zorder=2)

    if "precip_tot" in df.columns and df["precip_tot"].notna().any():
        ax2 = ax1.twinx()
        ax2.bar(x, df["precip_tot"].fillna(0), color=C_PRECIP, alpha=0.6,
                label="Précip. (mm)", zorder=2)
        ax2.set_ylabel("Précipitations (mm)", color=C_PRECIP, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=C_PRECIP)
        ax2.set_ylim(bottom=0)

    ax1.plot(x, df["t_moy"], color=C_MOY, linewidth=2.2,
             marker="o", markersize=5, label="T° moy", zorder=3)
    if "t_min" in df.columns and "t_max" in df.columns:
        ax1.plot(x, df["t_max"], color=C_MAX, linewidth=1.5,
                 marker="o", markersize=4, label="T° max", zorder=3)
        ax1.plot(x, df["t_min"], color=C_MIN, linewidth=1.5,
                 marker="o", markersize=4, label="T° min", zorder=3)
        ax1.fill_between(x, df["t_min"], df["t_max"],
                         alpha=0.08, color=C_MOY, zorder=1)

    ax1.set_ylabel("Température (°C)", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    if "date_dt" in df.columns:
        pas = max(1, n // 12)
        fmt = "%d/%m" if n <= 62 else "%m/%Y"
        labels = [pd.Timestamp(d).strftime(fmt) for d in df["date_dt"]]
        ax1.set_xticks(x[::pas])
        ax1.set_xticklabels(labels[::pas], rotation=35, ha="right", fontsize=9)

    lines1, lbl1 = ax1.get_legend_handles_labels()
    if "precip_tot" in df.columns and df["precip_tot"].notna().any():
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="upper left", fontsize=9)
    else:
        ax1.legend(lines1, lbl1, loc="upper left", fontsize=9)

    ax1.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig


# ==============================================================================
# 5. EVOLUTION TEMPERATURES ANNUELLES
# ==============================================================================

def graphique_temperatures_annuelles(df_annuel, titre="Évolution des températures annuelles",
                                      df_normales=None,
                                      label_normale=DEFAULT_LABEL_NORMALE):
    if df_annuel.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    x = df_annuel["annee"].values

    if df_normales is not None and not df_normales.empty \
            and "t_moy_norm" in df_normales.columns:
        t_moy_n = df_normales["t_moy_norm"].mean()
        t_min_n = df_normales["t_min_norm"].mean() if "t_min_norm" in df_normales.columns else None
        t_max_n = df_normales["t_max_norm"].mean() if "t_max_norm" in df_normales.columns else None
        ax.axhline(t_moy_n, color=C_NORM, linewidth=1.2, linestyle="--",
                   label=f"T° moy. {label_normale}", zorder=2)
        if t_min_n is not None and t_max_n is not None:
            ax.axhspan(t_min_n, t_max_n, color=C_NORM, alpha=0.1, zorder=1,
                       label=f"Plage {label_normale}")

    ax.fill_between(x, df_annuel["t_min"], df_annuel["t_max"],
                    alpha=0.12, color=C_MOY, label="Amplitude T°min-T°max")
    ax.plot(x, df_annuel["t_max"], color=C_MAX, linewidth=2,
            marker="o", markersize=5, label="T° max moy. annuelle")
    ax.plot(x, df_annuel["t_moy"], color=C_MOY, linewidth=2.5,
            marker="o", markersize=6, label="T° moyenne annuelle")
    ax.plot(x, df_annuel["t_min"], color=C_MIN, linewidth=2,
            marker="o", markersize=5, label="T° min moy. annuelle")

    ax.set_xlabel("Année", fontsize=10)
    ax.set_ylabel("Température (°C)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in x], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.92)
    ax.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig

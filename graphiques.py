"""
Graphiques météo statiques (matplotlib).
Chaque fonction retourne une figure matplotlib prête pour st.pyplot().
Les normales décennales (1991-2020) sont superposées quand disponibles :
  - plage grisée (t_min_norm / t_max_norm)
  - courbe en pointillés (t_moy_norm)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

SOURCE_LABEL = "Source : Météo-France — Données climatologiques de base (data.gouv.fr)"
MOIS_LABELS  = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                 "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

# Couleurs SCE
C_ORANGE  = "#E07020"
C_BLEU    = "#4A90D9"
C_ROUGE   = "#D62728"
C_VERT    = "#2CA02C"
C_NORM    = "#888888"   # couleur des normales décennales


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


def _ajouter_legende_normales(ax, label="Normales 1991-2020"):
    """Ajoute une entrée de légende pour la plage des normales."""
    patch = mpatches.Patch(color=C_NORM, alpha=0.18, label=label)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [patch], labels=labels + [label],
              loc="upper left", fontsize=9, framealpha=0.9)


# ==============================================================================
# 1. EVOLUTION TEMPERATURE / PRECIPITATIONS
# ==============================================================================

def graphique_temp_precip(df_agg, titre="Évolution température / précipitations",
                           df_normales=None):
    """
    Courbe T° (axe gauche) + barres précip (axe droit).
    df_normales : DataFrame avec mois, t_moy_norm, t_min_norm, t_max_norm (optionnel).
    """
    fig, ax1 = plt.subplots(figsize=(13, 5))

    ax1.plot(df_agg["date_dt"], df_agg["t_celsius"],
             color=C_ROUGE, linewidth=1.6, label="T° (°C)")
    ax1.set_ylabel("Température (°C)", color=C_ROUGE, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=C_ROUGE)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Normales décennales sur la courbe T° : plage mensuelle projetée sur la série
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
                 label="Normale moy. 1991-2020", zorder=2)
        if "t_min_norm" in df_plot.columns and "t_max_norm" in df_plot.columns:
            ax1.fill_between(df_plot["date_dt"],
                             df_plot["t_min_norm"], df_plot["t_max_norm"],
                             color=C_NORM, alpha=0.15, zorder=1,
                             label="Plage normale 1991-2020")

    if "rr1_mm" in df_agg.columns and df_agg["rr1_mm"].notna().any():
        ax2 = ax1.twinx()
        ax2.bar(df_agg["date_dt"], df_agg["rr1_mm"],
                width=0.03, color=C_BLEU, alpha=0.55, label="Précip. (mm/h)")
        ax2.set_ylabel("Précipitations (mm/h)", color=C_BLEU, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=C_BLEU)
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

def graphique_rose_vents(df_agg, titre="Rose des vents", n_secteurs=16):
    df = df_agg[["ff_ms", "dd_deg"]].dropna()
    if df.empty or len(df) < 3:
        return None

    bins_v   = [0, 2, 5, 8, 11, 17, np.inf]
    labels_v = ["0-2 m/s", "2-5 m/s", "5-8 m/s", "8-11 m/s", "11-17 m/s", ">17 m/s"]
    couleurs = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C", "#08306B"]

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
               bottom=bottom, color=couleur, label=cat, alpha=0.92)
        bottom += freqs

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
# 3. THERMOPLUVIOGRAMME
# ==============================================================================

def graphique_thermopluviogramme(df_mensuel, titre="Thermopluviogramme",
                                  df_normales=None):
    if df_mensuel.empty:
        return None

    x = np.arange(len(df_mensuel))
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    # Normales en fond
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
                             color=C_NORM, alpha=0.15, zorder=1)
        ax1.plot(x, norm["t_moy_norm"], color=C_NORM, linewidth=1.5,
                 linestyle="--", label="T° norm. 1991-2020", zorder=2)
        if "precip_norm" in norm.columns and norm["precip_norm"].notna().any():
            ax2.bar(x - 0.15, norm["precip_norm"].fillna(0),
                    color=C_NORM, alpha=0.25, width=0.28,
                    label="Précip. norm. 1991-2020", zorder=2)

    # Précipitations observées
    p = df_mensuel["precip_tot"].fillna(0).values if "precip_tot" in df_mensuel.columns \
        else np.zeros(len(df_mensuel))
    ax2.bar(x + 0.15, p, color=C_BLEU, alpha=0.55, width=0.28,
            label="Précip. observées (mm)", zorder=3)
    ax2.set_ylabel("Précipitations (mm/mois)", color=C_BLEU, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=C_BLEU)
    ax2.set_ylim(0, max(p.max() if len(p) else 1, 1) * 2.8)

    # Températures observées
    ax1.plot(x, df_mensuel["t_max"], color=C_ROUGE, linewidth=2,
             marker="o", markersize=5, label="T° max", zorder=4)
    ax1.plot(x, df_mensuel["t_moy"], color=C_ORANGE, linewidth=2.5,
             marker="o", markersize=5, label="T° moy.", zorder=4)
    ax1.plot(x, df_mensuel["t_min"], color="#1F77B4", linewidth=2,
             marker="o", markersize=5, label="T° min", zorder=4)
    ax1.fill_between(x, df_mensuel["t_min"], df_mensuel["t_max"],
                     alpha=0.07, color=C_ORANGE, zorder=1)

    ax1.set_ylabel("Température (°C)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MOIS_LABELS[:len(df_mensuel)], fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2,
               loc="upper left", fontsize=9, framealpha=0.9)
    ax1.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig


# ==============================================================================
# 4. HISTOGRAMME JOURNALIER / MENSUEL
# ==============================================================================

def graphique_histogramme_periode(df, titre="Précipitations et températures",
                                   df_normales=None):
    if df.empty or "t_moy" not in df.columns:
        return None

    n = len(df)
    fig, ax1 = plt.subplots(figsize=(13, 5))
    x = np.arange(n)

    # Normales superposées si disponibles et si df est indexé par mois
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
                             label="Plage normale 1991-2020")
        if "t_moy_norm" in df_m.columns:
            ax1.plot(x, df_m["t_moy_norm"], color=C_NORM, linewidth=1.2,
                     linestyle="--", label="T° norm. 1991-2020", zorder=2)

    # Précipitations
    if "precip_tot" in df.columns and df["precip_tot"].notna().any():
        ax2 = ax1.twinx()
        ax2.bar(x, df["precip_tot"].fillna(0), color=C_BLEU, alpha=0.55,
                label="Précip. (mm)", zorder=2)
        ax2.set_ylabel("Précipitations (mm)", color=C_BLEU, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=C_BLEU)
        ax2.set_ylim(bottom=0)

    # Températures
    ax1.plot(x, df["t_moy"], color=C_ORANGE, linewidth=2.2,
             marker="o", markersize=5, label="T° moy", zorder=3)
    if "t_min" in df.columns and "t_max" in df.columns:
        ax1.plot(x, df["t_max"], color=C_ROUGE, linewidth=1.5,
                 marker="o", markersize=4, label="T° max", zorder=3)
        ax1.plot(x, df["t_min"], color="#1F77B4", linewidth=1.5,
                 marker="o", markersize=4, label="T° min", zorder=3)
        ax1.fill_between(x, df["t_min"], df["t_max"],
                         alpha=0.08, color=C_ORANGE, zorder=1)

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
                                      df_normales=None):
    if df_annuel.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    x = df_annuel["annee"].values

    # Normales horizontales (valeur unique sur toute la période)
    if df_normales is not None and not df_normales.empty \
            and "t_moy_norm" in df_normales.columns:
        t_moy_n = df_normales["t_moy_norm"].mean()
        t_min_n = df_normales["t_min_norm"].mean() if "t_min_norm" in df_normales.columns else None
        t_max_n = df_normales["t_max_norm"].mean() if "t_max_norm" in df_normales.columns else None
        ax.axhline(t_moy_n, color=C_NORM, linewidth=1.2, linestyle="--",
                   label="T° moy. normale 1991-2020", zorder=2)
        if t_min_n is not None and t_max_n is not None:
            ax.axhspan(t_min_n, t_max_n, color=C_NORM, alpha=0.12, zorder=1,
                       label="Plage normale 1991-2020")

    ax.fill_between(x, df_annuel["t_min"], df_annuel["t_max"],
                    alpha=0.13, color=C_ORANGE, label="Amplitude T°min-T°max")
    ax.plot(x, df_annuel["t_max"], color=C_ROUGE, linewidth=2,
            marker="o", markersize=5, label="T° max moy. annuelle")
    ax.plot(x, df_annuel["t_moy"], color=C_ORANGE, linewidth=2.5,
            marker="o", markersize=6, label="T° moyenne annuelle")
    ax.plot(x, df_annuel["t_min"], color="#1F77B4", linewidth=2,
            marker="o", markersize=5, label="T° min moy. annuelle")

    ax.set_xlabel("Année", fontsize=10)
    ax.set_ylabel("Température (°C)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in x], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig

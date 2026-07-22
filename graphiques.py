"""
Graphiques météo statiques (matplotlib).
Tous les graphiques retournent une figure matplotlib prête à passer à st.pyplot().
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SOURCE_LABEL = "Source : Météo-France — Données climatologiques de base horaires (data.gouv.fr)"
MOIS_LABELS  = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                 "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]


def _source(fig):
    fig.text(0.5, 0.01, SOURCE_LABEL, ha="center", fontsize=8, style="italic")


def _fmt_axe_dates(ax, n_points):
    """Formate l'axe des dates selon la densité de points."""
    if n_points <= 48:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    elif n_points <= 360:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


# ==============================================================================
# 1. EVOLUTION TEMPERATURE / PRECIPITATIONS
# ==============================================================================

def graphique_temp_precip(df_agg, titre="Evolution température / précipitations"):
    """
    Courbe de température (axe gauche) + barres de précipitations horaires (axe droit).
    df_agg doit avoir : date_dt, t_celsius, rr1_mm (optionnel).
    """
    fig, ax1 = plt.subplots(figsize=(13, 5))

    ax1.plot(
        df_agg["date_dt"], df_agg["t_celsius"],
        color="#D62728", linewidth=1.6, label="T° (°C)",
    )
    ax1.set_ylabel("Température (°C)", color="#D62728", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#D62728")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    if "rr1_mm" in df_agg.columns and df_agg["rr1_mm"].notna().any():
        ax2 = ax1.twinx()
        ax2.bar(
            df_agg["date_dt"], df_agg["rr1_mm"],
            width=0.03, color="#4A90D9", alpha=0.55, label="Précip. (mm/h)",
        )
        ax2.set_ylabel("Précipitations (mm/h)", color="#4A90D9", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#4A90D9")
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
    """
    Rose des vents polaire colorée par classe de vitesse.
    df_agg doit avoir : ff_ms, dd_deg.
    """
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
            df[df["cat_v"] == cat]
            .groupby("secteur")
            .size()
            .reindex(range(n_secteurs), fill_value=0)
            .values
        )
        freqs = counts / len(df) * 100
        ax.bar(
            theta, freqs,
            width=np.radians(sec_deg) * 0.88,
            bottom=bottom, color=couleur, label=cat, alpha=0.92,
        )
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
    ax.legend(
        loc="lower left", bbox_to_anchor=(-0.18, -0.16),
        title="Vitesse", fontsize=8, title_fontsize=9,
    )
    ax.set_title(titre, fontsize=11, fontweight="bold", pad=18)
    fig.text(0.5, 0.01, SOURCE_LABEL, ha="center", fontsize=8, style="italic")
    plt.tight_layout()
    return fig


# ==============================================================================
# 3. THERMOPLUVIOGRAMME (normales mensuelles)
# ==============================================================================

def graphique_thermopluviogramme(df_mensuel, titre="Thermopluviogramme — normales mensuelles"):
    """
    Diagramme ombrothermique : barres précipitations (axe droit) + courbes T°.
    df_mensuel doit avoir : mois (1-12), t_moy, t_min, t_max, precip_tot.
    """
    if df_mensuel.empty:
        return None

    x = np.arange(len(df_mensuel))
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    # Precipitations
    p = df_mensuel["precip_tot"].fillna(0).values
    ax2.bar(x, p, color="#4A90D9", alpha=0.45, width=0.55, label="Précip. (mm/mois)", zorder=2)
    ax2.set_ylabel("Précipitations (mm/mois)", color="#4A90D9", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#4A90D9")
    ax2.set_ylim(0, max(p.max(), 1) * 2.8)

    # Temperatures
    ax1.plot(x, df_mensuel["t_max"], color="#D62728", linewidth=2,
             marker="o", markersize=5, label="T° max moy.", zorder=3)
    ax1.plot(x, df_mensuel["t_moy"], color="#FF7F0E", linewidth=2.5,
             marker="o", markersize=5, label="T° moyenne", zorder=3)
    ax1.plot(x, df_mensuel["t_min"], color="#1F77B4", linewidth=2,
             marker="o", markersize=5, label="T° min moy.", zorder=3)
    ax1.fill_between(x, df_mensuel["t_min"], df_mensuel["t_max"],
                     alpha=0.07, color="#FF7F0E", zorder=1)

    ax1.set_ylabel("Température (°C)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MOIS_LABELS[:len(df_mensuel)], fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="upper left", fontsize=9, framealpha=0.9)

    ax1.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig


# ==============================================================================
# 4. HISTOGRAMME PRECIP + TEMPERATURE (fenetre adaptative)
# ==============================================================================

def graphique_histogramme_periode(df, titre="Précipitations et températures"):
    """
    Histogramme journalier ou mensuel adapté à la fenêtre temporelle.
    df doit avoir : date_dt, t_moy, t_min (optionnel), t_max (optionnel),
                    precip_tot (optionnel).
    """
    if df.empty or "t_moy" not in df.columns:
        return None

    n = len(df)
    fig, ax1 = plt.subplots(figsize=(13, 5))
    x = np.arange(n)

    # Precipitations (axe droit)
    if "precip_tot" in df.columns and df["precip_tot"].notna().any():
        ax2 = ax1.twinx()
        ax2.bar(x, df["precip_tot"].fillna(0), color="#4A90D9", alpha=0.55,
                label="Précip. (mm)", zorder=2)
        ax2.set_ylabel("Précipitations (mm)", color="#4A90D9", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#4A90D9")
        ax2.set_ylim(bottom=0)

    # Temperatures (axe gauche)
    ax1.plot(x, df["t_moy"], color="#FF7F0E", linewidth=2.2,
             marker="o", markersize=5, label="T° moy", zorder=3)
    if "t_min" in df.columns and "t_max" in df.columns:
        ax1.plot(x, df["t_max"], color="#D62728", linewidth=1.5,
                 marker="o", markersize=4, label="T° max", zorder=3)
        ax1.plot(x, df["t_min"], color="#1F77B4", linewidth=1.5,
                 marker="o", markersize=4, label="T° min", zorder=3)
        ax1.fill_between(x, df["t_min"], df["t_max"],
                         alpha=0.08, color="#FF7F0E", zorder=1)

    ax1.set_ylabel("Température (°C)", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Labels axe x : format adapté + pas d'affichage si trop dense
    if "date_dt" in df.columns:
        pas = max(1, n // 12)
        fmt = "%d/%m" if n <= 62 else "%m/%Y"
        labels = [pd.Timestamp(d).strftime(fmt) for d in df["date_dt"]]
        ax1.set_xticks(x[::pas])
        ax1.set_xticklabels(labels[::pas], rotation=35, ha="right", fontsize=9)

    # Legende combinee
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

def graphique_temperatures_annuelles(df_annuel, titre="Evolution des températures annuelles"):
    """
    Courbes T°min/moy/max par année + plage amplitude.
    df_annuel doit avoir : annee, t_moy, t_min, t_max.
    """
    if df_annuel.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    x = df_annuel["annee"].values

    ax.fill_between(x, df_annuel["t_min"], df_annuel["t_max"],
                    alpha=0.13, color="#FF7F0E", label="Amplitude T°min-T°max")
    ax.plot(x, df_annuel["t_max"], color="#D62728", linewidth=2,
            marker="o", markersize=5, label="T° max moy. annuelle")
    ax.plot(x, df_annuel["t_moy"], color="#FF7F0E", linewidth=2.5,
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

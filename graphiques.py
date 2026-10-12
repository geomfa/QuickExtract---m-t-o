"""
Graphiques météo statiques (matplotlib).
Chaque fonction retourne une figure prête pour st.pyplot(), ou None si les
données sont insuffisantes.

Les normales (n dernières années disponibles) sont superposées quand elles sont
fournies. La rose des vents ne montre que les observations : il n'existe pas de
normale directionnelle exploitable dans les sources utilisées.
"""

import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SOURCE_LABEL = "Source : Météo-France — Données climatologiques de base (data.gouv.fr)"
MOIS_LABELS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
               "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

C_MOY = "#FF7A1A"      # T° moyenne
C_MAX = "#E63950"      # T° max
C_MIN = "#0FA3A3"      # T° min
C_PRECIP = "#2F8FEF"   # précipitations
C_NORM = "#8C7A66"     # normales
C_ETP = "#4CAF50"      # évapotranspiration
C_BILAN = "#7B3FA0"    # bilan hydrique P - ETP

DEFAULT_LABEL_NORMALE = "Normale (réf.)"


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
# 1. TEMPERATURE / PRECIPITATIONS, VUE HORAIRE
# ==============================================================================

def graphique_temp_precip(df_agg, titre="Évolution température / précipitations",
                          df_normales=None, label_normale=DEFAULT_LABEL_NORMALE):
    if df_agg is None or df_agg.empty or "t_celsius" not in df_agg.columns:
        return None
    fig, ax1 = plt.subplots(figsize=(13, 5))

    ax1.plot(df_agg["date_dt"], df_agg["t_celsius"],
             color=C_MAX, linewidth=1.6, label="T° observée (°C)")
    ax1.set_ylabel("Température (°C)", color=C_MAX, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=C_MAX)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    if df_normales is not None and not df_normales.empty \
            and "t_moy_norm" in df_normales.columns:
        df_plot = df_agg[["date_dt"]].copy()
        df_plot["mois"] = df_plot["date_dt"].dt.month
        cols = [c for c in ["t_moy_norm", "t_min_norm", "t_max_norm"]
                if c in df_normales.columns]
        df_plot = df_plot.merge(df_normales[["mois"] + cols], on="mois", how="left")
        ax1.plot(df_plot["date_dt"], df_plot["t_moy_norm"],
                 color=C_NORM, linewidth=1.2, linestyle="--",
                 label=f"T° moy. {label_normale}", zorder=2)
        if "t_min_norm" in cols and "t_max_norm" in cols:
            ax1.fill_between(df_plot["date_dt"], df_plot["t_min_norm"],
                             df_plot["t_max_norm"], color=C_NORM, alpha=0.15,
                             zorder=1, label=f"Plage {label_normale}")

    if "rr1_mm" in df_agg.columns and df_agg["rr1_mm"].notna().any():
        ax2 = ax1.twinx()
        ax2.bar(df_agg["date_dt"], df_agg["rr1_mm"].fillna(0),
                width=0.03, color=C_PRECIP, alpha=0.55, label="Précip. (mm/h)")
        ax2.set_ylabel("Précipitations (mm/h)", color=C_PRECIP, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=C_PRECIP)
        ax2.set_ylim(bottom=0)
        l1, n1 = ax1.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, n1 + n2, loc="upper left", fontsize=9)
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
    if df_agg is None or df_agg.empty \
            or "ff_ms" not in df_agg.columns or "dd_deg" not in df_agg.columns:
        return None
    df = df_agg[["ff_ms", "dd_deg"]].dropna()
    if len(df) < 3:
        return None

    bins_v = [0, 2, 5, 8, 11, 17, np.inf]
    labels_v = ["0-2 m/s", "2-5 m/s", "5-8 m/s", "8-11 m/s", "11-17 m/s", ">17 m/s"]
    couleurs = ["#D6F5EC", "#8FE3CF", "#4FCDAE", "#1FB08C", "#0E8F6F", "#066352"]

    sec_deg = 360 / n_secteurs
    df = df.copy()
    df["secteur"] = (((df["dd_deg"] + sec_deg / 2) % 360) // sec_deg).astype(int).clip(0, n_secteurs - 1)
    df["cat_v"] = pd.cut(df["ff_ms"], bins=bins_v, labels=labels_v, right=False)
    theta = np.linspace(0, 2 * np.pi, n_secteurs, endpoint=False)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    bottom = np.zeros(n_secteurs)

    for cat, couleur in zip(labels_v, couleurs):
        counts = (df[df["cat_v"] == cat].groupby("secteur", observed=False).size()
                  .reindex(range(n_secteurs), fill_value=0).values)
        freqs = counts / len(df) * 100
        ax.bar(theta, freqs, width=np.radians(sec_deg) * 0.88,
               bottom=bottom, color=couleur, label=cat, alpha=0.95)
        bottom += freqs

    labels_dir = (
        ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
         "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
        if n_secteurs == 16 else ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    )
    ax.set_xticks(theta)
    ax.set_xticklabels(labels_dir[:n_secteurs], fontsize=9)
    ax.set_ylabel("Fréquence (%)", labelpad=30, fontsize=9)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.2, -0.16),
              title="Vitesse", fontsize=8, title_fontsize=9)
    ax.set_title(titre, fontsize=10.5, fontweight="bold", pad=16)
    fig.text(0.5, 0.01, SOURCE_LABEL, ha="center", fontsize=7.5, style="italic")
    plt.tight_layout()
    return fig


# ==============================================================================
# 3. THERMOPLUVIOGRAMME (mois complets de la période)
# ==============================================================================

def graphique_thermopluviogramme(df_mensuel, titre="Thermopluviogramme — bilan mensuel",
                                 df_normales=None,
                                 label_normale=DEFAULT_LABEL_NORMALE,
                                 etp_normale=None, label_etp="ETP"):
    """
    df_mensuel : une ligne par mois (colonne 'mois' de 1 à 12) avec t_moy, t_min,
    t_max et precip_tot, moyennés sur les mois complets observés.
    etp_normale : valeurs (mm/mois) alignées sur les lignes de df_mensuel.
    """
    if df_mensuel is None or df_mensuel.empty or "mois" not in df_mensuel.columns:
        return None

    df = df_mensuel.reset_index(drop=True)
    x = df["mois"].astype(int).values - 1       # position = mois - 1
    fig, ax1 = plt.subplots(figsize=(13, 6.5))
    ax2 = ax1.twinx()
    pmax = 1.0

    if df_normales is not None and not df_normales.empty \
            and "t_moy_norm" in df_normales.columns:
        cols = [c for c in ["t_moy_norm", "t_min_norm", "t_max_norm", "precip_norm"]
                if c in df_normales.columns]
        norm = df[["mois"]].merge(df_normales[["mois"] + cols], on="mois", how="left")
        if "t_min_norm" in cols and "t_max_norm" in cols:
            ax1.fill_between(x, norm["t_min_norm"], norm["t_max_norm"],
                             color=C_NORM, alpha=0.13, zorder=1)
        ax1.plot(x, norm["t_moy_norm"], color=C_NORM, linewidth=1.6, linestyle="--",
                 marker="o", markersize=3, label=f"T° moy. {label_normale}", zorder=2)
        if "precip_norm" in cols and norm["precip_norm"].notna().any():
            ax2.bar(x - 0.16, norm["precip_norm"].fillna(0), color=C_NORM, alpha=0.35,
                    width=0.30, label=f"Précip. {label_normale}", zorder=2)
            pmax = max(pmax, float(norm["precip_norm"].max()))

    if "precip_tot" in df.columns:
        p = df["precip_tot"].fillna(0).values
        ax2.bar(x + 0.16, p, color=C_PRECIP, alpha=0.65, width=0.30,
                label="Précip. observées (mm)", zorder=3)
        pmax = max(pmax, float(p.max()) if len(p) else 1.0)

    if etp_normale is not None:
        e = np.asarray(etp_normale, dtype=float)
        if len(e) == len(df) and np.isfinite(e).any():
            ax2.plot(x, e, color=C_ETP, linewidth=1.6, linestyle=":", marker="^",
                     markersize=4, label=label_etp, zorder=5, alpha=0.9)
            pmax = max(pmax, float(np.nanmax(e)))

    ax2.set_ylabel("Précipitations / ETP (mm/mois)", color=C_PRECIP, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=C_PRECIP)
    ax2.set_ylim(0, pmax * 2.4)

    ax1.plot(x, df["t_max"], color=C_MAX, linewidth=2, marker="o", markersize=5,
             label="T° max moy. observée", zorder=4)
    ax1.plot(x, df["t_moy"], color=C_MOY, linewidth=2.5, marker="o", markersize=5,
             label="T° moy. observée", zorder=4)
    ax1.plot(x, df["t_min"], color=C_MIN, linewidth=2, marker="o", markersize=5,
             label="T° min moy. observée", zorder=4)
    ax1.fill_between(x, df["t_min"], df["t_max"], alpha=0.08, color=C_MOY, zorder=1)

    ax1.set_ylabel("Température (°C)", fontsize=10)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(MOIS_LABELS, fontsize=10)
    ax1.set_xlim(-0.6, 11.6)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, loc="upper left", fontsize=8.5, framealpha=0.92, ncol=2)
    ax1.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig


# ==============================================================================
# 4. ETP MENSUELLE ET BILAN HYDRIQUE (normales)
# ==============================================================================

def graphique_etp(df_etp_normale, df_normales=None,
                  titre="Évapotranspiration et bilan hydrique",
                  label_normale=DEFAULT_LABEL_NORMALE,
                  label_etp="ETP Penman-Monteith (Météo-France)"):
    if df_etp_normale is None or df_etp_normale.empty:
        return None

    df = df_etp_normale.copy().sort_values("mois")
    if df_normales is not None and not df_normales.empty and "precip_norm" in df_normales.columns:
        df = df.merge(df_normales[["mois", "precip_norm"]], on="mois", how="left")

    x = df["mois"].astype(int).values - 1
    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    etp = df["etp_normale_mm"].values
    ax1.bar(x - 0.18, etp, width=0.34, color=C_ETP, alpha=0.85,
            label=f"{label_etp} - {label_normale}", zorder=3)

    if "precip_norm" in df.columns and df["precip_norm"].notna().any():
        p = df["precip_norm"].values
        ax1.bar(x + 0.18, np.nan_to_num(p), width=0.34, color=C_PRECIP, alpha=0.75,
                label=f"Précipitations {label_normale}", zorder=3)
        ax2 = ax1.twinx()
        ax2.plot(x, p - etp, color=C_BILAN, linewidth=2, marker="o", markersize=5,
                 label="Bilan hydrique (P - ETP)", zorder=5)
        ax2.axhline(0, color=C_BILAN, linewidth=0.8, linestyle="--", alpha=0.5)
        ax2.set_ylabel("Bilan hydrique P - ETP (mm)", color=C_BILAN, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=C_BILAN)
        l1, n1 = ax1.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, n1 + n2, loc="upper left", fontsize=9)
    else:
        ax1.legend(loc="upper left", fontsize=9)

    ax1.set_ylabel("mm / mois", fontsize=10)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(MOIS_LABELS, fontsize=10)
    ax1.set_xlim(-0.6, 11.6)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig


# ==============================================================================
# 5. ETP JOURNALIERE (avec normale répartie par jour)
# ==============================================================================

def graphique_etp_journalier(df_etp_jour, df_etp_normale=None,
                             titre="Évapotranspiration journalière",
                             label_normale=DEFAULT_LABEL_NORMALE):
    """
    df_etp_jour : date_dt, etp_mm.
    df_etp_normale : mois, etp_normale_mm. La normale mensuelle est répartie à
    parts égales sur les jours du mois ; courbes cumulées sur l'axe de droite.
    """
    if df_etp_jour is None or df_etp_jour.empty:
        return None
    df = df_etp_jour.sort_values("date_dt").reset_index(drop=True)
    dates = pd.to_datetime(df["date_dt"])

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.bar(dates, df["etp_mm"], width=0.8, color=C_ETP, alpha=0.8,
            label="ETP observée (mm/jour)", zorder=3)

    norm_j = None
    if df_etp_normale is not None and not df_etp_normale.empty:
        table = dict(zip(df_etp_normale["mois"], df_etp_normale["etp_normale_mm"]))
        norm_j = np.array([
            table.get(d.month, np.nan) / calendar.monthrange(d.year, d.month)[1]
            for d in dates
        ], dtype=float)
        ax1.plot(dates, norm_j, color=C_NORM, linewidth=1.6, linestyle="--",
                 label=f"ETP {label_normale} (mm/jour)", zorder=4)

    ax1.set_ylabel("ETP (mm/jour)", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(dates, df["etp_mm"].cumsum(), color=C_BILAN, linewidth=2,
             label="Cumul observé (mm)", zorder=5)
    if norm_j is not None and np.isfinite(norm_j).any():
        ax2.plot(dates, np.nancumsum(norm_j), color=C_BILAN, linewidth=1.4,
                 linestyle=":", label=f"Cumul {label_normale} (mm)", zorder=5)
    ax2.set_ylabel("Cumul (mm)", color=C_BILAN, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=C_BILAN)
    ax2.set_ylim(bottom=0)

    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, loc="upper left", fontsize=9)
    ax1.set_title(titre, fontsize=12, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    _source(fig)
    plt.tight_layout()
    return fig


# ==============================================================================
# 6. BILAN SUR LA PERIODE : NORMALE vs OBSERVE
# ==============================================================================

def graphique_bilan_annuel_etp(etp_normale, etp_observee, precip_normale, precip_observee,
                               titre="Bilan sur la période",
                               label_normale=DEFAULT_LABEL_NORMALE,
                               label_periode="Période observée"):
    """
    Barres groupées précipitations / ETP / bilan P - ETP. Normale et observé
    portent sur les mêmes jours (fournis par l'appelant).
    """
    def _val(a, b):
        return (None if a is None or pd.isna(a) else float(a),
                None if b is None or pd.isna(b) else float(b))

    pn, po = _val(precip_normale, precip_observee)
    en, eo = _val(etp_normale, etp_observee)
    bn = pn - en if pn is not None and en is not None else None
    bo = po - eo if po is not None and eo is not None else None

    categories = [("Précipitations", pn, po), ("ETP", en, eo), ("Bilan P - ETP", bn, bo)]
    categories = [c for c in categories if c[1] is not None or c[2] is not None]
    if not categories:
        return None

    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(9, 5))
    vn = [c[1] if c[1] is not None else 0 for c in categories]
    vo = [c[2] if c[2] is not None else 0 for c in categories]
    b1 = ax.bar(x - 0.19, vn, width=0.36, color=C_NORM, alpha=0.7, label=label_normale)
    b2 = ax.bar(x + 0.19, vo, width=0.36, color=C_PRECIP, alpha=0.85, label=label_periode)
    for barres, vals, src in ((b1, vn, [c[1] for c in categories]),
                              (b2, vo, [c[2] for c in categories])):
        for rect, v, s in zip(barres, vals, src):
            if s is not None:
                ax.annotate(f"{v:.0f}", (rect.get_x() + rect.get_width() / 2, v),
                            ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in categories], fontsize=10)
    ax.set_ylabel("mm", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig


# ==============================================================================
# 7. INTENSITE DES PRECIPITATIONS
# ==============================================================================

def graphique_intensite_pluie(repartition_df, titre="Répartition des intensités de pluie"):
    if repartition_df is None or repartition_df.empty or repartition_df["heures"].sum() == 0:
        return None
    couleurs = ["#9CCBF7", "#2F8FEF", "#1A5FB4", "#0B2F6B"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(repartition_df))[::-1]
    ax.barh(y, repartition_df["heures"], color=couleurs[:len(repartition_df)], alpha=0.9)
    for yi, h in zip(y, repartition_df["heures"]):
        ax.annotate(f" {int(h)} h", (h, yi), va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(repartition_df["classe"], fontsize=9)
    ax.set_xlabel("Nombre d'heures", fontsize=10)
    ax.set_xlim(0, repartition_df["heures"].max() * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig


# ==============================================================================
# 8. SUIVI JOURNALIER (fenêtres courtes)
# ==============================================================================

def graphique_histogramme_periode(df, titre="Précipitations et températures — suivi journalier",
                                  df_normales=None, label_normale=DEFAULT_LABEL_NORMALE):
    if df is None or df.empty or "t_moy" not in df.columns:
        return None

    df = df.reset_index(drop=True)
    n = len(df)
    fig, ax1 = plt.subplots(figsize=(13, 5))
    x = np.arange(n)

    if df_normales is not None and not df_normales.empty \
            and "mois" in df.columns and "t_moy_norm" in df_normales.columns:
        cols = [c for c in ["t_moy_norm", "t_min_norm", "t_max_norm"] if c in df_normales.columns]
        df_m = df[["mois"]].merge(df_normales[["mois"] + cols], on="mois", how="left")
        if "t_min_norm" in cols and "t_max_norm" in cols:
            ax1.fill_between(x, df_m["t_min_norm"], df_m["t_max_norm"], color=C_NORM,
                             alpha=0.15, zorder=1, label=f"Plage {label_normale}")
        ax1.plot(x, df_m["t_moy_norm"], color=C_NORM, linewidth=1.2, linestyle="--",
                 label=f"T° moy. {label_normale}", zorder=2)

    ax2 = None
    if "precip_tot" in df.columns and df["precip_tot"].notna().any():
        ax2 = ax1.twinx()
        ax2.bar(x, df["precip_tot"].fillna(0), color=C_PRECIP, alpha=0.6,
                label="Précip. (mm)", zorder=2)
        ax2.set_ylabel("Précipitations (mm)", color=C_PRECIP, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=C_PRECIP)
        ax2.set_ylim(bottom=0)

    ax1.plot(x, df["t_moy"], color=C_MOY, linewidth=2.2, marker="o", markersize=5,
             label="T° moy", zorder=3)
    if "t_min" in df.columns and "t_max" in df.columns:
        ax1.plot(x, df["t_max"], color=C_MAX, linewidth=1.5, marker="o", markersize=4,
                 label="T° max", zorder=3)
        ax1.plot(x, df["t_min"], color=C_MIN, linewidth=1.5, marker="o", markersize=4,
                 label="T° min", zorder=3)
        ax1.fill_between(x, df["t_min"], df["t_max"], alpha=0.08, color=C_MOY, zorder=1)

    ax1.set_ylabel("Température (°C)", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    if "date_dt" in df.columns:
        pas = max(1, n // 12)
        fmt = "%d/%m" if n <= 62 else "%m/%Y"
        labels = [pd.Timestamp(d).strftime(fmt) for d in df["date_dt"]]
        ax1.set_xticks(x[::pas])
        ax1.set_xticklabels(labels[::pas], rotation=35, ha="right", fontsize=9)

    l1, n1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        l2, n2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, n1 + n2, loc="upper left", fontsize=9)
    else:
        ax1.legend(l1, n1, loc="upper left", fontsize=9)

    ax1.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig


# ==============================================================================
# 9. TEMPERATURES ANNUELLES
# ==============================================================================

def graphique_temperatures_annuelles(df_annuel, titre="Évolution des températures annuelles",
                                     df_normales=None, label_normale=DEFAULT_LABEL_NORMALE):
    if df_annuel is None or df_annuel.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    x = df_annuel["annee"].values

    if df_normales is not None and not df_normales.empty \
            and "t_moy_norm" in df_normales.columns:
        ax.axhline(df_normales["t_moy_norm"].mean(), color=C_NORM, linewidth=1.2,
                   linestyle="--", label=f"T° moy. {label_normale}", zorder=2)
        if "t_min_norm" in df_normales.columns and "t_max_norm" in df_normales.columns:
            ax.axhspan(df_normales["t_min_norm"].mean(), df_normales["t_max_norm"].mean(),
                       color=C_NORM, alpha=0.1, zorder=1, label=f"Plage {label_normale}")

    ax.fill_between(x, df_annuel["t_min"], df_annuel["t_max"], alpha=0.12, color=C_MOY,
                    label="Amplitude T° min - T° max")
    ax.plot(x, df_annuel["t_max"], color=C_MAX, linewidth=2, marker="o", markersize=5,
            label="T° max moy. annuelle")
    ax.plot(x, df_annuel["t_moy"], color=C_MOY, linewidth=2.5, marker="o", markersize=6,
            label="T° moyenne annuelle")
    ax.plot(x, df_annuel["t_min"], color=C_MIN, linewidth=2, marker="o", markersize=5,
            label="T° min moy. annuelle")

    ax.set_xlabel("Année", fontsize=10)
    ax.set_ylabel("Température (°C)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(a)) for a in x], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.92)
    ax.set_title(titre, fontsize=12, fontweight="bold")
    _source(fig)
    plt.tight_layout()
    return fig
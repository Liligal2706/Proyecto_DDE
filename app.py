import warnings
import textwrap
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

st.set_page_config(
    page_title="Redes sociales vs productividad",
    page_icon="📊",
    layout="wide"
)

CSV_PATH = Path("data/social_media_vs_productivity.csv")

ALPHA = 0.05

COLORS = {
    "bg": "#0B1220",
    "panel": "#111827",
    "panel_2": "#172033",
    "panel_3": "#1F2937",
    "border": "#2D3748",
    "text": "#E5E7EB",
    "muted": "#9CA3AF",
    "blue_dark": "#3D5A80",
    "blue_mid": "#5C7EA4",
    "blue_light": "#98C1D9",
    "teal": "#7FBFBF",
    "lavender": "#B8D4E3",
    "warning": "#F59E0B",
    "success": "#10B981",
    "danger": "#EF4444"
}

PALETTE = [
    COLORS["blue_light"],
    COLORS["blue_mid"],
    COLORS["blue_dark"],
    COLORS["teal"],
    COLORS["lavender"],
    "#A0B0BB"
]

PAL_SML = {
    "Bajo": COLORS["blue_light"],
    "Medio": COLORS["blue_mid"],
    "Alto": COLORS["blue_dark"]
}


# =============================================================================
# ESTILO GENERAL
# =============================================================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLORS["bg"]};
        color: {COLORS["text"]};
    }}

    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }}

    h1, h2, h3, h4 {{
        color: {COLORS["text"]};
        font-weight: 800;
    }}

    p, li, span, div {{
        color: {COLORS["text"]};
    }}

    .main-title {{
        background: linear-gradient(135deg, #111827 0%, #1F2937 55%, #26364D 100%);
        padding: 2rem 2.2rem;
        border-radius: 26px;
        border: 1px solid {COLORS["border"]};
        box-shadow: 0 18px 45px rgba(0,0,0,0.35);
        margin-bottom: 1.2rem;
    }}

    .main-title h1 {{
        font-size: 2.6rem;
        line-height: 1.05;
        margin-bottom: 0.5rem;
        color: #FFFFFF;
    }}

    .main-title p {{
        color: {COLORS["muted"]};
        margin-bottom: 0.25rem;
        font-size: 0.98rem;
    }}

    .metric-card {{
        background: linear-gradient(145deg, #111827 0%, #172033 100%);
        padding: 1.2rem 1.35rem;
        border-radius: 20px;
        border: 1px solid {COLORS["border"]};
        box-shadow: 0 12px 28px rgba(0,0,0,0.24);
        min-height: 118px;
    }}

    .metric-title {{
        color: {COLORS["muted"]};
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-bottom: 0.45rem;
    }}

    .metric-value {{
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 850;
        line-height: 1.1;
    }}

    .metric-note {{
        color: {COLORS["muted"]};
        font-size: 0.78rem;
        margin-top: 0.45rem;
        line-height: 1.35;
    }}

    .info-box {{
        background: linear-gradient(145deg, #111827 0%, #172033 100%);
        border-left: 5px solid {COLORS["blue_light"]};
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin: 1rem 0 1.2rem 0;
        border-top: 1px solid {COLORS["border"]};
        border-right: 1px solid {COLORS["border"]};
        border-bottom: 1px solid {COLORS["border"]};
        box-shadow: 0 10px 24px rgba(0,0,0,0.22);
    }}

    .interp-box {{
        background: #101827;
        border-left: 5px solid {COLORS["teal"]};
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-top: 0.75rem;
        margin-bottom: 1.4rem;
        border-top: 1px solid {COLORS["border"]};
        border-right: 1px solid {COLORS["border"]};
        border-bottom: 1px solid {COLORS["border"]};
        box-shadow: 0 8px 20px rgba(0,0,0,0.20);
        line-height: 1.55;
        font-size: 0.96rem;
        width: 100%;
        max-width: 100%;
        overflow: visible;
        white-space: normal;
        word-break: normal;
        overflow-wrap: anywhere;
    }}

    .interp-title {{
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.65rem;
    }}

    .interp-text {{
        color: {COLORS["text"]};
        white-space: normal;
        word-break: normal;
        overflow-wrap: anywhere;
        line-height: 1.65;
    }}

    .interp-box pre,
    .interp-box code {{
        white-space: normal !important;
        overflow-x: visible !important;
        font-family: inherit !important;
    }}

    .warn-box {{
        background: rgba(245, 158, 11, 0.11);
        border-left: 5px solid {COLORS["warning"]};
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        border-top: 1px solid rgba(245,158,11,.25);
        border-right: 1px solid rgba(245,158,11,.25);
        border-bottom: 1px solid rgba(245,158,11,.25);
        line-height: 1.55;
    }}

    .good-box {{
        background: rgba(16, 185, 129, 0.12);
        border-left: 5px solid {COLORS["success"]};
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        border-top: 1px solid rgba(16,185,129,.25);
        border-right: 1px solid rgba(16,185,129,.25);
        border-bottom: 1px solid rgba(16,185,129,.25);
        line-height: 1.55;
    }}

    .section-card {{
        background: #111827;
        border: 1px solid {COLORS["border"]};
        border-radius: 22px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.24);
    }}

    div[data-testid="stDataFrame"] {{
        background: {COLORS["panel"]};
        border-radius: 16px;
        border: 1px solid {COLORS["border"]};
        padding: .25rem;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.35rem;
        background-color: #0B1220;
        padding: 0.5rem 0;
        flex-wrap: wrap;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: #111827;
        border: 1px solid {COLORS["border"]};
        border-radius: 999px;
        padding: 0.55rem 1rem;
        color: {COLORS["muted"]};
        font-weight: 700;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #3D5A80 0%, #5C7EA4 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #98C1D9 !important;
    }}

    hr {{
        border-color: {COLORS["border"]};
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# FUNCIONES DE PRESENTACIÓN
# =============================================================================

def metric_card(title, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def info_box(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def interpretation_box(title, text):
    clean_text = textwrap.dedent(text).strip()
    clean_text = re.sub(r"\s+", " ", clean_text)

    st.markdown(
        f"""
        <div class="interp-box">
            <div class="interp-title">{title}</div>
            <div class="interp-text">{clean_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def warn_box(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def good_box(text):
    st.markdown(f'<div class="good-box">{text}</div>', unsafe_allow_html=True)

def format_p(p):
    if pd.isna(p):
        return "No disponible"
    if p < 0.0001:
        return "< 0.0001"
    return f"{p:.4f}"

def pvalue_decision(p, alpha=0.05):
    if pd.isna(p):
        return "No disponible."
    if p < alpha:
        return f"Se rechaza H₀ porque p = {format_p(p)} < {alpha}."
    return f"No se rechaza H₀ porque p = {format_p(p)} ≥ {alpha}."


def plot_layout(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=title if title else fig.layout.title.text,
            font=dict(size=21, color="#FFFFFF"),
            x=0.02
        ),
        font=dict(color=COLORS["text"]),
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        margin=dict(l=35, r=35, t=75, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)"
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            font_size=13,
            font_color="#FFFFFF"
        )
    )

    fig.update_xaxes(
        gridcolor="#2D3748",
        zerolinecolor="#2D3748",
        linecolor="#4B5563"
    )
    fig.update_yaxes(
        gridcolor="#2D3748",
        zerolinecolor="#2D3748",
        linecolor="#4B5563"
    )

    return fig

def show_plot(fig, key):
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key
    )

# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

@st.cache_data
def load_data():
    if not CSV_PATH.exists():
        return None

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def prepare_data(df):
    df = df.copy()

    numeric_cols = [
        "age",
        "daily_social_media_time",
        "number_of_notifications",
        "work_hours_per_day",
        "perceived_productivity_score",
        "actual_productivity_score",
        "stress_level",
        "sleep_hours",
        "screen_time_before_sleep",
        "breaks_during_work",
        "coffee_consumption_per_day",
        "days_feeling_burnout_per_month",
        "weekly_offline_hours",
        "job_satisfaction_score"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "gender" in df.columns:
        df["gender"] = df["gender"].replace({
            "Male": "Masculino",
            "Female": "Femenino",
            "Other": "Otro"
        })

    if "job_type" in df.columns:
        df["job_type"] = df["job_type"].replace({
            "Education": "Educación",
            "Finance": "Finanzas",
            "Health": "Salud",
            "IT": "Tecnología",
            "Student": "Estudiante",
            "Unemployed": "Desempleado"
        })

    for col in ["uses_focus_apps", "has_digital_wellbeing_enabled"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({
                "True": "Sí",
                "False": "No",
                "TRUE": "Sí",
                "FALSE": "No",
                "true": "Sí",
                "false": "No",
                "1": "Sí",
                "0": "No"
            })

    if "daily_social_media_time" in df.columns:
        q1, q2 = df["daily_social_media_time"].quantile([1 / 3, 2 / 3])

        def classify_social_time(x):
            if pd.isna(x):
                return np.nan
            if x <= q1:
                return "Bajo"
            if x <= q2:
                return "Medio"
            return "Alto"

        df["social_media_level"] = df["daily_social_media_time"].apply(classify_social_time)
        df["social_media_level"] = pd.Categorical(
            df["social_media_level"],
            categories=["Bajo", "Medio", "Alto"],
            ordered=True
        )

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[17, 29, 44, 59, np.inf],
            labels=["18–29", "30–44", "45–59", "60+"]
        )

    if {"perceived_productivity_score", "actual_productivity_score"}.issubset(df.columns):
        df["productivity_gap"] = (
            df["perceived_productivity_score"] -
            df["actual_productivity_score"]
        )

    return df


def validate_columns(df):
    required_cols = [
        "age",
        "gender",
        "job_type",
        "daily_social_media_time",
        "social_platform_preference",
        "number_of_notifications",
        "work_hours_per_day",
        "perceived_productivity_score",
        "actual_productivity_score",
        "stress_level",
        "sleep_hours",
        "screen_time_before_sleep",
        "breaks_during_work",
        "uses_focus_apps",
        "has_digital_wellbeing_enabled",
        "coffee_consumption_per_day",
        "days_feeling_burnout_per_month",
        "weekly_offline_hours",
        "job_satisfaction_score"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    return missing


# =============================================================================
# FUNCIONES DE MUESTREO
# =============================================================================

def sample_size_fpc(y, error_margin=0.10, confidence=0.95):
    y = pd.Series(y).dropna()
    N = len(y)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    s2 = y.var(ddof=1)

    n0 = (z**2 * s2) / (error_margin**2)
    n = n0 / (1 + ((n0 - 1) / N))

    return {
        "N": N,
        "z": z,
        "s2": s2,
        "n0": n0,
        "n": min(int(np.ceil(n)), N)
    }


def proportional_allocation(df, n_total, strata_col="job_type", y_col="actual_productivity_score"):
    base = df[[strata_col, y_col]].dropna().copy()

    info = (
        base
        .groupby(strata_col)
        .agg(
            N_h=(y_col, "size"),
            S_h=(y_col, "std")
        )
        .reset_index()
    )

    info["n_h"] = np.round(n_total * info["N_h"] / info["N_h"].sum()).astype(int)
    info.loc[info["n_h"] < 1, "n_h"] = 1

    diff = n_total - info["n_h"].sum()

    while diff != 0:
        if diff > 0:
            idx = info["N_h"].idxmax()
            info.loc[idx, "n_h"] += 1
            diff -= 1
        else:
            candidates = info[info["n_h"] > 1]
            if candidates.empty:
                break
            idx = candidates["n_h"].idxmax()
            info.loc[idx, "n_h"] -= 1
            diff += 1

    info["peso_h"] = info["N_h"] / info["n_h"]
    info["pct_poblacion"] = 100 * info["N_h"] / info["N_h"].sum()

    return info


def draw_stratified_sample(df, allocation, strata_col="job_type", y_col="actual_productivity_score"):
    pieces = []

    for _, row in allocation.iterrows():
        stratum = row[strata_col]
        n_h = int(row["n_h"])

        part = df[(df[strata_col] == stratum) & df[y_col].notna()].copy()
        n_h = min(n_h, len(part))

        pieces.append(part.sample(n=n_h, random_state=123))

    sample = pd.concat(pieces, ignore_index=True)

    sample = sample.merge(
        allocation[[strata_col, "N_h", "n_h", "peso_h"]],
        on=strata_col,
        how="left"
    )

    return sample


def stratified_mean_ci(sample, allocation, y_col="actual_productivity_score"):
    N = allocation["N_h"].sum()

    stats_h = (
        sample
        .groupby("job_type")
        .agg(
            ybar_h=(y_col, "mean"),
            s2_h=(y_col, "var"),
            n_h_sample=(y_col, "size")
        )
        .reset_index()
        .merge(allocation[["job_type", "N_h"]], on="job_type", how="left")
    )

    stats_h["W_h"] = stats_h["N_h"] / N
    stats_h["f_h"] = stats_h["n_h_sample"] / stats_h["N_h"]

    ybar = (stats_h["W_h"] * stats_h["ybar_h"]).sum()

    var_est = (
        (stats_h["W_h"] ** 2) *
        (1 - stats_h["f_h"]) *
        stats_h["s2_h"] /
        stats_h["n_h_sample"]
    ).sum()

    se = np.sqrt(var_est)
    ci_low = ybar - 1.96 * se
    ci_high = ybar + 1.96 * se

    return ybar, se, ci_low, ci_high


def mas_mean_ci(df, n, y_col="actual_productivity_score"):
    base = df[df[y_col].notna()].copy()
    N = len(base)
    n = min(n, N)

    sample = base.sample(n=n, random_state=456)

    mean = sample[y_col].mean()
    s2 = sample[y_col].var(ddof=1)
    se = np.sqrt((1 - n / N) * s2 / n)
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se

    return mean, se, ci_low, ci_high


def auxiliary_estimators(df, sample):
    y_col = "actual_productivity_score"
    x_col = "perceived_productivity_score"

    complete_sample = sample[[y_col, x_col]].dropna().copy()
    complete_pop = df[[y_col, x_col]].dropna().copy()

    if complete_sample.empty or complete_pop.empty:
        return pd.DataFrame()

    mu_x_pop = complete_pop[x_col].mean()
    mu_y_pop = complete_pop[y_col].mean()

    ybar_sample = complete_sample[y_col].mean()
    xbar_sample = complete_sample[x_col].mean()

    ratio = ybar_sample / xbar_sample
    ratio_est = ratio * mu_x_pop

    model = smf.ols(f"{y_col} ~ {x_col}", data=complete_sample).fit()
    b1 = model.params[x_col]
    reg_est = ybar_sample + b1 * (mu_x_pop - xbar_sample)

    out = pd.DataFrame({
        "Estimador": ["Directo", "Razón", "Regresión"],
        "Media estimada": [ybar_sample, ratio_est, reg_est],
        "Media real de referencia": [mu_y_pop, mu_y_pop, mu_y_pop]
    })

    out["Error absoluto"] = (
        out["Media estimada"] - out["Media real de referencia"]
    ).abs()

    return out


# =============================================================================
# FUNCIONES DE DBCA / ANOVA
# =============================================================================

def create_balanced_dbca(df):
    needed = ["actual_productivity_score", "social_media_level", "job_type"]

    missing_cols = [col for col in needed if col not in df.columns]
    if missing_cols:
        st.error(
            "No se puede construir el DBCA porque faltan estas columnas: "
            + ", ".join(missing_cols)
        )
        return pd.DataFrame(columns=needed), pd.DataFrame(), 0

    base = df[needed].copy()

    base["actual_productivity_score"] = pd.to_numeric(
        base["actual_productivity_score"],
        errors="coerce"
    )

    base["social_media_level"] = (
        base["social_media_level"]
        .astype(str)
        .str.strip()
    )

    base["job_type"] = (
        base["job_type"]
        .astype(str)
        .str.strip()
    )

    base = base.dropna(subset=["actual_productivity_score"])

    # Conserva únicamente los tres tratamientos válidos.
    base = base[
        base["social_media_level"].isin(["Bajo", "Medio", "Alto"])
    ].copy()

    # Elimina etiquetas vacías que pueden aparecer distinto en Streamlit Cloud.
    base = base[
        ~base["job_type"].isin(["nan", "None", "", "NaN", "<NA>"])
    ].copy()

    if base.empty:
        st.error(
            "No hay registros válidos para construir el DBCA. "
            "Revisa que existan datos completos en productividad real, nivel de redes y tipo de trabajo."
        )
        return pd.DataFrame(columns=needed), pd.DataFrame(), 0

    conteos = (
        base
        .groupby(["social_media_level", "job_type"], observed=True)
        .size()
        .reset_index(name="n")
    )

    conteos = conteos[conteos["n"] > 0].copy()

    if conteos.empty:
        st.error(
            "No hay combinaciones observadas entre nivel de redes y tipo de trabajo para construir el DBCA."
        )
        return pd.DataFrame(columns=needed), pd.DataFrame(), 0

    # Se usan solo bloques que tengan observaciones en los tres tratamientos.
    tabla_completa = conteos.pivot_table(
        index="job_type",
        columns="social_media_level",
        values="n",
        fill_value=0,
        observed=True
    )

    for nivel in ["Bajo", "Medio", "Alto"]:
        if nivel not in tabla_completa.columns:
            tabla_completa[nivel] = 0

    bloques_validos = tabla_completa[
        (tabla_completa["Bajo"] > 0) &
        (tabla_completa["Medio"] > 0) &
        (tabla_completa["Alto"] > 0)
    ].index.tolist()

    if len(bloques_validos) == 0:
        st.error(
            "No existe ningún tipo de trabajo con observaciones en los tres niveles "
            "de redes sociales. Por eso no se puede formar un DBCA completo."
        )

        balance_debug = pd.crosstab(
            base["social_media_level"],
            base["job_type"]
        )

        return pd.DataFrame(columns=needed), balance_debug, 0

    base = base[base["job_type"].isin(bloques_validos)].copy()

    tabla_balance_previa = pd.crosstab(
        base["social_media_level"],
        base["job_type"]
    )

    n_cell = int(tabla_balance_previa.min().min())

    if n_cell <= 0:
        st.error(
            "No se pudo calcular un número positivo de réplicas por celda para el DBCA."
        )
        return pd.DataFrame(columns=needed), tabla_balance_previa, 0

    partes = []

    for bloque in bloques_validos:
        for tratamiento in ["Bajo", "Medio", "Alto"]:
            celda = base[
                (base["job_type"] == bloque) &
                (base["social_media_level"] == tratamiento)
            ].copy()

            if len(celda) >= n_cell:
                partes.append(
                    celda.sample(n=n_cell, random_state=123)
                )

    if len(partes) == 0:
        st.error(
            "No se pudo construir correctamente la base balanceada del DBCA."
        )
        return pd.DataFrame(columns=needed), tabla_balance_previa, 0

    balanced = pd.concat(partes, ignore_index=True)

    balance_table = pd.crosstab(
        balanced["social_media_level"],
        balanced["job_type"]
    )

    return balanced, balance_table, n_cell


def fit_dbca_anova(dbca):
    anova_cols = ["Fuente", "sum_sq", "df", "F", "PR(>F)"]

    if dbca is None or dbca.empty:
        return None, pd.DataFrame(columns=anova_cols)

    required = ["actual_productivity_score", "social_media_level", "job_type"]
    missing = [col for col in required if col not in dbca.columns]

    if missing:
        st.error(
            "No se puede ajustar el ANOVA porque faltan estas columnas: "
            + ", ".join(missing)
        )
        return None, pd.DataFrame(columns=anova_cols)

    dbca_model = dbca[required].dropna().copy()

    if dbca_model.empty:
        return None, pd.DataFrame(columns=anova_cols)

    if dbca_model["social_media_level"].nunique() < 2:
        st.error(
            "No se puede ajustar el ANOVA porque hay menos de dos niveles de tratamiento."
        )
        return None, pd.DataFrame(columns=anova_cols)

    if dbca_model["job_type"].nunique() < 2:
        st.error(
            "No se puede ajustar el ANOVA porque hay menos de dos bloques."
        )
        return None, pd.DataFrame(columns=anova_cols)

    try:
        model = smf.ols(
            "actual_productivity_score ~ C(social_media_level) + C(job_type)",
            data=dbca_model
        ).fit()

        anova = sm.stats.anova_lm(model, typ=2)

        anova = anova.reset_index().rename(columns={"index": "Fuente"})

        anova["Fuente"] = anova["Fuente"].replace({
            "C(social_media_level)": "Tratamiento: nivel de redes",
            "C(job_type)": "Bloque: tipo de trabajo",
            "Residual": "Error"
        })

        for col in anova_cols:
            if col not in anova.columns:
                anova[col] = np.nan

        anova = anova[anova_cols]

        return model, anova

    except Exception as e:
        st.error(
            "No se pudo ajustar el modelo ANOVA. "
            "Revisa que existan suficientes observaciones por tratamiento y bloque."
        )
        return None, pd.DataFrame(columns=anova_cols)

def get_anova_pvalue(anova, fuente):
    if anova is None or anova.empty:
        return np.nan

    if "Fuente" not in anova.columns or "PR(>F)" not in anova.columns:
        return np.nan

    value = anova.loc[anova["Fuente"] == fuente, "PR(>F)"]

    if value.empty:
        return np.nan

    return float(value.iloc[0])

def tukey_table(dbca):
    if dbca is None or dbca.empty:
        return pd.DataFrame()

    required = ["actual_productivity_score", "social_media_level"]
    if any(col not in dbca.columns for col in required):
        return pd.DataFrame()

    try:
        tukey = pairwise_tukeyhsd(
            endog=dbca["actual_productivity_score"],
            groups=dbca["social_media_level"],
            alpha=0.05
        )

        return pd.DataFrame(
            data=tukey.summary().data[1:],
            columns=tukey.summary().data[0]
        )
    except Exception:
        return pd.DataFrame()


def compute_power(anova, dbca):
    if anova is None or anova.empty or dbca is None or dbca.empty:
        return None

    if "Fuente" not in anova.columns or "sum_sq" not in anova.columns:
        return None

    ss_treat = anova.loc[
        anova["Fuente"] == "Tratamiento: nivel de redes",
        "sum_sq"
    ]

    ss_total = anova["sum_sq"].sum()

    if ss_treat.empty or ss_total <= 0:
        return None

    eta2 = float(ss_treat.iloc[0] / ss_total)
    eta2 = min(eta2, 0.999)

    f_cohen = np.sqrt(eta2 / (1 - eta2)) if eta2 > 0 else 0.01
    f_for_power = max(f_cohen, 0.01)

    k = dbca["social_media_level"].nunique()
    n_per_group = int(dbca.groupby("social_media_level").size().min())

    power_analysis = FTestAnovaPower()

    current_power = power_analysis.power(
        effect_size=f_for_power,
        nobs=n_per_group * k,
        alpha=0.05,
        k_groups=k
    )

    n_required_total = power_analysis.solve_power(
        effect_size=f_for_power,
        power=0.80,
        alpha=0.05,
        k_groups=k
    )

    return {
        "eta2": eta2,
        "f_cohen": f_cohen,
        "k": k,
        "n_per_group": n_per_group,
        "current_power": current_power,
        "n_required_total": int(np.ceil(n_required_total)),
        "n_required_group": int(np.ceil(n_required_total / k))
    }


# =============================================================================
# FUNCIONES DE GRÁFICAS
# =============================================================================

def histogram_fig(df, col, title, x_label, color):
    clean = df.dropna(subset=[col]).copy()

    fig = px.histogram(
        clean,
        x=col,
        nbins=40,
        color_discrete_sequence=[color],
        opacity=0.88
    )

    mean_value = clean[col].mean()

    fig.add_vline(
        x=mean_value,
        line_dash="dash",
        line_color="#E5E7EB",
        annotation_text=f"Media: {mean_value:.2f}",
        annotation_position="top right"
    )

    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="Frecuencia")

    return plot_layout(fig, title)


def boxplot_fig(df, x, y, title, x_label, y_label):
    fig = px.box(
        df.dropna(subset=[x, y]),
        x=x,
        y=y,
        color=x,
        color_discrete_map=PAL_SML if x == "social_media_level" else None,
        color_discrete_sequence=PALETTE,
        points="outliers"
    )

    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)

    return plot_layout(fig, title)


def scatter_fig(df, x, y, title, x_label, y_label):
    clean = df[[x, y]].dropna().copy()

    fig = px.scatter(
        clean,
        x=x,
        y=y,
        trendline="ols",
        opacity=0.38,
        color_discrete_sequence=[COLORS["blue_light"]]
    )

    fig.update_traces(marker=dict(size=6))
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)

    return plot_layout(fig, title)


def correlation_heatmap(df):
    cols = [
        "daily_social_media_time",
        "actual_productivity_score",
        "perceived_productivity_score",
        "stress_level",
        "sleep_hours",
        "work_hours_per_day",
        "number_of_notifications",
        "job_satisfaction_score",
        "days_feeling_burnout_per_month",
        "weekly_offline_hours",
        "breaks_during_work",
        "screen_time_before_sleep",
        "coffee_consumption_per_day"
    ]

    cols = [c for c in cols if c in df.columns]

    corr = df[cols].dropna().corr()

    labels = {
        "daily_social_media_time": "Tiempo redes",
        "actual_productivity_score": "Prod. real",
        "perceived_productivity_score": "Prod. percibida",
        "stress_level": "Estrés",
        "sleep_hours": "Sueño",
        "work_hours_per_day": "Horas trabajo",
        "number_of_notifications": "Notificaciones",
        "job_satisfaction_score": "Satisfacción",
        "days_feeling_burnout_per_month": "Burnout",
        "weekly_offline_hours": "Horas offline",
        "breaks_during_work": "Pausas",
        "screen_time_before_sleep": "Pantalla sueño",
        "coffee_consumption_per_day": "Café"
    }

    corr = corr.rename(index=labels, columns=labels)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=[
            COLORS["blue_dark"],
            "#111827",
            COLORS["teal"]
        ],
        zmin=-1,
        zmax=1,
        aspect="auto"
    )

    fig.update_layout(coloraxis_colorbar=dict(title="r"))

    return plot_layout(fig, "Matriz de correlación entre variables numéricas")


def ci_comparison_fig(results):
    fig = go.Figure()

    for _, row in results.iterrows():
        if pd.notna(row["IC inferior"]) and pd.notna(row["IC superior"]):
            fig.add_trace(
                go.Scatter(
                    x=[row["IC inferior"], row["IC superior"]],
                    y=[row["Estimador"], row["Estimador"]],
                    mode="lines",
                    line=dict(color=COLORS["blue_light"], width=8),
                    showlegend=False
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[row["Media"], row["Media"]],
                    y=[row["Estimador"], row["Estimador"]],
                    mode="markers",
                    marker=dict(color="#FFFFFF", size=14),
                    showlegend=False
                )
            )

    fig.update_xaxes(title="Productividad real media estimada")
    fig.update_yaxes(title="")

    return plot_layout(fig, "Comparación de intervalos de confianza")


def means_dbca_fig(dbca):
    means = (
        dbca
        .groupby("social_media_level")
        .agg(
            media=("actual_productivity_score", "mean"),
            sd=("actual_productivity_score", "std"),
            n=("actual_productivity_score", "size")
        )
        .reset_index()
    )

    means["se"] = means["sd"] / np.sqrt(means["n"])
    means["ci"] = 1.96 * means["se"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=means["social_media_level"],
            y=means["media"],
            error_y=dict(type="data", array=means["ci"]),
            marker_color=[
                PAL_SML.get(x, COLORS["blue_mid"])
                for x in means["social_media_level"]
            ],
            text=[f"{x:.2f}" for x in means["media"]],
            textposition="outside"
        )
    )

    fig.update_xaxes(title="Nivel de uso de redes sociales")
    fig.update_yaxes(title="Productividad real media", range=[0, 10])

    return plot_layout(fig, "Medias de productividad real por nivel de redes sociales")


def residual_diagnostic_figs(model):
    residuals = model.resid
    fitted = model.fittedvalues

    fig_hist = px.histogram(
        x=residuals,
        nbins=40,
        color_discrete_sequence=[COLORS["blue_light"]],
        opacity=0.88
    )
    fig_hist.add_vline(
        x=0,
        line_dash="dash",
        line_color="#FFFFFF"
    )
    fig_hist.update_xaxes(title="Residuos")
    fig_hist.update_yaxes(title="Frecuencia")
    fig_hist = plot_layout(fig_hist, "Histograma de residuos")

    sorted_res = np.sort(residuals)

    theoretical = stats.norm.ppf(
        (np.arange(1, len(sorted_res) + 1) - 0.5) / len(sorted_res)
    )

    fig_qq = go.Figure()

    fig_qq.add_trace(
        go.Scatter(
            x=theoretical,
            y=sorted_res,
            mode="markers",
            marker=dict(color=COLORS["blue_light"], opacity=0.55, size=5),
            name="Residuos"
        )
    )

    min_val = min(theoretical.min(), sorted_res.min())
    max_val = max(theoretical.max(), sorted_res.max())

    fig_qq.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#FFFFFF", dash="dash"),
            name="Referencia"
        )
    )

    fig_qq.update_xaxes(title="Cuantiles teóricos")
    fig_qq.update_yaxes(title="Cuantiles de residuos")
    fig_qq = plot_layout(fig_qq, "Q-Q plot de residuos")

    fig_rv = px.scatter(
        x=fitted,
        y=residuals,
        color_discrete_sequence=[COLORS["blue_light"]],
        opacity=0.48
    )

    fig_rv.add_hline(
        y=0,
        line_dash="dash",
        line_color="#FFFFFF"
    )

    fig_rv.update_xaxes(title="Valores ajustados")
    fig_rv.update_yaxes(title="Residuos")
    fig_rv = plot_layout(fig_rv, "Residuos vs valores ajustados")

    fig_seq = px.line(
        x=np.arange(len(residuals)),
        y=residuals,
        color_discrete_sequence=[COLORS["blue_light"]]
    )

    fig_seq.add_hline(
        y=0,
        line_dash="dash",
        line_color="#FFFFFF"
    )

    fig_seq.update_xaxes(title="Índice de observación")
    fig_seq.update_yaxes(title="Residuos")
    fig_seq = plot_layout(fig_seq, "Residuos en secuencia")

    return fig_hist, fig_qq, fig_rv, fig_seq


def power_curve_fig(power_info):
    power_analysis = FTestAnovaPower()

    k = power_info["k"]
    f = max(power_info["f_cohen"], 0.01)

    max_n = max(600, power_info["n_required_group"] * 2)
    n_values = np.arange(10, max_n, 10)

    power_values = [
        power_analysis.power(
            effect_size=f,
            nobs=int(n * k),
            alpha=0.05,
            k_groups=k
        )
        for n in n_values
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=n_values,
            y=power_values,
            mode="lines",
            fill="tozeroy",
            line=dict(color=COLORS["blue_light"], width=3),
            fillcolor="rgba(152,193,217,0.28)",
            name="Potencia"
        )
    )

    fig.add_hline(
        y=0.80,
        line_dash="dash",
        line_color="#FFFFFF",
        annotation_text="Potencia 0.80"
    )

    fig.add_vline(
        x=power_info["n_required_group"],
        line_dash="dot",
        line_color=COLORS["teal"],
        annotation_text=f"n ≈ {power_info['n_required_group']} por grupo"
    )

    fig.update_xaxes(title="Tamaño de muestra por grupo")
    fig.update_yaxes(title="Potencia", range=[0, 1])

    return plot_layout(fig, "Curva de potencia para ANOVA")


# =============================================================================
# CARGA BASE
# =============================================================================

df_raw = load_data()

if df_raw is None:
    st.error(
        "No se encontró el archivo CSV. Ubica el archivo en "
        "`data/social_media_vs_productivity.csv`."
    )
    st.stop()

missing = validate_columns(df_raw)

if missing:
    st.warning(
        "Faltan algunas columnas esperadas en la base: "
        + ", ".join(missing)
    )

df = prepare_data(df_raw)

if df.empty:
    st.error("La base de datos está vacía.")
    st.stop()


# =============================================================================
# ENCABEZADO PRINCIPAL
# =============================================================================

st.markdown(
    """
    <div class="main-title">
        <h1>Redes sociales y productividad</h1>
        <p><b>Aplicación de muestreo estadístico y diseño experimental</b></p>
        <p>Asignatura: Diseño de Experimentos</p>
        <p>Docente: Javier Mauricio Sierra</p>
        <p>Estudiantes: Lina María Galvis Barragán y Julián Mateo Valderrama Tibaduiza</p>
        <p>Universidad Santo Tomás</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# TABS PRINCIPALES
# =============================================================================

tabs = st.tabs([
    "📌 Resumen",
    "📊 Exploración",
    "🧾 Variables",
    "🧮 Muestreo",
    "🧪 DBCA",
    "📈 ANOVA y Tukey",
    "✅ Supuestos",
    "⚡ Potencia",
    "🔗 Integración",
    "⚖️ Pros y limitaciones"
])


# =============================================================================
# TAB 1: RESUMEN
# =============================================================================

with tabs[0]:
    st.header("Resumen ejecutivo")

    info_box(
        """
        <b>Pregunta de investigación:</b><br>
        ¿El nivel de uso diario de redes sociales se asocia con diferencias en la productividad real,
        controlando por el tipo de trabajo?
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Registros", f"{len(df):,}", "Unidades disponibles en la base")
    with c2:
        metric_card("Variables", f"{df.shape[1]}", "Originales y derivadas")
    with c3:
        metric_card(
            "Productividad media",
            f"{df['actual_productivity_score'].mean():.2f}",
            "Escala de 0 a 10"
        )
    with c4:
        metric_card(
            "Tiempo medio en redes",
            f"{df['daily_social_media_time'].mean():.2f} h",
            "Horas diarias"
        )

    st.markdown("### Enfoque metodológico")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="section-card">
            <h4>Fase I — Diseño muestral</h4>
            <ul>
                <li>Muestreo estratificado proporcional.</li>
                <li>Estrato: tipo de trabajo.</li>
                <li>Variable de interés: productividad real.</li>
                <li>Estimación puntual e intervalos de confianza.</li>
                <li>Comparación con MAS.</li>
                <li>Estimadores de razón y regresión.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="section-card">
            <h4>Fase II — Diseño experimental</h4>
            <ul>
                <li>Diseño en Bloques Completos Aleatorizados.</li>
                <li>Tratamiento: nivel de uso de redes.</li>
                <li>Bloque: tipo de trabajo.</li>
                <li>Respuesta: productividad real.</li>
                <li>ANOVA, Tukey, supuestos y potencia.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    warn_box(
        """
        <b>Precaución:</b> la base es observacional. 
        El análisis permite estudiar asociaciones y diferencias estadísticas, pero no demostrar causalidad fuerte.
        """
    )


# =============================================================================
# TAB 2: EXPLORACIÓN
# =============================================================================

with tabs[1]:
    st.header("Análisis exploratorio de datos")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Filas", f"{df.shape[0]:,}", "Registros de la base")
    with c2:
        metric_card("Columnas", f"{df.shape[1]:,}", "Variables disponibles")
    with c3:
        missing_pct = 100 * df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        metric_card("Datos faltantes", f"{missing_pct:.2f}%", "Sobre toda la base")

    st.subheader("Estadísticas descriptivas")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    desc = (
        df[num_cols]
        .describe()
        .T
        .reset_index()
        .rename(columns={
            "index": "Variable",
            "count": "N",
            "mean": "Media",
            "std": "DE",
            "min": "Mínimo",
            "25%": "Q1",
            "50%": "Mediana",
            "75%": "Q3",
            "max": "Máximo"
        })
    )

    st.dataframe(desc.round(3), use_container_width=True)

    interpretation_box(
        "Interpretación para sustentación",
        """
        Esta tabla resume el comportamiento general de las variables numéricas.
        Para sustentar, conviene mencionar la media de productividad real, el promedio de horas en redes,
        el estrés, las horas de sueño y la satisfacción laboral, porque estas variables conectan directamente
        con la pregunta del proyecto.
        """
    )

    st.subheader("Distribuciones principales")

    c1, c2 = st.columns(2)

    with c1:
        fig = histogram_fig(
            df,
            "actual_productivity_score",
            "Distribución de la productividad real",
            "Productividad real (0–10)",
            COLORS["blue_mid"]
        )
        show_plot(fig, "eda_hist_productividad")

        interpretation_box(
            "Interpretación",
            """
            La distribución muestra cómo se concentra la productividad real en la población analizada.
            Si la mayor parte de observaciones se ubica cerca del centro de la escala, se puede afirmar que
            la productividad promedio es intermedia. Esta gráfica también permite revisar si hay asimetrías
            o valores extremos que puedan afectar la media.
            """
        )

    with c2:
        fig = histogram_fig(
            df,
            "daily_social_media_time",
            "Distribución del tiempo diario en redes sociales",
            "Horas diarias en redes",
            COLORS["blue_light"]
        )
        show_plot(fig, "eda_hist_tiempo_redes")

        interpretation_box(
            "Interpretación",
            """
            Esta gráfica permite observar si la mayoría de personas usa redes durante pocas horas o si existen
            usuarios con exposición muy alta. Esto justifica crear los niveles bajo, medio y alto de uso de redes,
            que luego se usan como tratamiento en el diseño experimental.
            """
        )

    st.subheader("Productividad por grupos")

    c1, c2 = st.columns(2)

    with c1:
        fig = boxplot_fig(
            df,
            "social_media_level",
            "actual_productivity_score",
            "Productividad real según nivel de uso de redes",
            "Nivel de uso de redes",
            "Productividad real"
        )
        show_plot(fig, "eda_box_redes")

        interpretation_box(
            "Interpretación",
            """
            Este boxplot compara la productividad real entre los niveles bajo, medio y alto de uso de redes.
            Si la mediana disminuye al pasar de bajo a alto, existe una tendencia descriptiva negativa.
            Sin embargo, la decisión estadística formal no se toma aquí, sino en el ANOVA del DBCA.
            """
        )

    with c2:
        fig = boxplot_fig(
            df,
            "job_type",
            "actual_productivity_score",
            "Productividad real según tipo de trabajo",
            "Tipo de trabajo",
            "Productividad real"
        )
        show_plot(fig, "eda_box_trabajo")

        interpretation_box(
            "Interpretación",
            """
            Este gráfico permite observar si la productividad cambia entre tipos de trabajo.
            Por esta razón, el tipo de trabajo se usa como estrato en la fase de muestreo y como bloque en el DBCA:
            ayuda a controlar una fuente externa de variabilidad.
            """
        )

    st.subheader("Relaciones bivariadas")

    c1, c2 = st.columns(2)

    with c1:
        fig = scatter_fig(
            df,
            "daily_social_media_time",
            "actual_productivity_score",
            "Tiempo en redes vs productividad real",
            "Horas diarias en redes",
            "Productividad real"
        )
        show_plot(fig, "eda_scatter_redes")

        interpretation_box(
            "Interpretación",
            """
            La recta de tendencia resume la asociación entre tiempo en redes y productividad real.
            Una pendiente negativa sugiere que un mayor uso de redes se asocia con menor productividad.
            No obstante, como los datos son observacionales, esta relación debe presentarse como asociación,
            no como causalidad.
            """
        )

    with c2:
        fig = scatter_fig(
            df,
            "sleep_hours",
            "actual_productivity_score",
            "Sueño vs productividad real",
            "Horas de sueño",
            "Productividad real"
        )
        show_plot(fig, "eda_scatter_sueño")

        interpretation_box(
            "Interpretación",
            """
            Esta gráfica permite revisar si dormir más horas se asocia con mayor productividad.
            Si la pendiente es positiva, se puede explicar que la productividad no depende únicamente del uso
            de redes sociales, sino también de hábitos de bienestar como el sueño.
            """
        )

    st.subheader("Matriz de correlación")

    fig = correlation_heatmap(df)
    show_plot(fig, "eda_corr")

    interpretation_box(
        "Interpretación",
        """
        La matriz de correlación permite identificar relaciones lineales entre variables numéricas.
        La relación entre productividad percibida y productividad real es especialmente importante porque
        justifica el uso de estimadores auxiliares de razón o regresión en la fase muestral.
        """
    )


# =============================================================================
# TAB 3: VARIABLES
# =============================================================================

with tabs[2]:
    st.header("Selección, conservación y uso de variables")

    variables = [
        ["age", "Numérica", "Exploratoria / control", "Sí", "Caracteriza la muestra y permite revisar diferencias por edad."],
        ["gender", "Categórica", "Exploratoria", "Sí", "Describe la composición de la base."],
        ["job_type", "Categórica", "Estrato y bloque", "Sí", "Se usa como estrato en muestreo y como bloque en DBCA."],
        ["daily_social_media_time", "Numérica", "Variable explicativa base", "Sí", "Permite construir social_media_level."],
        ["social_platform_preference", "Categórica", "Exploratoria", "Sí", "Describe preferencias digitales."],
        ["number_of_notifications", "Numérica", "Exploratoria", "Sí", "Puede relacionarse con distracción y estrés."],
        ["work_hours_per_day", "Numérica", "Exploratoria / control", "Sí", "Contextualiza la productividad según carga laboral."],
        ["perceived_productivity_score", "Numérica", "Variable auxiliar", "Sí", "Se usa para estimadores de razón y regresión."],
        ["actual_productivity_score", "Numérica", "Respuesta principal", "Sí", "Variable objetivo del muestreo y del ANOVA."],
        ["stress_level", "Numérica", "Exploratoria", "Sí", "Variable de bienestar relacionada con productividad."],
        ["sleep_hours", "Numérica", "Exploratoria", "Sí", "Variable de bienestar potencialmente asociada con productividad."],
        ["screen_time_before_sleep", "Numérica", "Exploratoria", "Sí", "Analiza hábitos digitales antes de dormir."],
        ["breaks_during_work", "Numérica", "Exploratoria", "Sí", "Relacionada con pausas durante la jornada."],
        ["uses_focus_apps", "Binaria", "Exploratoria", "Sí", "Compara productividad según uso de apps de enfoque."],
        ["has_digital_wellbeing_enabled", "Binaria", "Exploratoria", "Sí", "Relacionada con autorregulación digital."],
        ["coffee_consumption_per_day", "Numérica", "Exploratoria", "Sí", "Variable complementaria de hábitos diarios."],
        ["days_feeling_burnout_per_month", "Numérica", "Exploratoria", "Sí", "Indicador de agotamiento."],
        ["weekly_offline_hours", "Numérica", "Exploratoria", "Sí", "Mide desconexión digital semanal."],
        ["job_satisfaction_score", "Numérica", "Exploratoria", "Sí", "Puede relacionarse con productividad."],
        ["social_media_level", "Categórica derivada", "Tratamiento", "Sí", "Define los niveles bajo, medio y alto de redes."],
        ["productivity_gap", "Numérica derivada", "Exploratoria", "Sí", "Diferencia entre productividad percibida y real."]
    ]

    var_df = pd.DataFrame(
        variables,
        columns=["Variable", "Tipo", "Rol en el proyecto", "Se usa", "Justificación"]
    )

    st.dataframe(var_df, use_container_width=True)

    interpretation_box(
        "Interpretación para sustentación",
        """
        Ninguna variable importante se descarta sin justificación.
        Algunas variables no hacen parte del modelo principal para no sobrecargar el diseño,
        pero se conservan como variables exploratorias o complementarias.
        La variable respuesta es productividad real; el tratamiento es el nivel de redes; y el bloque/estrato es tipo de trabajo.
        """
    )


# =============================================================================
# TAB 4: MUESTREO
# =============================================================================

with tabs[3]:
    st.header("Fase I: Diseño muestral")

    info_box(
        """
        <b>Diseño seleccionado:</b> muestreo estratificado proporcional.<br>
        <b>Estrato:</b> tipo de trabajo.<br>
        <b>Variable de interés:</b> productividad real.<br>
        <b>Nivel de confianza:</b> 95%.<br>
        <b>Margen de error:</b> 0.10 puntos.
        """
    )

    ss = sample_size_fpc(
        df["actual_productivity_score"],
        error_margin=0.10,
        confidence=0.95
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Población efectiva", f"{ss['N']:,}", "Registros con productividad real")
    with c2:
        metric_card("Varianza estimada", f"{ss['s2']:.3f}", "Desde la base completa")
    with c3:
        metric_card("n₀", f"{ss['n0']:.0f}", "Sin corrección finita")
    with c4:
        metric_card("n final", f"{ss['n']:,}", "Con corrección finita")

    interpretation_box(
        "Interpretación del tamaño de muestra",
        f"""
        Para estimar la media de productividad real con 95% de confianza y margen de error de 0.10 puntos,
        se requieren aproximadamente {ss['n']:,} observaciones. 
        La corrección por población finita reduce el tamaño necesario porque el marco muestral disponible es conocido.
        """
    )

    allocation = proportional_allocation(
        df,
        ss["n"],
        strata_col="job_type",
        y_col="actual_productivity_score"
    )

    st.subheader("Afijación proporcional por estrato")

    st.dataframe(allocation.round(3), use_container_width=True)

    interpretation_box(
        "Interpretación de la afijación",
        """
        La afijación proporcional asigna más observaciones a los tipos de trabajo con mayor presencia en la base.
        Esto garantiza que todos los estratos estén representados y que la muestra conserve la estructura de la población.
        """
    )

    sample = draw_stratified_sample(
        df,
        allocation,
        strata_col="job_type",
        y_col="actual_productivity_score"
    )

    ybar_strat, se_strat, ci_l_strat, ci_u_strat = stratified_mean_ci(sample, allocation)
    mas_mean, mas_se, mas_l, mas_u = mas_mean_ci(df, ss["n"])
    pop_mean = df["actual_productivity_score"].mean()

    results = pd.DataFrame({
        "Estimador": ["Población completa", "MAS", "Estratificado proporcional"],
        "Media": [pop_mean, mas_mean, ybar_strat],
        "Error estándar": [np.nan, mas_se, se_strat],
        "IC inferior": [np.nan, mas_l, ci_l_strat],
        "IC superior": [np.nan, mas_u, ci_u_strat]
    })

    results["Amplitud IC"] = results["IC superior"] - results["IC inferior"]

    st.subheader("Comparación de estimadores")

    st.dataframe(results.round(4), use_container_width=True)

    fig = ci_comparison_fig(results)
    show_plot(fig, "muestreo_ic_comparacion")

    interpretation_box(
        "Interpretación de los intervalos",
        """
        Esta gráfica compara la precisión de los estimadores.
        Un intervalo más estrecho representa menor incertidumbre.
        La media de la población completa se muestra como referencia académica, mientras que el MAS y el estimador estratificado
        representan escenarios de muestreo.
        """
    )

    st.subheader("Estimadores auxiliares")

    aux = auxiliary_estimators(df, sample)

    st.dataframe(aux.round(4), use_container_width=True)

    fig = scatter_fig(
        sample,
        "perceived_productivity_score",
        "actual_productivity_score",
        "Productividad percibida vs productividad real",
        "Productividad percibida",
        "Productividad real"
    )

    show_plot(fig, "muestreo_scatter_auxiliares")

    interpretation_box(
        "Interpretación de estimadores auxiliares",
        """
        La productividad percibida se usa como variable auxiliar porque tiene relación con la productividad real.
        Si el estimador de razón o de regresión reduce el error absoluto frente al estimador directo,
        entonces la información auxiliar mejora la precisión de la estimación.
        """
    )


# =============================================================================
# TAB 5: DBCA
# =============================================================================

with tabs[4]:
    st.header("Fase II: Diseño en Bloques Completos Aleatorizados")

    info_box(
        """
        <b>Diseño elegido:</b> DBCA.<br>
        <b>Tratamiento:</b> nivel de uso de redes sociales: Bajo, Medio y Alto.<br>
        <b>Bloque:</b> tipo de trabajo.<br>
        <b>Variable respuesta:</b> productividad real.
        """
    )

    dbca, balance_table, n_cell = create_balanced_dbca(df)

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Tratamientos", "3", "Bajo, Medio y Alto")
    with c2:
        metric_card("Bloques", f"{balance_table.shape[1]}", "Tipos de trabajo usados en el DBCA")
    with c3:
        metric_card("Réplicas por celda", f"{n_cell:,}", "Submuestra balanceada")

    st.subheader("Balance del diseño")

    st.dataframe(balance_table, use_container_width=True)

    interpretation_box(
        "Interpretación del balance",
        """
        Para aplicar el DBCA de forma clara, se construye una base balanceada con igual número de observaciones
        por combinación tratamiento × bloque. Esto permite comparar los niveles de redes sociales controlando
        por tipo de trabajo.
        """
    )

    st.subheader("Justificación del DBCA")

    st.markdown(
        """
        <div class="section-card">
        <ul>
            <li><b>DBCA:</b> es adecuado porque el tipo de trabajo puede modificar la productividad y debe controlarse.</li>
            <li><b>DBIA:</b> no se elige porque ignoraría diferencias entre tipos de trabajo.</li>
            <li><b>DCL:</b> no se elige porque requeriría dos factores de bloqueo con estructura más rígida.</li>
            <li><b>Youden o greco-latino:</b> no son necesarios para esta estructura de datos.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    if dbca.empty or n_cell == 0:
        warn_box(
            """
            No se pudo construir el DBCA balanceado con los datos disponibles.
            Revisa la tabla de balance para identificar si todos los bloques tienen observaciones
            en los tres niveles de uso de redes sociales.
            """
        )
    else:
        fig = means_dbca_fig(dbca)
        show_plot(fig, "dbca_medias_tratamiento")

        interpretation_box(
            "Interpretación de medias por tratamiento",
            """
            La gráfica muestra la productividad promedio en los niveles bajo, medio y alto de uso de redes.
            Si el grupo alto presenta menor media, existe una tendencia descriptiva de menor productividad asociada
            con mayor uso de redes. La significancia de esta diferencia se evalúa formalmente con el ANOVA.
            """
        )


# =============================================================================
# TAB 6: ANOVA Y TUKEY
# =============================================================================

with tabs[5]:
    st.header("ANOVA y comparaciones múltiples")

    dbca, balance_table, n_cell = create_balanced_dbca(df)
    model, anova = fit_dbca_anova(dbca)

    if dbca.empty or n_cell == 0:
        st.error("No se puede realizar el ANOVA porque no se logró construir una base DBCA balanceada.")
        st.dataframe(balance_table, use_container_width=True)
    elif model is None:
        st.error("No hay datos suficientes para ajustar el modelo DBCA.")
    else:
        anova_display = anova.copy()
        anova_display["Decisión"] = anova_display["PR(>F)"].apply(
            lambda p: "Significativo" if pd.notna(p) and p < ALPHA else "No significativo"
        )

        st.subheader("Tabla ANOVA")

        st.dataframe(anova_display.round(5), use_container_width=True)

        p_trat = get_anova_pvalue(anova, "Tratamiento: nivel de redes")
        p_bloque = get_anova_pvalue(anova, "Bloque: tipo de trabajo")

        c1, c2 = st.columns(2)

        with c1:
            if pd.notna(p_trat) and p_trat < ALPHA:
                good_box(
                    f"""
                    <b>Tratamiento significativo.</b><br>
                    {pvalue_decision(p_trat)}
                    Existen diferencias estadísticamente significativas en productividad según nivel de uso de redes.
                    """
                )
            else:
                warn_box(
                    f"""
                    <b>Tratamiento no significativo.</b><br>
                    {pvalue_decision(p_trat)}
                    No hay evidencia suficiente para afirmar diferencias de productividad entre niveles de redes.
                    """
                )

        with c2:
            if pd.notna(p_bloque) and p_bloque < ALPHA:
                good_box(
                    f"""
                    <b>Bloque significativo.</b><br>
                    {pvalue_decision(p_bloque)}
                    El tipo de trabajo explica variabilidad relevante, por lo que el DBCA está justificado.
                    """
                )
            else:
                warn_box(
                    f"""
                    <b>Bloque no significativo.</b><br>
                    {pvalue_decision(p_bloque)}
                    En esta muestra balanceada, el tipo de trabajo no muestra efecto estadístico fuerte.
                    """
                )

        interpretation_box(
            "Interpretación del ANOVA",
            """
            El ANOVA compara la variabilidad explicada por los tratamientos y bloques contra la variabilidad residual.
            El efecto de tratamiento responde directamente a la pregunta del proyecto.
            El efecto de bloque indica si el tipo de trabajo aporta variabilidad y si fue útil controlarlo.
            """
        )

        if dbca.empty or n_cell == 0:
            warn_box(
                """
                No se muestra la gráfica de medias porque no se logró construir una base DBCA balanceada.
                """
            )
        else:
            fig = means_dbca_fig(dbca)
            show_plot(fig, "anova_medias_tratamiento")

            interpretation_box(
                "Interpretación de la gráfica de medias",
                """
                Las barras representan la productividad media por nivel de redes sociales.
                Si el ANOVA es significativo y las medias están separadas, se fortalece la evidencia de diferencias reales
                entre niveles. Si el ANOVA no es significativo, la gráfica debe entenderse solo como descriptiva.
                """
            )

        st.subheader("Comparaciones múltiples — Tukey")

        tukey = tukey_table(dbca)
        st.dataframe(tukey, use_container_width=True)

        if pd.notna(p_trat) and p_trat < ALPHA:
            interpretation_box(
                "Interpretación de Tukey",
                """
                Como el ANOVA del tratamiento fue significativo, Tukey permite identificar entre cuáles niveles
                de uso de redes existen diferencias. Una comparación con p ajustado menor que 0.05 indica diferencia
                estadísticamente significativa entre esos dos niveles.
                """
            )
        else:
            interpretation_box(
                "Interpretación de Tukey",
                """
                Como el ANOVA del tratamiento no fue significativo, Tukey no debe usarse como evidencia confirmatoria.
                Puede revisarse de forma exploratoria, pero la conclusión principal sigue siendo que no hay evidencia
                suficiente de diferencias entre niveles de uso de redes.
                """
            )


# =============================================================================
# TAB 7: SUPUESTOS
# =============================================================================

with tabs[6]:
    st.header("Verificación de supuestos del modelo")

    dbca, balance_table, n_cell = create_balanced_dbca(df)
    model, anova = fit_dbca_anova(dbca)

    if dbca.empty or n_cell == 0:
        st.error("No se pueden evaluar los supuestos porque no se logró construir una base DBCA balanceada.")
        st.dataframe(balance_table, use_container_width=True)
    elif model is None:
        st.error("No hay datos suficientes para evaluar supuestos.")
    else:
        fig_hist, fig_qq, fig_rv, fig_seq = residual_diagnostic_figs(model)

        c1, c2 = st.columns(2)

        with c1:
            show_plot(fig_hist, "supuestos_hist_residuos")
            interpretation_box(
                "Interpretación",
                """
                El histograma permite revisar si los residuos se distribuyen aproximadamente alrededor de cero.
                Una forma simétrica y centrada favorece el supuesto de normalidad de los errores.
                """
            )

        with c2:
            show_plot(fig_qq, "supuestos_qq_residuos")
            interpretation_box(
                "Interpretación",
                """
                En el Q-Q plot, si los puntos siguen la línea de referencia, la normalidad es razonable.
                Con muestras grandes pueden aparecer desviaciones en las colas sin invalidar necesariamente el ANOVA.
                """
            )

        c1, c2 = st.columns(2)

        with c1:
            show_plot(fig_rv, "supuestos_residuos_ajustados")
            interpretation_box(
                "Interpretación",
                """
                Este gráfico evalúa homocedasticidad. Si los puntos se dispersan de forma aleatoria alrededor de cero
                y no aparece una forma de embudo, el supuesto de varianza constante es razonable.
                """
            )

        with c2:
            show_plot(fig_seq, "supuestos_residuos_secuencia")
            interpretation_box(
                "Interpretación",
                """
                Este gráfico revisa independencia visualmente. Si no se observan patrones, ciclos o tendencias,
                no hay señales fuertes de dependencia entre residuos.
                """
            )

        residuals = model.resid

        sample_res = residuals.sample(
            n=min(5000, len(residuals)),
            random_state=123
        )

        shapiro_stat, shapiro_p = stats.shapiro(sample_res)

        groups = [
            group["actual_productivity_score"].values
            for _, group in dbca.groupby("social_media_level")
        ]

        levene_stat, levene_p = stats.levene(*groups, center="median")
        dw_stat = durbin_watson(residuals)

        tests = pd.DataFrame({
            "Supuesto": [
                "Normalidad",
                "Homocedasticidad",
                "Independencia"
            ],
            "Prueba / criterio": [
                "Shapiro-Wilk sobre máximo 5000 residuos",
                "Levene por nivel de redes",
                "Durbin-Watson"
            ],
            "Estadístico": [
                shapiro_stat,
                levene_stat,
                dw_stat
            ],
            "p-valor": [
                shapiro_p,
                levene_p,
                np.nan
            ],
            "Interpretación": [
                pvalue_decision(shapiro_p),
                pvalue_decision(levene_p),
                "Valores cercanos a 2 sugieren ausencia de autocorrelación fuerte."
            ]
        })

        st.subheader("Pruebas formales")

        st.dataframe(tests.round(5), use_container_width=True)

        interpretation_box(
            "Lectura para sustentación",
            """
            Las pruebas formales deben interpretarse junto con los gráficos.
            Shapiro-Wilk puede rechazar normalidad con muestras grandes por desviaciones pequeñas.
            Por eso, el Q-Q plot y la robustez del ANOVA son esenciales para sustentar la idoneidad del modelo.
            """
        )


# =============================================================================
# TAB 8: POTENCIA
# =============================================================================

with tabs[7]:
    st.header("Potencia y tamaño de muestra experimental")

    dbca, balance_table, n_cell = create_balanced_dbca(df)
    model, anova = fit_dbca_anova(dbca)

    power_info = compute_power(anova, dbca)

    if dbca.empty or n_cell == 0:
        st.error("No se puede calcular la potencia porque no se logró construir una base DBCA balanceada.")
        st.dataframe(balance_table, use_container_width=True)
    elif power_info is None:
        st.error("No fue posible calcular la potencia.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("η²", f"{power_info['eta2']:.4f}", "Tamaño de efecto")
        with c2:
            metric_card("f de Cohen", f"{power_info['f_cohen']:.4f}", "Efecto para ANOVA")
        with c3:
            metric_card("Potencia actual", f"{power_info['current_power']:.3f}", "Probabilidad de detectar efecto")
        with c4:
            metric_card("n requerido/grupo", f"{power_info['n_required_group']}", "Para potencia 0.80")

        fig = power_curve_fig(power_info)
        show_plot(fig, "potencia_curva")

        interpretation_box(
            "Interpretación de potencia",
            """
            La potencia estadística indica la probabilidad de detectar un efecto real si efectivamente existe.
            Una potencia de 0.80 suele considerarse adecuada. Si el tamaño del efecto es pequeño, se necesita
            una muestra mayor para detectarlo con seguridad.
            """
        )


# =============================================================================
# TAB 9: INTEGRACIÓN
# =============================================================================

with tabs[8]:
    st.header("Integración de resultados")

    dbca, balance_table, n_cell = create_balanced_dbca(df)
    model, anova = fit_dbca_anova(dbca)

    ss = sample_size_fpc(
        df["actual_productivity_score"],
        error_margin=0.10,
        confidence=0.95
    )

    allocation = proportional_allocation(
        df,
        ss["n"],
        strata_col="job_type",
        y_col="actual_productivity_score"
    )

    sample = draw_stratified_sample(
        df,
        allocation,
        strata_col="job_type",
        y_col="actual_productivity_score"
    )

    ybar_strat, se_strat, ci_l_strat, ci_u_strat = stratified_mean_ci(sample, allocation)

    p_trat = get_anova_pvalue(anova, "Tratamiento: nivel de redes")

    info_box(
        f"""
        <b>Resultado de la fase muestral:</b><br>
        La productividad real media estimada es <b>{ybar_strat:.3f}</b>, con IC 95%:
        [<b>{ci_l_strat:.3f}</b>, <b>{ci_u_strat:.3f}</b>].
        """
    )

    if pd.notna(p_trat) and p_trat < ALPHA:
        good_box(
            f"""
            <b>Resultado de la fase experimental:</b><br>
            El DBCA encontró evidencia estadística de diferencias entre niveles de uso de redes sociales
            sobre la productividad real, con p = {format_p(p_trat)}.
            """
        )
    else:
        warn_box(
            f"""
            <b>Resultado de la fase experimental:</b><br>
            El DBCA no encontró evidencia estadística suficiente de diferencias entre niveles de redes,
            con p = {format_p(p_trat)}.
            """
        )

    if dbca.empty or n_cell == 0:
        warn_box(
            """
            No se muestra la gráfica integradora porque no se logró construir una base DBCA balanceada.
            Aun así, se conserva la estimación muestral y se reporta la limitación del análisis experimental.
            """
        )
        if not balance_table.empty:
            st.dataframe(balance_table, use_container_width=True)
    else:
        fig = px.box(
            dbca,
            x="job_type",
            y="actual_productivity_score",
            color="social_media_level",
            color_discrete_map=PAL_SML
        )

        fig.update_xaxes(title="Tipo de trabajo")
        fig.update_yaxes(title="Productividad real")

        fig = plot_layout(
            fig,
            "Productividad real por tipo de trabajo y nivel de redes"
        )

        show_plot(fig, "integracion_boxplot")

        interpretation_box(
            "Interpretación integradora",
            """
            Esta gráfica conecta ambas fases del proyecto. El tipo de trabajo se usó como estrato en el muestreo
            y como bloque en el DBCA. Si el patrón entre niveles de redes se mantiene dentro de varios tipos de trabajo,
            la conclusión descriptiva gana coherencia. Si cambia mucho entre bloques, debe discutirse el papel del contexto laboral.
            """
        )

    st.subheader("Respuesta al problema de investigación")

    st.markdown(
        """
        <div class="section-card">
        El análisis permite responder que el uso de redes sociales se relaciona con la productividad principalmente
        como una asociación estadística y descriptiva. La fase muestral aporta una estimación precisa de la productividad
        media de la población, mientras que el DBCA permite contrastar si los niveles de uso de redes presentan diferencias
        controlando por tipo de trabajo. Dado que los datos son observacionales, la conclusión debe evitar afirmaciones
        causales fuertes.
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# TAB 10: PROS Y LIMITACIONES
# =============================================================================

with tabs[9]:
    st.header("Pros, contras y limitaciones")

    pros_contras = pd.DataFrame({
        "Aspectos favorables": [
            "Base de datos amplia.",
            "Variables relevantes sobre productividad, sueño, estrés y hábitos digitales.",
            "Permite estratificar por tipo de trabajo.",
            "Permite aplicar DBCA controlando por bloque.",
            "Permite comparar estimadores simples y auxiliares.",
            "El tablero facilita interpretación visual e interactiva."
        ],
        "Aspectos desfavorables o precauciones": [
            "Base observacional, no experimental estricta.",
            "No se puede demostrar causalidad fuerte.",
            "Posible sesgo de autoselección.",
            "Variables autoinformadas pueden tener error de medición.",
            "El balance del DBCA requiere submuestreo por celda.",
            "Puede haber variables de confusión no incluidas."
        ]
    })

    st.dataframe(pros_contras, use_container_width=True)

    interpretation_box(
        "Interpretación",
        """
        Los aspectos favorables fortalecen la utilidad descriptiva y analítica del estudio.
        Los aspectos desfavorables limitan principalmente la validez causal y la generalización.
        Por tanto, las conclusiones deben hablar de asociación y evidencia estadística, no de causalidad definitiva.
        """
    )

    st.subheader("Limitaciones principales")

    st.markdown(
        """
        <div class="section-card">
        <ol>
            <li>La base es secundaria y no fue recolectada directamente por los investigadores.</li>
            <li>No hay asignación aleatoria real al nivel de uso de redes sociales.</li>
            <li>Variables como estrés, sueño y productividad percibida pueden estar autoinformadas.</li>
            <li>No se conoce con certeza la representatividad geográfica.</li>
            <li>El análisis experimental es analítico, no experimental puro.</li>
            <li>Puede haber variables de confusión no observadas.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True
    )
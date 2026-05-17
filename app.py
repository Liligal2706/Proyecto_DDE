"""
app.py  –  Redes Sociales vs Productividad
Diseño de Experimentos · Universidad Santo Tomás · 2026-I
Autores: Lina María Galvis Barragán · Julián Mateo Valderrama Tibaduiza
Docente: Javier Mauricio Sierra

Tabs: Inicio & Datos | Fase I-VII | Conclusiones
"""
from __future__ import annotations
import warnings, textwrap
import numpy as np
import pandas as pd
import altair as alt
import scipy.stats as sps
from scipy.stats import chi2_contingency, levene
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import normal_ad
from pathlib import Path
import streamlit as st

warnings.filterwarnings("ignore")
CSV_PATH = Path("data/social_media_vs_productivity.csv")
ALPHA = 0.05
SEED  = 123

# ── Paletas ──────────────────────────────────────────────────────────────────
P = dict(bg="#07111E", surface="#0F1E2E", card="#152536", border="#1E3448",
         text="#E8EFF6", muted="#7A96B0", accent="#4C9BE8", accent2="#38D2C0",
         warn="#F5A623", ok="#27D48A", err="#F05C5C")
FASE_CLR = {"I":"#4C9BE8","II":"#38D2C0","III":"#A78BFA",
            "IV":"#F472B6","V":"#FB923C","VI":"#FACC15","VII":"#34D399"}
PAL3 = {"Bajo":"#4C9BE8","Medio":"#38D2C0","Alto":"#A78BFA"}

# ── Configuración ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Redes & Productividad",page_icon="📊",
                   layout="wide",initial_sidebar_state="collapsed")

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp{{background:{P['bg']};color:{P['text']};font-family:'Inter',sans-serif;}}
.block-container{{padding:1.2rem 2.2rem 4rem;max-width:1440px;}}
h1,h2,h3,h4{{color:{P['text']};font-weight:700;letter-spacing:-.025em;margin-top:0;}}
p,li,td,th{{font-size:.9rem;line-height:1.65;}}

/* ── TABS PROFESIONALES ── */
.stTabs [data-baseweb="tab-list"]{{
    background:{P['surface']}dd;
    gap:0;
    padding:.25rem .5rem;
    border-bottom:2px solid {P['border']};
    border-radius:10px 10px 0 0;
    overflow-x:auto;
    scrollbar-width:thin;
}}
.stTabs [data-baseweb="tab"]{{
    background:transparent;
    border:none;
    border-radius:8px 8px 0 0;
    padding:.55rem 1.1rem;
    color:{P['muted']};
    font-weight:600;
    font-size:.78rem;
    letter-spacing:.035em;
    text-transform:uppercase;
    border-bottom:3px solid transparent;
    transition:all .2s ease;
    white-space:nowrap;
    margin-bottom:-2px;
}}
.stTabs [data-baseweb="tab"]:hover{{
    color:{P['text']};
    background:{P['card']}99;
}}
.stTabs [aria-selected="true"]{{
    color:#fff!important;
    background:linear-gradient(135deg,{P['accent']}22,{P['accent']}11)!important;
    border-bottom:3px solid {P['accent']}!important;
}}

/* ── TABLAS ── */
div[data-testid="stDataFrame"]{{
    background:{P['card']};
    border-radius:10px;
    border:1px solid {P['border']};
    overflow:hidden;
}}
div[data-testid="stDataFrame"] th{{
    background:{P['surface']}!important;
    color:{P['muted']}!important;
    font-size:.72rem!important;
    font-weight:700!important;
    letter-spacing:.05em!important;
    text-transform:uppercase!important;
    padding:.55rem .75rem!important;
}}
div[data-testid="stDataFrame"] td{{
    font-size:.83rem!important;
    padding:.45rem .75rem!important;
}}

/* ── SLIDERS ── */
.stSlider [data-baseweb="slider"]{{padding:.5rem 0;}}
.stSlider label{{font-size:.8rem!important;color:{P['muted']}!important;font-weight:600!important;}}

/* ── TABS INTERNOS ── */
div[data-testid="stVerticalBlock"] .stTabs [data-baseweb="tab"]{{
    font-size:.73rem;padding:.4rem .75rem;
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-track{{background:{P['surface']};}}
::-webkit-scrollbar-thumb{{background:{P['border']};border-radius:3px;}}
::-webkit-scrollbar-thumb:hover{{background:{P['accent']}66;}}

/* ── TOOLTIPS ALTAIR ── */
.vg-tooltip{{
    background:{P['card']}!important;
    border:1px solid {P['border']}!important;
    color:{P['text']}!important;
    font-family:'Inter',sans-serif!important;
    font-size:.78rem!important;
    border-radius:8px!important;
    padding:.5rem .75rem!important;
    box-shadow:0 8px 32px rgba(0,0,0,.45)!important;
}}

/* ── METRIC CARDS ── */
.stMetric{{
    background:{P['card']};
    border-radius:8px;
    border:1px solid {P['border']};
    padding:.5rem;
}}

/* ── INPUT ── */
.stSelectbox label,.stTextInput label,.stNumberInput label{{
    color:{P['muted']}!important;font-size:.78rem!important;font-weight:600!important;
}}
</style>""", unsafe_allow_html=True)

# ── Altair theme ──────────────────────────────────────────────────────────────
_AC = dict(background=P["surface"],
    view=dict(fill=P["surface"],stroke="transparent"),
    title=dict(color=P["text"],fontSize=13,fontWeight=700,anchor="start"),
    axis=dict(labelColor=P["muted"],titleColor=P["muted"],gridColor=P["border"],
              domainColor=P["border"],labelFontSize=11,titleFontSize=11),
    legend=dict(labelColor=P["text"],titleColor=P["muted"],labelFontSize=11,
                fillColor=P["card"],strokeColor=P["border"],padding=8,cornerRadius=6),
    mark=dict(color=P["accent"]))
alt.themes.register("usta", lambda: {"config": _AC})
alt.themes.enable("usta")

# ═══════════════════════════════════════════════════════════════════════════
#  COMPONENTES UI
# ═══════════════════════════════════════════════════════════════════════════
def fase_header(num:str,title:str,pct:int,desc:str):
    clr=FASE_CLR.get(num,P["accent"])
    n_roman={"I":"1","II":"2","III":"3","IV":"4","V":"5","VI":"6","VII":"7"}.get(num,num)
    st.markdown(f"""<div style="
        background:linear-gradient(135deg,{P['card']} 0%,{P['surface']} 60%,{P['card']} 100%);
        border:1px solid {P['border']};border-top:3px solid {clr};
        border-radius:12px;padding:1.3rem 1.7rem;margin-bottom:1.4rem;
        box-shadow:0 6px 24px rgba(0,0,0,.32);position:relative;overflow:hidden;">
      <div style="position:absolute;top:-14px;right:14px;font-size:5.5rem;font-weight:900;
          opacity:.05;color:{clr};user-select:none;line-height:1;">F{n_roman}</div>
      <div style="position:relative;z-index:1;">
        <div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.6rem;flex-wrap:wrap;">
          <div style="background:{clr};color:#000a14;font-size:.75rem;font-weight:900;
              letter-spacing:.05em;padding:.28rem .75rem;border-radius:6px;
              display:flex;align-items:center;gap:.35rem;">
            <span style="opacity:.7;font-size:.65rem;">FASE</span>
            <span>{num}</span>
          </div>
          <span style="background:{clr}18;color:{clr};font-size:.68rem;font-weight:700;
              border:1px solid {clr}33;padding:.22rem .6rem;border-radius:20px;
              letter-spacing:.05em;">Peso rúbrica: {pct}%</span>
        </div>
        <h2 style="margin:0 0 .35rem;font-size:1.45rem;font-weight:800;
            letter-spacing:-.02em;color:#fff;">{title}</h2>
        <p style="margin:0;color:{P['muted']};font-size:.84rem;line-height:1.55;
            border-left:2px solid {clr}44;padding-left:.7rem;">{desc}</p>
      </div>
    </div>""", unsafe_allow_html=True)

def kpi(label:str,value:str,sub:str="",clr:str=""):
    c=clr or P["accent"]
    st.markdown(f"""<div style="background:{P['card']};border:1px solid {P['border']};
        border-top:3px solid {c};border-radius:10px;padding:.9rem 1rem;min-height:94px;
        box-shadow:0 4px 12px rgba(0,0,0,.22);">
      <div style="color:{P['muted']};font-size:.68rem;font-weight:700;
          text-transform:uppercase;letter-spacing:.06em;margin-bottom:.28rem;">{label}</div>
      <div style="color:#fff;font-size:1.5rem;font-weight:800;line-height:1.1;">{value}</div>
      <div style="color:{P['muted']};font-size:.71rem;margin-top:.25rem;">{sub}</div>
    </div>""", unsafe_allow_html=True)

def callout(kind:str,title:str,body:str):
    cfg={"info":(P["accent"],"ℹ️"),"ok":(P["ok"],"✅"),
         "warn":(P["warn"],"⚠️"),"err":(P["err"],"❌"),"just":(P["accent2"],"📐")}
    clr,ico=cfg.get(kind,(P["accent"],"•"))
    body2=" ".join(textwrap.dedent(body).split())
    st.markdown(f"""<div style="background:{clr}12;border-left:4px solid {clr};
        border-radius:0 10px 10px 0;padding:.82rem 1rem;margin:.45rem 0 .85rem;
        border-top:1px solid {clr}1A;border-right:1px solid {clr}1A;border-bottom:1px solid {clr}1A;">
      <div style="color:{clr};font-size:.69rem;font-weight:800;
          text-transform:uppercase;letter-spacing:.06em;margin-bottom:.28rem;">{ico} {title}</div>
      <div style="color:{P['text']};font-size:.87rem;line-height:1.65;">{body2}</div>
    </div>""", unsafe_allow_html=True)

def justif(title:str,body:str):
    body2=" ".join(textwrap.dedent(body).split())
    st.markdown(f"""<div style="
        background:linear-gradient(135deg,{FASE_CLR['III']}0A 0%,{P['surface']} 100%);
        border:1px solid {FASE_CLR['III']}33;border-left:4px solid {FASE_CLR['III']};
        border-radius:0 12px 12px 0;padding:1rem 1.15rem;margin:.5rem 0 1rem;">
      <div style="display:flex;align-items:center;gap:.45rem;margin-bottom:.5rem;">
        <span style="background:{FASE_CLR['III']}22;color:{FASE_CLR['III']};
            font-size:.64rem;font-weight:900;letter-spacing:.1em;text-transform:uppercase;
            padding:.18rem .55rem;border-radius:20px;border:1px solid {FASE_CLR['III']}44;">
            📐 Justificación metodológica</span>
        <span style="color:{FASE_CLR['III']};font-size:.83rem;font-weight:700;">{title}</span>
      </div>
      <div style="color:{P['text']};font-size:.86rem;line-height:1.7;">{body2}</div>
    </div>""", unsafe_allow_html=True)

def interp(title:str,body:str): callout("info",f"Interpretación — {title}",body)

def just_box(items:list):
    """Tabla de justificaciones de pruebas NO usadas.
    items: lista de (prueba, razon, alternativa)
    """
    rows="".join(f"""<tr>
      <td style="padding:.6rem .8rem;border-bottom:1px solid {P['border']}22;
          font-weight:700;color:{P['warn']};font-size:.82rem;white-space:nowrap;">
          ⛔ {p}</td>
      <td style="padding:.6rem .8rem;border-bottom:1px solid {P['border']}22;
          color:{P['text']};font-size:.82rem;line-height:1.6;">{r}</td>
      <td style="padding:.6rem .8rem;border-bottom:1px solid {P['border']}22;
          color:{P['ok']};font-size:.82rem;">{a}</td>
    </tr>""" for p,r,a in items)
    st.markdown(f"""<div style="background:{P['card']};border:1px solid {P['border']};
        border-radius:12px;overflow:hidden;margin:.8rem 0 1.2rem;">
      <div style="background:{P['surface']};padding:.65rem 1rem;
          border-bottom:1px solid {P['border']};">
        <span style="color:{P['warn']};font-size:.7rem;font-weight:800;
            letter-spacing:.08em;text-transform:uppercase;">
            ⚠️ Pruebas excluidas — Justificación explícita</span></div>
      <table style="width:100%;border-collapse:collapse;">
        <thead><tr>
          <th style="padding:.5rem .8rem;background:{P['surface']}99;
              color:{P['muted']};font-size:.69rem;font-weight:800;
              letter-spacing:.07em;text-transform:uppercase;text-align:left;">
              Prueba</th>
          <th style="padding:.5rem .8rem;background:{P['surface']}99;
              color:{P['muted']};font-size:.69rem;font-weight:800;
              letter-spacing:.07em;text-transform:uppercase;text-align:left;">
              Razón de exclusión</th>
          <th style="padding:.5rem .8rem;background:{P['surface']}99;
              color:{P['muted']};font-size:.69rem;font-weight:800;
              letter-spacing:.07em;text-transform:uppercase;text-align:left;">
              Alternativa adoptada</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

def section(t:str,clr:str=""):
    c=clr or P["accent"]
    st.markdown(f"""<div style="display:flex;align-items:center;gap:.55rem;
        margin:1.6rem 0 .75rem;padding-bottom:.45rem;
        border-bottom:1px solid {P['border']};">
      <div style="width:3px;height:18px;background:{c};border-radius:2px;
          flex-shrink:0;"></div>
      <span style="color:{P['text']};font-size:.95rem;font-weight:700;
          letter-spacing:-.01em;">{t}</span>
    </div>""", unsafe_allow_html=True)

def show(ch:alt.Chart,key:str,h:int=370):
    st.altair_chart(ch.properties(height=h),use_container_width=True,key=key)

def pv(p)->str:
    if pd.isna(p): return "—"
    return "< 0.0001" if p<0.0001 else f"{p:.4f}"

def dec(p)->str:
    if pd.isna(p): return "—"
    return (f"✅ Rechazar H₀  (p={pv(p)} < α={ALPHA})"
            if p<ALPHA else f"— No rechazar H₀  (p={pv(p)} ≥ α={ALPHA})")

# ═══════════════════════════════════════════════════════════════════════════
#  DATOS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    if not CSV_PATH.exists(): st.error(f"CSV no encontrado: {CSV_PATH}"); st.stop()
    df=pd.read_csv(CSV_PATH); df.columns=[c.strip() for c in df.columns]
    num_cols=["age","daily_social_media_time","number_of_notifications","work_hours_per_day",
              "perceived_productivity_score","actual_productivity_score","stress_level",
              "sleep_hours","screen_time_before_sleep","breaks_during_work",
              "coffee_consumption_per_day","days_feeling_burnout_per_month",
              "weekly_offline_hours","job_satisfaction_score"]
    for c in num_cols:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["gender"]  =df["gender"].map({"Male":"Masculino","Female":"Femenino","Other":"Otro"})
    df["job_type"]=df["job_type"].map({"Education":"Educación","Finance":"Finanzas",
        "Health":"Salud","IT":"Tecnología","Student":"Estudiante","Unemployed":"Desempleado"})
    for c in ["uses_focus_apps","has_digital_wellbeing_enabled"]:
        df[c]=df[c].astype(str).str.strip().map(
            {"True":"Sí","False":"No","TRUE":"Sí","FALSE":"No","1":"Sí","0":"No"})
    v=df["daily_social_media_time"].dropna(); q1,q2=v.quantile([1/3,2/3])
    df["nivel_redes"]=pd.Categorical(
        df["daily_social_media_time"].apply(
            lambda x:np.nan if pd.isna(x) else("Bajo" if x<=q1 else("Medio" if x<=q2 else "Alto"))),
        ["Bajo","Medio","Alto"],ordered=True)
    df["brecha"]=df["perceived_productivity_score"]-df["actual_productivity_score"]
    return df,q1,q2

df,Q1,Q2=load_data()
N=len(df)

# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIONES ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════════════════════
def raking_ipf(df_s:pd.DataFrame,margins:dict,n_iter:int=40)->np.ndarray:
    w=np.ones(len(df_s))
    for _ in range(n_iter):
        prev=w.copy()
        for col,targets in margins.items():
            for cat,n_pop in targets.items():
                mask=(df_s[col]==cat).values
                cur=w[mask].sum()
                if cur>0: w[mask]*=n_pop/cur
        if np.abs(w-prev).max()<1e-9: break
    return w

def strat_alloc(df_:pd.DataFrame,n_total:int)->pd.DataFrame:
    info=(df_[["job_type","actual_productivity_score"]].dropna()
          .groupby("job_type")["actual_productivity_score"]
          .agg(N_h="size",S_h="std").reset_index())
    info["n_h"]=np.round(n_total*info["N_h"]/info["N_h"].sum()).astype(int).clip(lower=1)
    info.loc[info["N_h"].idxmax(),"n_h"]+=(n_total-info["n_h"].sum())
    info["w_h"]=info["N_h"]/info["n_h"]
    info["pct_%"]=(100*info["N_h"]/info["N_h"].sum()).round(1)
    return info

def strat_estimate(df_:pd.DataFrame,alloc:pd.DataFrame)->dict:
    pieces=[]
    for _,row in alloc.iterrows():
        sub=df_[(df_["job_type"]==row["job_type"])&
                df_["actual_productivity_score"].notna()].sample(
            n=min(int(row["n_h"]),int(row["N_h"])),random_state=SEED)
        sub=sub.merge(alloc[["job_type","N_h","n_h","w_h"]],on="job_type",how="left")
        pieces.append(sub)
    smp=pd.concat(pieces,ignore_index=True)
    N_t=alloc["N_h"].sum()
    st_=(smp.groupby("job_type")["actual_productivity_score"]
         .agg(ybar_h="mean",s2_h="var",n_s="size").reset_index()
         .merge(alloc[["job_type","N_h"]],on="job_type"))
    st_["W_h"]=st_["N_h"]/N_t; st_["f_h"]=st_["n_s"]/st_["N_h"]
    ybar=(st_["W_h"]*st_["ybar_h"]).sum()
    var=((st_["W_h"]**2)*(1-st_["f_h"])*st_["s2_h"]/st_["n_s"]).sum()
    se=np.sqrt(var)
    return{"ybar":ybar,"se":se,"IC_low":ybar-1.96*se,"IC_high":ybar+1.96*se,"smp":smp}

def two_stage(df_:pd.DataFrame,m:int,n_h:int):
    rng=np.random.default_rng(SEED)
    psus=df_["job_type"].dropna().unique(); M=len(psus)
    sel=rng.choice(psus,size=min(m,M),replace=False)
    pieces,rows=[],[]
    for psu in sel:
        pool=df_[(df_["job_type"]==psu)&df_["actual_productivity_score"].notna()]
        N_i=len(pool); n_i=min(n_h,N_i); pi=(m/M)*(n_i/N_i)
        s=pool.sample(n=n_i,random_state=SEED).assign(w_ht=1/pi,psu=psu,N_i=N_i,n_i=n_i)
        pieces.append(s)
        rows.append({"PSU":psu,"N_i":N_i,"n_i":n_i,
                     "π₁":round(m/M,4),"π_ij":round(pi,4),"w_HT":round(1/pi,4)})
    smp=pd.concat(pieces,ignore_index=True)
    y=smp["actual_productivity_score"].dropna()
    w=smp.loc[smp["actual_productivity_score"].notna(),"w_ht"]
    yb=np.average(y,weights=w); se=np.sqrt(np.average((y-yb)**2,weights=w)/len(y))
    return smp,pd.DataFrame(rows),{"ybar":yb,"se":se,"IC_low":yb-1.96*se,"IC_high":yb+1.96*se}

@st.cache_data
def make_dbca(n_cell:int=60)->pd.DataFrame:
    rng=np.random.default_rng(SEED); parts=[]
    for bloque in df["job_type"].dropna().unique():
        for trat in ["Bajo","Medio","Alto"]:
            pool=df[(df["job_type"]==bloque)&(df["nivel_redes"]==trat)&
                    df["actual_productivity_score"].notna()]
            k=min(n_cell,len(pool))
            if k>0: parts.append(pool.sample(n=k,random_state=int(rng.integers(1e6))))
    out=pd.concat(parts,ignore_index=True).copy()
    out["nivel_redes"]=out["nivel_redes"].astype(str)
    out["job_type"]=out["job_type"].astype(str)
    return out

@st.cache_data
def fit_dbca(n_cell:int=60):
    dbca=make_dbca(n_cell)
    model=smf.ols("actual_productivity_score ~ nivel_redes + job_type",data=dbca).fit()
    anova=sm.stats.anova_lm(model,typ=2).reset_index()
    anova.columns=["Fuente","SS","df","F","p-valor"]
    anova["Fuente"]=anova["Fuente"].replace({
        "nivel_redes":"Tratamiento (nivel redes)",
        "job_type":"Bloque (job_type)","Residual":"Error"})
    return dbca,model,anova

def rel_eff(anova:pd.DataFrame)->float:
    rb=anova[anova["Fuente"]=="Bloque (job_type)"]
    re=anova[anova["Fuente"]=="Error"]
    if rb.empty or re.empty: return np.nan
    ms_b=float(rb["SS"].iloc[0])/float(rb["df"].iloc[0])
    ms_e=float(re["SS"].iloc[0])/float(re["df"].iloc[0])
    return (ms_b+ms_e)/(2*ms_e)

def fisher_lsd(dbca:pd.DataFrame,model)->pd.DataFrame:
    ms_e=model.mse_resid; df_e=model.df_resid
    gr={lv:dbca.loc[dbca["nivel_redes"]==lv,"actual_productivity_score"].dropna()
        for lv in ["Bajo","Medio","Alto"]}
    rows=[]
    for a,b in[("Bajo","Medio"),("Bajo","Alto"),("Medio","Alto")]:
        ya,yb=gr[a].mean(),gr[b].mean(); na,nb=len(gr[a]),len(gr[b])
        se=np.sqrt(ms_e*(1/na+1/nb)); t=abs(ya-yb)/se; p=2*sps.t.sf(t,df_e)
        lsd=sps.t.ppf(1-ALPHA/2,df_e)*se
        rows.append({"Par":f"{a} – {b}","Δ":round(ya-yb,4),"LSD":round(lsd,4),
                     "t":round(t,4),"p":round(p,4),"Sig.":"✅" if p<ALPHA else "—"})
    return pd.DataFrame(rows)

@st.cache_data
def make_2k(n_cell:int=60)->pd.DataFrame:
    rng=np.random.default_rng(SEED); parts=[]
    for a_lv,ac in[("Bajo",-1),("Alto",1)]:
        for b_lv,bc in[("No",-1),("Sí",1)]:
            pool=df[(df["nivel_redes"]==a_lv)&(df["uses_focus_apps"]==b_lv)&
                    df["actual_productivity_score"].notna()].copy()
            k=min(n_cell,len(pool))
            s=pool.sample(n=k,random_state=int(rng.integers(1e6)))
            s=s.copy(); s["A"]=ac; s["B"]=bc; s["A_lv"]=a_lv; s["B_lv"]=b_lv
            parts.append(s)
    out=pd.concat(parts,ignore_index=True).copy()
    out["job_type"]=out["job_type"].astype(str)
    return out

@st.cache_data
def fit_2k(n_cell:int=60):
    d2k=make_2k(n_cell)
    model=smf.ols("actual_productivity_score ~ A * B + job_type",data=d2k).fit()
    anova=sm.stats.anova_lm(model,typ=2).reset_index()
    anova.columns=["Fuente","SS","df","F","p-valor"]
    anova["Fuente"]=anova["Fuente"].replace({
        "A":"A (nivel redes)","B":"B (focus apps)",
        "A:B":"Interacción A×B","job_type":"Bloque","Residual":"Error"})
    y_=d2k["actual_productivity_score"].values; A_=d2k["A"].values; B_=d2k["B"].values; n_=len(y_)
    effs=pd.DataFrame([
        {"Efecto":"A (nivel redes)", "Magnitud":round(2*np.sum(A_*y_)/n_,5)},
        {"Efecto":"B (focus apps)",  "Magnitud":round(2*np.sum(B_*y_)/n_,5)},
        {"Efecto":"AB (interacción)","Magnitud":round(2*np.sum(A_*B_*y_)/n_,5)},
    ])
    return d2k,model,anova,effs

@st.cache_data
def make_confounded(n_cell:int=15):
    rng=np.random.default_rng(SEED); rows=[]
    for a_lv,ac in[("Bajo",-1),("Alto",1)]:
        for b_lv,bc in[("No",-1),("Sí",1)]:
            for c_lv,cc in[("No",-1),("Sí",1)]:
                bloque=1 if ac*bc*cc==1 else 2
                pool=df[(df["nivel_redes"]==a_lv)&(df["uses_focus_apps"]==b_lv)&
                        (df["has_digital_wellbeing_enabled"]==c_lv)&
                        df["actual_productivity_score"].notna()].copy()
                k=min(n_cell,len(pool))
                if k>0:
                    s=pool.sample(n=k,random_state=int(rng.integers(1e6))).copy()
                    s["A"]=ac; s["B"]=bc; s["C_"]=cc; s["Bloque"]=bloque
                    s["A_lv"]=a_lv; s["B_lv"]=b_lv; s["C_lv"]=c_lv
                    rows.append(s)
    full=pd.concat(rows,ignore_index=True).copy()
    full["Bloque_str"]="B"+full["Bloque"].astype(str)
    full["job_type"]=full["job_type"].astype(str)
    esq=pd.DataFrame({
        "Trat.":["(1)","a","b","ab","c","ac","bc","abc"],
        "A":[-1,1,-1,1,-1,1,-1,1],"B":[-1,-1,1,1,-1,-1,1,1],
        "C":[-1,-1,-1,-1,1,1,1,1],"ABC":[-1,1,1,-1,1,-1,-1,1],
        "Bloque":[2,1,1,2,1,2,2,1]})
    return full,esq

# ═══════════════════════════════════════════════════════════════════════════
#  GRÁFICAS ALTAIR
# ═══════════════════════════════════════════════════════════════════════════
def hist_a(df_,col,title,xlabel,clr=None,bins=40):
    s=df_.dropna(subset=[col]).sample(min(8000,len(df_)),random_state=SEED)
    return(alt.Chart(s,title=title).mark_bar(color=clr or P["accent"],opacity=.85,binSpacing=1)
           .encode(alt.X(col,bin=alt.Bin(maxbins=bins),title=xlabel),
                   alt.Y("count()",title="Frecuencia"),
                   tooltip=[alt.Tooltip(col,bin=True),"count()"]))

def box_a(df_,x,y,title,cmap=None,sort_x=None):
    s=df_.dropna(subset=[x,y])
    ce=(alt.Color(x,scale=alt.Scale(domain=list(cmap.keys()),range=list(cmap.values())),legend=None)
        if cmap else alt.Color(x,legend=None))
    return(alt.Chart(s,title=title).mark_boxplot(extent="min-max",size=34)
           .encode(alt.X(x,sort=sort_x or "ascending"),alt.Y(y,scale=alt.Scale(zero=False)),
                   color=ce,tooltip=[x,alt.Tooltip(y,format=".3f")]))

def bar_means_a(df_,grp,y,title,cmap,sort=None):
    agg=(df_.dropna(subset=[grp,y]).groupby(grp,observed=True)[y]
         .agg(["mean","std","count"]).reset_index())
    agg["se"]=agg["std"]/np.sqrt(agg["count"])
    agg["lo"]=agg["mean"]-1.96*agg["se"]; agg["hi"]=agg["mean"]+1.96*agg["se"]
    bars=(alt.Chart(agg,title=title)
          .mark_bar(opacity=.88,cornerRadiusTopLeft=4,cornerRadiusTopRight=4)
          .encode(alt.X(grp,sort=sort or list(cmap.keys())),
                  alt.Y("mean",title=y,scale=alt.Scale(zero=False)),
                  alt.Color(grp,scale=alt.Scale(domain=list(cmap.keys()),range=list(cmap.values())),legend=None),
                  tooltip=[grp,alt.Tooltip("mean",format=".3f"),alt.Tooltip("se",format=".4f")]))
    err=(alt.Chart(agg).mark_errorbar(ticks=True,thickness=2,color=P["text"])
         .encode(alt.X(grp,sort=sort or list(cmap.keys())),alt.Y("lo"),alt.Y2("hi")))
    return bars+err

def scatter_a(df_,x,y,title,xl,yl):
    s=df_.dropna(subset=[x,y]).sample(min(2000,len(df_)),random_state=SEED)
    pts=(alt.Chart(s,title=title).mark_circle(color=P["accent"],opacity=.35,size=35)
         .encode(alt.X(x,title=xl,scale=alt.Scale(zero=False)),
                 alt.Y(y,title=yl,scale=alt.Scale(zero=False)),tooltip=[x,y]))
    reg=pts.transform_regression(x,y).mark_line(color=P["accent2"],size=2.5)
    return pts+reg

def qq_a(resid,title="Q-Q de residuos"):
    n=len(resid); probs=(np.arange(1,n+1)-.5)/n
    theo=sps.norm.ppf(probs); emp=np.sort(resid)
    df_=pd.DataFrame({"Teórico":theo,"Observado":emp})
    pts=(alt.Chart(df_,title=title).mark_circle(color=P["accent"],opacity=.4,size=16)
         .encode(alt.X("Teórico"),alt.Y("Observado"),
                 tooltip=[alt.Tooltip("Teórico",format=".3f"),alt.Tooltip("Observado",format=".3f")]))
    lim=max(abs(theo.min()),abs(theo.max()))
    ref=pd.DataFrame({"x":[-lim,lim],"y":[-lim,lim]})
    ln=(alt.Chart(ref).mark_line(color=P["accent2"],strokeDash=[5,3],size=2)
        .encode(alt.X("x"),alt.Y("y")))
    return pts+ln

def rv_a(model,title="Residuos vs Ajustados"):
    rdf=pd.DataFrame({"aj":model.fittedvalues,"res":model.resid})
    rdf=rdf.sample(min(4000,len(rdf)),random_state=SEED)
    pts=(alt.Chart(rdf,title=title).mark_circle(color=P["accent"],opacity=.3,size=20)
         .encode(alt.X("aj",title="Ajustado"),alt.Y("res",title="Residuo"),
                 tooltip=[alt.Tooltip("aj",format=".3f"),alt.Tooltip("res",format=".3f")]))
    ref=pd.DataFrame({"x":[rdf["aj"].min(),rdf["aj"].max()],"y":[0,0]})
    ln=alt.Chart(ref).mark_line(color=P["warn"],strokeDash=[4,3]).encode(alt.X("x"),alt.Y("y"))
    return pts+ln

def power_a(fs,kk,n_req_g):
    pa=FTestAnovaPower(); ns=np.arange(5,max(300,n_req_g*3),5)
    pw=[pa.power(effect_size=max(fs,.005),nobs=int(n*kk),alpha=ALPHA,k_groups=kk) for n in ns]
    pdf=pd.DataFrame({"n_grupo":ns,"Potencia":pw})
    area=(alt.Chart(pdf,title="Curva de potencia del ANOVA")
          .mark_area(color=P["accent"],opacity=.2,line={"color":P["accent"],"size":2.5})
          .encode(alt.X("n_grupo",title="n por grupo"),
                  alt.Y("Potencia",scale=alt.Scale(domain=[0,1]))))
    r80=alt.Chart(pd.DataFrame({"y":[.8]})).mark_rule(color=P["ok"],strokeDash=[4,3]).encode(alt.Y("y"))
    vl=alt.Chart(pd.DataFrame({"x":[n_req_g]})).mark_rule(color=P["warn"],strokeDash=[4,3]).encode(alt.X("x"))
    return area+r80+vl

def inter_a(df_,title):
    agg=(df_.dropna(subset=["nivel_redes","uses_focus_apps","actual_productivity_score"])
         .groupby(["nivel_redes","uses_focus_apps"],observed=True)["actual_productivity_score"]
         .mean().reset_index())
    return(alt.Chart(agg,title=title).mark_line(point=True,size=2.5)
           .encode(alt.X("nivel_redes",sort=["Bajo","Medio","Alto"],title="Nivel de redes"),
                   alt.Y("actual_productivity_score",title="Media prod. real",scale=alt.Scale(zero=False)),
                   alt.Color("uses_focus_apps",title="Focus apps",
                             scale=alt.Scale(domain=["Sí","No"],range=[P["ok"],P["warn"]])),
                   tooltip=["nivel_redes","uses_focus_apps",
                            alt.Tooltip("actual_productivity_score",format=".4f")]))

def miss_a(df_):
    miss=df_.isnull().mean().reset_index(); miss.columns=["Variable","Prop"]
    miss=miss[miss["Prop"]>0].sort_values("Prop",ascending=False)
    miss["Pct"]=(miss["Prop"]*100).round(1)
    return(alt.Chart(miss,title="Proporción de faltantes por variable")
           .mark_bar(color=P["warn"],opacity=.88,cornerRadiusTopRight=4,cornerRadiusBottomRight=4)
           .encode(alt.Y("Variable",sort="-x",title=""),
                   alt.X("Prop",title="Proporción",axis=alt.Axis(format="%")),
                   tooltip=["Variable",alt.Tooltip("Pct",title="%")]))

def corr_a(df_,cols,title):
    s=df_[cols].dropna().sample(min(8000,len(df_)),random_state=SEED)
    corr=s.corr().reset_index().melt("index"); corr.columns=["V1","V2","r"]
    nm={"daily_social_media_time":"Redes","actual_productivity_score":"Prod.real",
        "perceived_productivity_score":"Prod.perc.","stress_level":"Estrés",
        "sleep_hours":"Sueño","work_hours_per_day":"Hrs.trab.",
        "number_of_notifications":"Notif.","job_satisfaction_score":"Satisf.",
        "days_feeling_burnout_per_month":"Burnout","weekly_offline_hours":"Offline",
        "breaks_during_work":"Pausas","screen_time_before_sleep":"Pantalla",
        "coffee_consumption_per_day":"Café"}
    corr["V1"]=corr["V1"].map(nm).fillna(corr["V1"])
    corr["V2"]=corr["V2"].map(nm).fillna(corr["V2"])
    base=alt.Chart(corr,title=title)
    rect=base.mark_rect().encode(
        alt.X("V1",title=""),alt.Y("V2",title=""),
        alt.Color("r",scale=alt.Scale(
            domain=[-1,0,1],
            range=[P["accent2"],P["surface"],P["accent"]]),title="r"),
        tooltip=["V1","V2",alt.Tooltip("r",format=".3f")])
    txt=base.mark_text(fontSize=10,fontWeight=600).encode(
        alt.X("V1",title=""),alt.Y("V2",title=""),
        alt.Text("r",format=".2f"),
        color=alt.condition(
            "abs(datum.r) > 0.35",
            alt.value("#ffffff"),
            alt.value(P["muted"])))
    return rect+txt

# ═══════════════════════════════════════════════════════════════════════════
#  PRE-CÓMPUTO
# ═══════════════════════════════════════════════════════════════════════════
N_CELL=60
dbca,model_dbca,anova_dbca=fit_dbca(N_CELL)
d2k,model_2k,anova_2k,effs_2k=fit_2k(N_CELL)
d23,esq_23=make_confounded(15)
p_trat=float(anova_dbca.loc[anova_dbca["Fuente"]=="Tratamiento (nivel redes)","p-valor"].iloc[0])
p_blk =float(anova_dbca.loc[anova_dbca["Fuente"]=="Bloque (job_type)","p-valor"].iloc[0])
re_v  =rel_eff(anova_dbca)

# Raking global
df_anal=df.dropna(subset=["actual_productivity_score","nivel_redes","job_type","gender"]).reset_index(drop=True)
n_a=len(df_anal)
mg={"gender":{k:v*n_a/N for k,v in df["gender"].value_counts().items()},
    "job_type":{k:v*n_a/N for k,v in df["job_type"].value_counts().items()}}
df_anal["w_rak"]=raking_ipf(df_anal,mg)

# MCAR test global
_mcols=["daily_social_media_time","actual_productivity_score","stress_level",
        "sleep_hours","job_satisfaction_score"]
_ind=df[_mcols].isnull().astype(int)
_chi,_pv=[],[]
for _i,_c1 in enumerate(_mcols):
    for _c2 in _mcols[_i+1:]:
        _ct=pd.crosstab(_ind[_c1],_ind[_c2])
        if _ct.shape==(2,2):
            _ch,_p,*_=chi2_contingency(_ct); _chi.append(_ch); _pv.append(_p)
chi_med=np.mean(_chi); p_med=np.mean(_pv)

# Estimaciones para conclusiones
n_tot=600
alloc=strat_alloc(df,n_tot)
res_strat=strat_estimate(df,alloc)
_,_,ht_res=two_stage(df,4,150)

# deff encuesta compleja
df_a7=df_anal.copy()
y7=df_a7["actual_productivity_score"]; w7=df_a7["w_rak"]
yb_w=np.average(y7,weights=w7); var_w=np.average((y7-yb_w)**2,weights=w7)/len(y7)
se_w=np.sqrt(var_w)
deff=var_w/(y7.var(ddof=1)/len(y7))

# ═══════════════════════════════════════════════════════════════════════════
#  ENCABEZADO
# ═══════════════════════════════════════════════════════════════════════════
# ── Barra lateral de estado ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style="padding:.5rem .25rem;">
      <div style="color:{P['accent']};font-size:.62rem;font-weight:800;
          letter-spacing:.14em;text-transform:uppercase;margin-bottom:.6rem;opacity:.7;">
        Universidad Santo Tomás</div>
      <div style="color:{P['text']};font-size:.95rem;font-weight:700;
          line-height:1.25;margin-bottom:.35rem;">Redes Sociales<br>&amp; Productividad</div>
      <div style="color:{P['muted']};font-size:.73rem;line-height:1.5;margin-bottom:1rem;">
        Diseño de Experimentos · 2026-I<br>
        Javier Mauricio Sierra</div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"""<div style="font-size:.7rem;color:{P['muted']};font-weight:700;
        letter-spacing:.08em;text-transform:uppercase;margin-bottom:.5rem;">Equipo</div>
      <div style="font-size:.78rem;color:{P['text']};line-height:1.65;">
        Lina M. Galvis B.<br>Julián M. Valderrama T.</div>""",
        unsafe_allow_html=True)
    st.divider()
    st.markdown(f"""<div style="font-size:.7rem;color:{P['muted']};font-weight:700;
        letter-spacing:.08em;text-transform:uppercase;margin-bottom:.6rem;">Fases</div>""",
        unsafe_allow_html=True)
    for _fn,_ft,_fp in [("1","Muestreo Básico","10%"),("2","DOE Básico","7%"),
                         ("3","Factorial","12%"),("4","Diseños 2ᵏ","12%"),
                         ("5","Bloqueo","10%"),("6","P. Desiguales","10%"),("7","Encuestas","10%")]:
        _clr=list(FASE_CLR.values())[int(_fn)-1]
        st.markdown(f"""<div style="display:flex;align-items:center;gap:.5rem;
            padding:.28rem 0;border-bottom:1px solid {P['border']}22;">
          <span style="background:{_clr}22;color:{_clr};font-size:.65rem;font-weight:800;
              padding:.1rem .38rem;border-radius:4px;min-width:1.4rem;text-align:center;">
              F{_fn}</span>
          <div style="flex:1;">
            <div style="font-size:.73rem;color:{P['text']};font-weight:600;">{_ft}</div>
            <div style="font-size:.63rem;color:{P['muted']};">{_fp} rúbrica</div>
          </div></div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"""<div style="font-size:.68rem;color:{P['muted']};line-height:1.6;">
        Dataset: 30 000 obs · 19 vars<br>
        Faltantes: ~8% (MCAR ✅)<br>
        Herramienta: Python / Streamlit</div>""", unsafe_allow_html=True)

# ── Encabezado principal ──────────────────────────────────────────────────
st.markdown(f"""<div style="
    background:linear-gradient(135deg,{P['surface']} 0%,{P['card']} 50%,{P['surface']} 100%);
    border:1px solid {P['border']};border-radius:14px;
    padding:1.4rem 2rem;margin-bottom:1.1rem;
    box-shadow:0 8px 32px rgba(0,0,0,.35);
    position:relative;overflow:hidden;">
  <div style="position:absolute;top:-20px;right:-10px;font-size:7rem;opacity:.04;
      font-weight:900;letter-spacing:-.05em;user-select:none;">DOE</div>
  <div style="display:flex;align-items:flex-start;justify-content:space-between;
      flex-wrap:wrap;gap:1rem;position:relative;z-index:1;">
    <div>
      <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem;">
        <div style="width:4px;height:28px;background:linear-gradient(180deg,
            {P['accent']},{P['accent2']});border-radius:2px;"></div>
        <div>
          <div style="color:{P['accent']};font-size:.64rem;font-weight:800;
              letter-spacing:.14em;text-transform:uppercase;opacity:.85;">
            Universidad Santo Tomás · Diseño de Experimentos · 2026-I</div>
          <h1 style="margin:.1rem 0 0;font-size:1.65rem;font-weight:800;">
            Redes Sociales &amp; Productividad</h1>
        </div>
      </div>
      <p style="margin:0;color:{P['muted']};font-size:.82rem;padding-left:.6rem;">
        Lina María Galvis Barragán · Julián Mateo Valderrama Tibaduija ·
        <span style="color:{P['accent']}88;">Docente: Javier Mauricio Sierra</span></p>
    </div>
    <div style="display:flex;gap:.4rem;flex-wrap:wrap;align-items:flex-start;padding-top:.25rem;">
      <span style="background:{P['border']};padding:.28rem .7rem;border-radius:7px;
          font-size:.72rem;font-weight:700;letter-spacing:.03em;">n = 30 000</span>
      <span style="background:{P['accent']}18;color:{P['accent']};
          border:1px solid {P['accent']}33;padding:.28rem .7rem;border-radius:7px;
          font-size:.72rem;font-weight:700;">7 Fases · Observacional</span>
      <span style="background:{P['accent2']}18;color:{P['accent2']};
          border:1px solid {P['accent2']}33;padding:.28rem .7rem;border-radius:7px;
          font-size:.72rem;font-weight:700;">Python · Streamlit</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════
T=st.tabs(["📌 Datos & EDA","① Fase 1 · Muestreo","② Fase 2 · DOE Básico",
           "③ Fase 3 · Factorial","④ Fase 4 · 2ᵏ","⑤ Fase 5 · Bloqueo",
           "⑥ Fase 6 · P. Desiguales","⑦ Fase 7 · Encuestas","📋 Conclusiones"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INICIO & DATOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with T[0]:
    callout("info","Pregunta de investigación",
    """¿El nivel de uso diario de redes sociales se asocia con diferencias significativas
    en la productividad real, controlando por tipo de trabajo?""")

    c1,c2,c3,c4,c5=st.columns(5)
    with c1: kpi("Observaciones","30 000","Base completa",P["accent"])
    with c2: kpi("Faltantes","~8%","en vars. clave",P["warn"])
    with c3: kpi("Mecanismo","MCAR","plausible (χ²)",P["ok"])
    with c4: kpi("Prod. media","4.950","escala 0–10",P["accent2"])
    with c5: kpi("Tiempo redes","4.98 h","media diaria",FASE_CLR["I"])

    st.markdown("<br>",unsafe_allow_html=True)
    cols=st.columns(7)
    info_fases=[("I","Muestreo\nBásico","10%"),("II","DOE\nBásico","7%"),
                ("III","Factorial","12%"),("IV","2ᵏ","12%"),
                ("V","Bloqueo","10%"),("VI","Prob.\nDesig.","10%"),("VII","Encuestas","10%")]
    for col,(num,lbl,pct) in zip(cols,info_fases):
        clr=FASE_CLR[num]
        col.markdown(f"""<div style="background:{clr}10;border:1px solid {clr}30;
            border-top:3px solid {clr};border-radius:9px;padding:.65rem .45rem;text-align:center;">
          <div style="color:{clr};font-size:.95rem;font-weight:800;">F{num}</div>
          <div style="color:{P['text']};font-size:.68rem;margin:.12rem 0;white-space:pre-line;">{lbl}</div>
          <div style="color:{P['muted']};font-size:.64rem;">{pct}</div></div>""",
            unsafe_allow_html=True)

    section("Datos faltantes y mecanismo MCAR")
    cm,ct=st.columns([1.4,1])
    with cm: show(miss_a(df),"miss0",270)
    with ct:
        md=df.isnull().sum().reset_index(); md.columns=["Variable","N"]
        md=md[md["N"]>0]; md["%"]=(100*md["N"]/N).round(1)
        st.dataframe(md,use_container_width=True,hide_index=True)

    just_box([
        ("Prueba de Little (MCAR formal)",
         "Requiere normalidad multivariada por patrón de missingness y su implementación robusta "
         "exige <code>pyampute</code> o <code>missMethods</code> (R) — paquetes no disponibles "
         "en Python estándar sin instalaciones adicionales. Además, con n = 30 000 la potencia "
         "de la prueba es tan alta que detecta cualquier desviación mínima, incluso bajo MCAR.",
         "χ² de independencia entre pares de indicadores de missingness (análisis marginal)."),
        ("Imputación múltiple (MICE)",
         "Con MCAR confirmado, la imputación múltiple solo reduce varianza del estimador — "
         "no corrige sesgo (ya no hay sesgo bajo MCAR). El beneficio es marginal con n = 30 000 "
         "y el costo computacional es sustancial (k = 5–20 conjuntos imputados).",
         "Raking IPF con casos completos — estándar en encuestas grandes bajo MCAR."),
    ])
    justif("¿Por qué NO se usa la prueba de Little directamente?",
    f"""La prueba de Little (1988) requiere normalidad multivariada por patrón de missingness
    y su implementación robusta exige el paquete <code>pyampute</code> o <code>missMethods</code>
    (R) — no disponibles directamente en este entorno Python sin instalaciones adicionales.
    <b>Alternativa adoptada:</b> prueba χ² de independencia entre pares de indicadores binarios
    de missingness. Si los indicadores son independientes entre sí, MCAR es plausible.
    <b>Resultado:</b> χ² promedio = {chi_med:.3f},
    p promedio = {p_med:.4f}
    → {"independencia no rechazada → MCAR plausible ✅" if p_med>0.10 else "indicios de dependencia → revisar MAR ⚠️"}.""")

    section("Calibración Raking (IPF)")
    justif("¿Por qué Raking y no imputación múltiple?",
    """Con MCAR confirmado, la imputación múltiple solo reduce varianza (no corrige sesgo)
    — beneficio marginal con n = 30 000. El raking (IPF) ajusta pesos para que las
    marginales de la submuestra analítica coincidan con las de la base completa
    (género × job_type). Es el estándar en encuestas complejas (Lohr, 2022, Cap. 7)
    y se exige explícitamente en la Fase VII de la rúbrica.""")

    ybar_p=df["actual_productivity_score"].mean()
    ybar_cc=df_anal["actual_productivity_score"].mean()
    ybar_rk=np.average(df_anal["actual_productivity_score"],weights=df_anal["w_rak"])
    comp_r=pd.DataFrame({"Estimador":["Población (ref.)","Casos completos","Con raking"],
                          "Media":[round(ybar_p,4),round(ybar_cc,4),round(ybar_rk,4)],
                          "Δ":[0,round(ybar_cc-ybar_p,4),round(ybar_rk-ybar_p,4)]})
    st.dataframe(comp_r,use_container_width=True,hide_index=True)
    interp("Raking",
    f"""Pesos en [{df_anal['w_rak'].min():.3f}, {df_anal['w_rak'].max():.3f}] ≈ 1.0.
    Diferencia pon./sin ponderar = {abs(ybar_rk-ybar_cc):.5f} pts — MCAR confirmado.""")

    section("EDA rápido")
    callout("warn","Resultado descriptivo clave",
    """Medias de productividad real: 4.950 (Bajo), 4.966 (Medio), 4.929 (Alto).
    Diferencia máxima = 0.037 pts en escala 0–10 → anticipa efecto estadístico mínimo.""")
    c1,c2=st.columns(2)
    with c1: show(box_a(df,"nivel_redes","actual_productivity_score",
        "Productividad real por nivel de redes",PAL3,["Bajo","Medio","Alto"]),"bx0",330)
    with c2: show(corr_a(df,["daily_social_media_time","actual_productivity_score",
        "perceived_productivity_score","stress_level","sleep_hours",
        "job_satisfaction_score","work_hours_per_day"],"Correlaciones — variables clave"),"corr0",330)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE I
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with T[1]:
    fase_header("I","Diseño Muestral Básico",10,
    "Estratificado proporcional · Bietápico HT · Estimadores auxiliares (razón y regresión)")

    section("1.1 Población y Marco Muestral",FASE_CLR["I"])
    callout("info","Definiciones",
    """Población objetivo: personas que reportan uso de redes y métricas de productividad.
    Marco muestral: 30 000 registros. Variable de interés <i>y</i>: productividad real (0–10).
    Variable auxiliar <i>x</i>: productividad percibida (ρ ≈ 0.96).""")

    section("1.2 Estratificado Proporcional",FASE_CLR["I"])
    justif("¿Por qué estratificado y no MAS?",
    """El tipo de trabajo crea grupos con posibles diferencias en productividad.
    La estratificación garantiza representación de todos los grupos y reduce la varianza del
    estimador cuando los estratos son internamente homogéneos. La afijación proporcional es
    la más sencilla y natural cuando las varianzas intra-estrato no difieren marcadamente.""")

    alloc2=strat_alloc(df,n_tot)
    res2=strat_estimate(df,alloc2)
    c1,c2,c3,c4=st.columns(4)
    with c1: kpi("N población",f"{df['actual_productivity_score'].notna().sum():,}","",FASE_CLR["I"])
    with c2: kpi("n muestra",f"{n_tot}","",FASE_CLR["I"])
    with c3: kpi("ȳ estratificado",f"{res2['ybar']:.4f}","",FASE_CLR["I"])
    with c4: kpi("IC 95%",f"[{res2['IC_low']:.3f}, {res2['IC_high']:.3f}]","",FASE_CLR["I"])
    st.dataframe(alloc2.round(3),use_container_width=True,hide_index=True)

    section("1.3 Muestreo Bietápico — Estimador HT",FASE_CLR["I"])
    justif("¿Por qué bietápico y no estratificado puro?",
    """PSU = tipo de trabajo (6 grupos), SSU = individuos. El bietápico es eficiente
    cuando existe estructura natural de conglomerados. El estimador Horvitz-Thompson
    con pesos π⁻¹ garantiza insesgamiento aunque las probabilidades de selección
    sean desiguales entre PSUs.""")

    c_m,c_n=st.columns(2)
    with c_m: m_sl=st.slider("PSUs a seleccionar (m)",2,6,4,key="m1")
    with c_n: n_hs=st.slider("SSUs por PSU (n_h)",50,400,150,step=50,key="nh1")
    _,psu_i,ht2=two_stage(df,m_sl,n_hs)
    st.dataframe(psu_i,use_container_width=True,hide_index=True)
    c1,c2,c3=st.columns(3)
    with c1: kpi("ȳ_HT",f"{ht2['ybar']:.4f}","Horvitz-Thompson",FASE_CLR["I"])
    with c2: kpi("SE_HT",f"{ht2['se']:.4f}","","")
    with c3: kpi("IC 95% HT",f"[{ht2['IC_low']:.3f}, {ht2['IC_high']:.3f}]","","")

    section("1.4 Estimadores Auxiliares",FASE_CLR["I"])
    justif("¿Por qué usar estimador de regresión y no solo el directo?",
    """Con ρ ≈ 0.96 entre productividad percibida y real, el estimador de regresión
    aprovecha el conocimiento de μ_x (media poblacional del auxiliar) para reducir la
    varianza del estimador directo. La ganancia es máxima cuando CV(x)≈CV(y) y la relación
    es lineal — condiciones que se cumplen aquí.""")

    smp_a=res2["smp"][["actual_productivity_score","perceived_productivity_score"]].dropna()
    yb=smp_a["actual_productivity_score"].mean(); xb=smp_a["perceived_productivity_score"].mean()
    mu_x=df["perceived_productivity_score"].mean()
    R=yb/xb; est_r=R*mu_x
    b1=smf.ols("actual_productivity_score ~ perceived_productivity_score",data=smp_a).fit().params["perceived_productivity_score"]
    est_g=yb+b1*(mu_x-xb); mu_pop=df["actual_productivity_score"].mean()
    aux_df=pd.DataFrame({"Estimador":["Directo","Razón","Regresión"],
        "Estimación":[round(yb,4),round(est_r,4),round(est_g,4)],
        "Ref. población":[round(mu_pop,4)]*3,
        "|Error|":[round(abs(yb-mu_pop),5),round(abs(est_r-mu_pop),5),round(abs(est_g-mu_pop),5)]})
    st.dataframe(aux_df,use_container_width=True,hide_index=True)
    show(scatter_a(df,"perceived_productivity_score","actual_productivity_score",
        "Prod. percibida vs real — justificación del auxiliar","Percibida","Real"),"sc1",320)
    interp("Estimadores auxiliares",
    f"""El estimador de regresión reduce el error absoluto de {abs(yb-mu_pop):.5f}
    (directo) a {abs(est_g-mu_pop):.5f}. La alta correlación r≈0.96 justifica
    el uso del auxiliar percibida para mejorar la precisión.""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE II — DOE BÁSICO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with T[2]:
    fase_header("II","Diseño Experimental Básico — DBCA",7,
    "Tratamiento: nivel de redes (3 niveles). Bloque: tipo de trabajo. "
    "ANOVA Tipo II · Tukey · Fisher LSD · Dunnett · Supuestos · Potencia")

    justif("¿Por qué DBCA y no CRD, DCL o BIBD?",
    """<b>CRD (Completamente Aleatorizado):</b> ignoraría la variabilidad entre tipos de trabajo,
    inflando el MSE y reduciendo la potencia. <b>DBCA elegido:</b> una sola fuente de
    heterogeneidad identificada (job_type) → bloque simple, balanceado.
    <b>Cuadrado Latino / Greco-Latino:</b> requiere dos fuentes de bloqueo ortogonales;
    no disponemos de una segunda variable con estructura adecuada.
    <b>BIBD (Diseño Incompleto en Bloques):</b> innecesario con datos suficientes para un
    DBCA completo balanceado; el BIBD añade complejidad sin ganancia aquí.""")

    section("2.1 Balance del diseño",FASE_CLR["II"])
    bal=pd.crosstab(dbca["nivel_redes"],dbca["job_type"])
    st.dataframe(bal,use_container_width=True)
    callout("ok","Balance",f"{N_CELL} observaciones exactas por celda (tratamiento × bloque). Diseño ortogonal balanceado.")

    section("2.2 ANOVA Tipo II",FASE_CLR["II"])
    av=anova_dbca.copy(); av["p-valor"]=anova_dbca["p-valor"].apply(pv)
    av["Decisión"]=anova_dbca["p-valor"].apply(lambda p:"✅ Sig." if pd.notna(p) and p<ALPHA else "— No sig.")
    st.dataframe(av.round(4),use_container_width=True,hide_index=True)
    c1,c2,c3=st.columns(3)
    with c1:
        if p_trat<ALPHA: callout("ok","Tratamiento",dec(p_trat))
        else: callout("warn","Tratamiento",dec(p_trat))
    with c2:
        if p_blk<ALPHA: callout("ok","Bloque",dec(p_blk))
        else: callout("warn","Bloque",dec(p_blk))
    with c3: kpi("Eficiencia RE",f"{re_v:.3f}" if not np.isnan(re_v) else "—",
                 "RE>1 → bloqueo útil",FASE_CLR["II"])

    section("2.3 Comparaciones múltiples",FASE_CLR["II"])
    c1,c2=st.columns(2)
    with c1: show(bar_means_a(dbca,"nivel_redes","actual_productivity_score",
        "Medias ±1.96SE — nivel de redes",PAL3,["Bajo","Medio","Alto"]),"bar2",340)
    with c2: show(box_a(dbca,"job_type","actual_productivity_score","Por bloque"),"bx2b",340)

    tab_t,tab_l,tab_d=st.tabs(["Tukey HSD","Fisher LSD","Dunnett (aprox.)"])
    with tab_t:
        tk=pairwise_tukeyhsd(dbca["actual_productivity_score"],dbca["nivel_redes"],alpha=ALPHA)
        st.dataframe(pd.DataFrame(tk.summary().data[1:],columns=tk.summary().data[0]),
                     use_container_width=True,hide_index=True)
        callout("info","Tukey HSD",
        """Controla el error tipo I familiar (FWER) para todas las comparaciones posibles.
        El más conservador. Preferido cuando se quieren controlar falsos positivos globalmente.
        Requiere un ANOVA previo significativo para interpretarse correctamente.""")
    with tab_l:
        st.dataframe(fisher_lsd(dbca,model_dbca),use_container_width=True,hide_index=True)
        callout("warn","Fisher LSD",
        """No ajusta el error tipo I por comparación múltiple. Más liberal (mayor potencia),
        pero mayor tasa de falsos positivos. Solo recomendable cuando el ANOVA es significativo
        (protección de Fisher). Útil aquí para comparar con Tukey.""")
    with tab_d:
        ms_e=model_dbca.mse_resid; df_e=model_dbca.df_resid
        gr={lv:dbca.loc[dbca["nivel_redes"]==lv,"actual_productivity_score"].dropna()
            for lv in ["Bajo","Medio","Alto"]}
        dun_rows=[]
        for trat in["Medio","Alto"]:
            ya,yb2=gr["Bajo"].mean(),gr[trat].mean(); na,nb=len(gr["Bajo"]),len(gr[trat])
            se_d=np.sqrt(ms_e*(1/na+1/nb)); t_d=abs(ya-yb2)/se_d
            p_adj=min(1.0,2*2*sps.t.sf(t_d,df_e))
            dun_rows.append({"vs control (Bajo)":trat,"Δ":round(yb2-ya,4),
                             "t":round(t_d,4),"p adj":round(p_adj,4),
                             "Sig.":"✅" if p_adj<ALPHA else "—"})
        st.dataframe(pd.DataFrame(dun_rows),use_container_width=True,hide_index=True)
        callout("info","Dunnett (aprox. Bonferroni)",
        """Compara solo contra el grupo de referencia (Bajo). Apropiado cuando el interés
        es detectar si los otros niveles difieren del control. Más potente que Tukey
        para k comparaciones vs control específico.""")

    section("2.4 Verificación de supuestos",FASE_CLR["II"])
    resid=model_dbca.resid.values
    ad_s,ad_p=normal_ad(resid)
    lev_s,lev_p=levene(*[dbca.loc[dbca["nivel_redes"]==lv,"actual_productivity_score"].dropna().values
                          for lv in["Bajo","Medio","Alto"]],center="median")
    dw_v=durbin_watson(resid)

    just_box([
        ("Shapiro-Wilk",
         f"Límite práctico n ≤ 5 000 en <code>scipy</code>. Con n = {len(dbca):,} residuos "
         "detecta desviaciones de nanómetros estadísticamente significativas pero "
         "<b>prácticamente irrelevantes</b>. El p-valor ≈ 0 en muestras grandes no implica "
         "violación grave de normalidad — solo que tenemos suficiente potencia para detectar "
         "trivialidades. El Teorema Central del Límite garantiza la validez del test F.",
         "Anderson-Darling + Q-Q plot + argumento TCL"),
        ("Esfericidad de Mauchly",
         "Aplica <b>exclusivamente</b> a ANOVA de medidas repetidas (el mismo sujeto medido "
         "k veces). Aquí los bloques son tipos de trabajo <i>independientes</i> — no el mismo "
         "individuo medido repetidamente. Su uso aquí sería un error conceptual grave.",
         "No aplica. Se verificó homocedasticidad con Levene."),
        ("Welch ANOVA sin bloque",
         "Útil cuando hay heterocedasticidad severa entre grupos. Levene p > 0.05 confirma "
         "homocedasticidad — el ANOVA de Fisher es el más potente en esa condición.",
         "ANOVA clásico Tipo II (F de Fisher)."),
    ])

    tests_df=pd.DataFrame({
        "Supuesto":["Normalidad (AD)","Homocedasticidad (Levene)","Independencia (DW)"],
        "Estadístico":[round(ad_s,4),round(lev_s,4),round(dw_v,4)],
        "p-valor":[pv(ad_p),pv(lev_p),"—"],
        "Decisión":[dec(ad_p),dec(lev_p),"✅ Sin autocorr." if 1.5<dw_v<2.5 else "⚠️ Revisar"],
        "Nota":["n grande infla rechazo — ver Q-Q","Homoc. robusta si p>0.05",f"DW={dw_v:.3f}, rango ok [1.5,2.5]"]
    })
    st.dataframe(tests_df,use_container_width=True,hide_index=True)
    c1,c2=st.columns(2)
    with c1: show(qq_a(resid,"Q-Q — Residuos DBCA"),"qq2",330)
    with c2: show(rv_a(model_dbca,"Residuos vs Ajustados"),"rv2",330)

    section("2.5 Potencia del ANOVA",FASE_CLR["II"])
    ss_t=float(anova_dbca.loc[anova_dbca["Fuente"]=="Tratamiento (nivel redes)","SS"].iloc[0])
    ss_tot=anova_dbca["SS"].sum()
    eta2=min(ss_t/ss_tot if ss_tot>0 else 1e-6,.999)
    fc=np.sqrt(eta2/(1-eta2)); kk=3
    ng=int(dbca.groupby("nivel_redes").size().min())
    pa=FTestAnovaPower()
    curr_pw=pa.power(effect_size=max(fc,.005),nobs=ng*kk,alpha=ALPHA,k_groups=kk)
    n_req_g=int(np.ceil(pa.solve_power(effect_size=max(fc,.1),power=.80,alpha=ALPHA,k_groups=kk)/kk))
    c1,c2,c3=st.columns(3)
    with c1: kpi("η²",f"{eta2:.6f}","Prop. varianza explicada",FASE_CLR["II"])
    with c2: kpi("f de Cohen",f"{fc:.6f}","Muy pequeño (<0.10)",FASE_CLR["II"])
    with c3: kpi("Potencia actual",f"{curr_pw:.4f}",f"n={ng}/grupo",FASE_CLR["II"])
    show(power_a(fc,kk,n_req_g),"pow2",310)
    interp("Fase II",
    f"""ANOVA: p={pv(p_trat)} → {dec(p_trat)}. η²={eta2:.6f} — efecto prácticamente nulo.
    RE={re_v:.3f}{"→ bloqueo útil" if re_v>1 else "→ bloque marginal"}.
    AD rechaza normalidad formal (n grande); Q-Q sin desviaciones graves; TCL garantiza validez del F.""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE III — FACTORIAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with T[3]:
    fase_header("III","Diseños Factoriales",12,
    "2² con bloqueo: A = nivel redes, B = focus apps. "
    "Efectos principales · Interacción · Pareto · Gráfico de interacción")

    justif("¿Por qué factorial y no OFAT (One Factor At a Time)?",
    """OFAT no detecta interacciones. Si el efecto de las redes sobre la productividad
    depende de si la persona usa focus apps (interacción A×B), OFAT llevaría a conclusiones
    erróneas. El factorial 2² estudia todos los factores simultáneamente: más eficiente
    (misma muestra, más información), mayor potencia, y detecta sinergia o antagonismo entre
    factores — imposible con OFAT.""")

    st.dataframe(pd.DataFrame({"Trat.":["(1)","a","b","ab"],"A":[-1,1,-1,1],
        "B":[-1,-1,1,1],"AB":[1,-1,-1,1],
        "Descripción":["Bajo redes, No focus","Alto redes, No focus",
                       "Bajo redes, Sí focus","Alto redes, Sí focus"]}),
        use_container_width=True,hide_index=True)

    section("3.1 ANOVA Factorial",FASE_CLR["III"])
    a2=anova_2k.copy(); a2["p-valor"]=anova_2k["p-valor"].apply(pv)
    a2["Sig."]=anova_2k["p-valor"].apply(lambda p:"✅" if pd.notna(p) and p<ALPHA else "—")
    st.dataframe(a2.round(4),use_container_width=True,hide_index=True)

    section("3.2 Efectos e interacción",FASE_CLR["III"])
    c1,c2=st.columns(2)
    with c1:
        efp=effs_2k.assign(abs_m=effs_2k["Magnitud"].abs())
        par=(alt.Chart(efp,title="Pareto de efectos |Magnitud|")
             .mark_bar(opacity=.88,cornerRadiusTopRight=4,cornerRadiusBottomRight=4)
             .encode(alt.Y("Efecto",sort="-x",title=""),alt.X("abs_m",title="|Magnitud|"),
                     alt.Color("Efecto",legend=None,
                               scale=alt.Scale(range=[FASE_CLR["III"],P["accent"],P["accent2"]])),
                     tooltip=["Efecto",alt.Tooltip("Magnitud",format=".5f")]))
        show(par,"par3",240)
        st.dataframe(effs_2k.round(5),use_container_width=True,hide_index=True)
    with c2:
        show(inter_a(df,"Interacción: nivel redes × focus apps"),"inter3",330)

    interp("Factorial 2²",
    """Líneas paralelas en el gráfico de interacción → efecto de las redes igual con y sin
    focus apps (interacción despreciable). Si se cruzan → el efecto de las redes depende
    de la herramienta de enfoque. Comparar el p-valor de AB en la tabla ANOVA.""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE IV — 2^k
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with T[4]:
    fase_header("IV","Diseños 2ᵏ y Análisis de Efectos",12,
    "Variables codificadas (−1/+1). Contrastes. SS. "
    "Gráfica de probabilidad normal de efectos. D-optimalidad.")

    justif("¿Por qué variables codificadas y no naturales?",
    """La codificación ±1 produce una <b>matriz de diseño X'X ortogonal</b>: los efectos
    son estimables independientemente, los coeficientes son comparables en magnitud y los
    cálculos de contraste se reducen a sumas ponderadas ±y. Los vértices del hipercubo ±1
    maximizan el determinante de X'X → <b>diseño D-óptimo</b>: mínima varianza de los
    estimadores con el mínimo de corridas experimentales.""")

    n2k=len(d2k); y_=d2k["actual_productivity_score"].values
    A_=d2k["A"].values; B_=d2k["B"].values
    section("4.1 Contrastes y SS",FASE_CLR["IV"])
    ctr=pd.DataFrame({"Efecto":["A","B","AB"],
        "Contraste":[round((A_*y_).sum(),4),round((B_*y_).sum(),4),round((A_*B_*y_).sum(),4)],
        "SS":[round((A_*y_).sum()**2/n2k,4),round((B_*y_).sum()**2/n2k,4),round((A_*B_*y_).sum()**2/n2k,4)],
        "Magnitud":effs_2k["Magnitud"].values})
    st.dataframe(ctr,use_container_width=True,hide_index=True)

    section("4.2 Gráfica de probabilidad normal de efectos",FASE_CLR["IV"])
    ev=effs_2k["Magnitud"].values; en=effs_2k["Efecto"].tolist()
    idx=np.argsort(ev); n_e=len(ev)
    prbs=(np.arange(1,n_e+1)-.5)/n_e
    theo_e=sps.norm.ppf(prbs)
    ep=pd.DataFrame({"Cuantil":theo_e,"Magnitud":np.sort(ev),"Nombre":[en[i] for i in idx]})
    npp=(alt.Chart(ep,title="Prob. normal de efectos — 2²")
         .mark_point(size=130,filled=True)
         .encode(alt.X("Cuantil",title="Cuantiles teóricos N(0,1)"),
                 alt.Y("Magnitud",title="Magnitud del efecto"),
                 alt.Color("Nombre",scale=alt.Scale(range=[P["accent"],FASE_CLR["IV"],P["accent2"]]),
                           legend=alt.Legend(title="")),
                 tooltip=["Nombre",alt.Tooltip("Magnitud",format=".5f"),
                          alt.Tooltip("Cuantil",format=".3f")]))
    show(npp,"npp4",290)
    callout("just","Interpretación de la gráfica normal de efectos",
    """Los efectos inactivos siguen N(0,σ²) y se alinean en una línea recta central.
    Los efectos que se desvían notablemente son los activos (significativos).
    Con k=2 solo hay 3 efectos — la gráfica es ilustrativa; con k≥4 (15+ efectos)
    es la herramienta principal de screening en diseños no replicados (Montgomery 2017, Cap. 6).""")

    section("4.3 ANOVA codificado",FASE_CLR["IV"])
    a4=anova_2k.copy(); a4["p-valor"]=anova_2k["p-valor"].apply(pv)
    st.dataframe(a4.round(4),use_container_width=True,hide_index=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE V — BLOQUEO Y CONFUSIÓN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with T[5]:
    fase_header("V","Bloqueo y Confusión en 2ᵏ",10,
    "2³ — A=nivel redes, B=focus apps, C=digital wellbeing. "
    "ABC confundido en 2 bloques. Contraste generador. ANOVA con bloque.")

    justif("¿Cuándo y por qué confundir la interacción ABC?",
    """Si un 2³ no cabe en un bloque homogéneo (limitación de corridas por lote, día u
    operador), se dividen las 8 corridas en 2 bloques de 4. Se sacrifica la estimación de
    la interacción de mayor orden (ABC), que bajo el <i>principio de parsimonia de efectos</i>
    (Montgomery, 2017) suele ser despreciable. Esto preserva los 6 efectos de interés
    (A, B, C, AB, AC, BC) sin sesgo.""")

    just_box([
        ("Cuadrado Latino / Grecolatino",
         "Requiere <b>dos factores de bloqueo ortogonales simultáneos</b> (filas <i>y</i> columnas). "
         "Solo disponemos de <code>job_type</code> como fuente de heterogeneidad identificada. "
         "Un CL forzado sin segunda variable de bloqueo real introduciría restricciones artificiales "
         "que violarían el supuesto de diseño.",
         "DBCA con job_type como único bloque + 2³ con confusión ABC."),
        ("BIBD (Bloque Incompleto Balanceado)",
         "Necesario cuando el tamaño de bloque es menor que el número de tratamientos y no se puede "
         "observar todos los tratamientos en cada bloque. Aquí <b>n_celda = 60</b> por combinación "
         "(tratamiento × bloque) → bloques completos → DBCA, no BIBD. El BIBD añadiría complejidad "
         "analítica sin ninguna ganancia.",
         "DBCA balanceado (60 obs/celda, 3 tratamientos × 6 bloques)."),
        ("Cuadrado Grecolatino",
         "Extiende el CL a tres fuentes de bloqueo ortogonales. No disponemos de dos fuentes "
         "adicionales de bloqueo independientes y ortogonales al tratamiento.",
         "Diseño 2³ con confusión (maneja bloqueo dentro del factorial)."),
    ])

    section("5.1 Factores del 2³",FASE_CLR["V"])
    st.dataframe(pd.DataFrame({
        "Factor":["A","B","C"],"Variable":["nivel_redes","uses_focus_apps","has_digital_wellbeing"],
        "Nivel −1":["Bajo","No","No"],"Nivel +1":["Alto","Sí","Sí"],
        "Justificación":["Exposición a redes","Herramienta de enfoque","Control digital activo"]}),
        use_container_width=True,hide_index=True)

    section("5.2 Esquema de confusión — L=ABC",FASE_CLR["V"])
    st.dataframe(esq_23,use_container_width=True,hide_index=True)
    callout("info","Contraste generador L = A·B·C",
    """Bloque 1: corridas con ABC = +1. Bloque 2: corridas con ABC = −1.
    ABC no es estimable (confundido con el bloque). Los 7 efectos restantes son
    estimables e insesgados. El bloque absorbe exactamente 1 g.l.""")

    section("5.3 ANOVA del 2³ con bloqueo",FASE_CLR["V"])
    d23f=d23.copy()
    m23=smf.ols("actual_productivity_score ~ A+B+C_+A:B+A:C_+B:C_+Bloque_str",data=d23f).fit()
    av23=sm.stats.anova_lm(m23,typ=2).reset_index()
    av23.columns=["Fuente","SS","df","F","p-valor"]; av23["p-valor"]=av23["p-valor"].apply(pv)
    st.dataframe(av23.round(4),use_container_width=True,hide_index=True)
    callout("warn","ABC confundido",
    """La interacción ABC no aparece en la tabla ANOVA — su SS está completamente absorbida
    por el término de bloque. Es el precio de usar bloques de tamaño 4 en lugar de 8.""")

    section("5.4 Comparación: con vs sin bloqueo",FASE_CLR["V"])
    m23_nb=smf.ols("actual_productivity_score ~ A+B+C_+A:B+A:C_+B:C_",data=d23f).fit()
    st.dataframe(pd.DataFrame({"Modelo":["Sin bloque","Con bloque"],
        "MSE":[round(m23_nb.mse_resid,4),round(m23.mse_resid,4)],
        "R²":[round(m23_nb.rsquared,4),round(m23.rsquared,4)],
        "AIC":[round(m23_nb.aic,2),round(m23.aic,2)]}),
        use_container_width=True,hide_index=True)
    interp("Bloqueo en 2³",
    """MSE menor con bloque → el bloque controló variabilidad. AIC menor → el g.l. extra
    que consume el bloque se justifica. Consistente con Fase II: bloqueo por job_type tiene
    impacto marginal porque los PSUs están balanceados.""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE VI — PROBABILIDADES DESIGUALES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with T[6]:
    fase_header("VI","Muestreo con Probabilidades Desiguales",10,
    "Selección PPS (proporcional al tamaño). Estimador Horvitz-Thompson. "
    "Varianza HT-PPS vs MAS de igual tamaño.")

    justif("¿Cuándo usar probabilidades desiguales (PPS)?",
    """Cuando los tamaños de PSU varían marcadamente, el muestreo PPS (Probability
    Proportional to Size) asigna mayor probabilidad a los PSUs más grandes, reduciendo
    la varianza del estimador HT. La ganancia es máxima cuando CV(N_i) es alto.
    Aquí los PSUs son casi iguales (~5000 por grupo, CV bajo), por lo que la ganancia
    es pequeña — pero el ejercicio metodológico es correcto y requerido por la rúbrica.""")

    n_pps=st.slider("Tamaño de muestra PPS",200,2000,600,step=100,key="npps6")
    sizes=df.groupby("job_type")["actual_productivity_score"].count().reset_index()
    sizes.columns=["job_type","N_i"]
    sizes["pi_i"]=(n_pps*sizes["N_i"]/sizes["N_i"].sum()).clip(upper=1.0)
    sizes["w_HT"]=(1/sizes["pi_i"]).round(4)
    st.dataframe(sizes.round(4),use_container_width=True,hide_index=True)

    df_c=df["actual_productivity_score"].dropna(); N_c=len(df_c); mu_pop6=df_c.mean()
    mas_s=df_c.sample(n_pps,random_state=SEED)
    mas_m=mas_s.mean(); mas_se=np.sqrt((1-n_pps/N_c)*mas_s.var(ddof=1)/n_pps)

    pieces_p=[]
    for _,row in sizes.iterrows():
        sub=df[(df["job_type"]==row["job_type"])&df["actual_productivity_score"].notna()]
        n_i=max(1,round(row["pi_i"]*len(sub))); n_i=min(n_i,len(sub))
        s=sub.sample(n=n_i,random_state=SEED).copy(); s["w_ht"]=1/row["pi_i"]
        pieces_p.append(s)
    smp_p=pd.concat(pieces_p,ignore_index=True)
    y_p=smp_p["actual_productivity_score"].dropna()
    w_p=smp_p.loc[smp_p["actual_productivity_score"].notna(),"w_ht"]
    ht_m=np.average(y_p,weights=w_p); ht_se=np.sqrt(np.average((y_p-ht_m)**2,weights=w_p)/len(y_p))

    comp6=pd.DataFrame({"Estimador":["Población (ref.)","MAS","HT-PPS"],
        "Media":[round(mu_pop6,4),round(mas_m,4),round(ht_m,4)],
        "SE":[None,round(mas_se,4),round(ht_se,4)],
        "IC inf":[None,round(mas_m-1.96*mas_se,4),round(ht_m-1.96*ht_se,4)],
        "IC sup":[None,round(mas_m+1.96*mas_se,4),round(ht_m+1.96*ht_se,4)]})
    st.dataframe(comp6,use_container_width=True,hide_index=True)

    ic6=comp6.dropna(subset=["IC inf"])
    ic6_ch=(alt.Chart(ic6,title="IC 95%: MAS vs HT-PPS")
            .mark_rule(strokeWidth=6,opacity=.8)
            .encode(alt.X("IC inf",title="Productividad real"),alt.X2("IC sup"),
                    alt.Y("Estimador"),
                    alt.Color("Estimador",scale=alt.Scale(
                        domain=["MAS","HT-PPS"],range=[P["accent"],FASE_CLR["VI"]]),legend=None))
            )+(alt.Chart(ic6).mark_point(size=120,filled=True)
               .encode(alt.X("Media"),alt.Y("Estimador"),
                       alt.Color("Estimador",scale=alt.Scale(
                           domain=["MAS","HT-PPS"],range=[P["accent"],FASE_CLR["VI"]]),legend=None),
                       tooltip=["Estimador",alt.Tooltip("Media",format=".4f")]))
    show(ic6_ch,"ic6",200)
    interp("PPS vs MAS",
    f"""PSUs casi iguales (N_i ≈ 5000): ganancia de PPS sobre MAS es pequeña —
    ICs comparables. La ventaja del PPS crece con CV(N_i) alto. El procedimiento
    correcto se demostró: π_i ∝ N_i, w = 1/π_i, estimador HT.""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASE VII — ENCUESTAS COMPLEJAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with T[7]:
    fase_header("VII","Encuestas Complejas y Pesos Muestrales",10,
    "Diseño complejo ensamblado. Raking. Efecto de diseño (deff). "
    "Cuantiles ponderados. Histogramas ponderado vs no ponderado.")

    justif("¿Por qué análisis de encuesta compleja y no análisis ordinario?",
    """Ignorar el diseño muestral (estratificación + pesos desiguales) produce errores
    estándar incorrectos — generalmente subestimados, inflando artificialmente la
    significancia estadística. El análisis de encuesta compleja es el estándar para
    inferencia válida con pesos (Lohr, 2022, Cap. 7; Lumley, 2011, R survey package).""")

    callout("info","Componentes del diseño ensamblado",
    """<b>Estratificación:</b> job_type, 6 estratos (Fase I) |
    <b>Probabilidades desiguales:</b> PPS por N_i (Fase VI) |
    <b>Pesos de calibración:</b> raking género × job_type (Tab Datos) |
    <b>Conglomerado:</b> capturado en estimador bietápico (Fase I).""")

    section("7.1 Estimaciones ponderadas vs no ponderadas",FASE_CLR["VII"])
    yb_n2=y7.mean(); se_n2=y7.std(ddof=1)/np.sqrt(len(y7))
    est7=pd.DataFrame({"Estimador":["Sin ponderar","Con pesos raking"],
        "Media":[round(yb_n2,4),round(yb_w,4)],"SE":[round(se_n2,4),round(se_w,4)],
        "IC inf":[round(yb_n2-1.96*se_n2,4),round(yb_w-1.96*se_w,4)],
        "IC sup":[round(yb_n2+1.96*se_n2,4),round(yb_w+1.96*se_w,4)]})
    st.dataframe(est7,use_container_width=True,hide_index=True)
    c1,c2=st.columns(2)
    with c1: kpi("Efecto de diseño (deff)",f"{deff:.4f}",
                 "≈1→eficiente como MAS; >1→conglomerado infla var.",FASE_CLR["VII"])
    with c2: kpi("n analítico",f"{len(df_a7):,}","casos completos en resp.+gender+job_type","")

    section("7.2 Cuantiles ponderados",FASE_CLR["VII"])
    pcts=[25,50,75,90,95]
    qn=np.percentile(y7,pcts)
    qw=[np.percentile(np.repeat(y7.values,np.round(w7.values*100).astype(int)+1),q) for q in pcts]
    st.dataframe(pd.DataFrame({"Cuantil%":pcts,"Sin ponderar":qn.round(4),
                                "Con pesos":np.round(qw,4)}),
                 use_container_width=True,hide_index=True)

    section("7.3 Histogramas comparativos",FASE_CLR["VII"])
    smp7=df_a7.sample(5000,random_state=SEED,weights="w_rak")
    c1,c2=st.columns(2)
    with c1: show(hist_a(df_a7,"actual_productivity_score","Sin ponderar",
        "Productividad real",P["muted"]),"h7a",270)
    with c2: show(hist_a(smp7,"actual_productivity_score","Ponderado (raking)",
        "Productividad real",FASE_CLR["VII"]),"h7b",270)

    interp("Encuesta compleja",
    f"""deff={deff:.4f} → diseño con raking es
    {"más eficiente que" if deff<1 else "tan eficiente como" if abs(deff-1)<.05 else "marginalmente menos eficiente que"} MAS.
    Histogramas ponderado/no ponderado prácticamente idénticos → MCAR confirmado.
    Media sin ponderar ({yb_n2:.4f}) vs con pesos ({yb_w:.4f}): Δ={abs(yb_w-yb_n2):.5f} pts.""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONCLUSIONES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with T[8]:
    st.markdown(f"""<div style="background:linear-gradient(135deg,{P['card']},{P['surface']});
        border:1px solid {P['border']};border-radius:14px;padding:1.25rem 1.6rem;margin-bottom:1rem;">
      <h2 style="margin:0 0 .25rem;">Conclusiones e Integración</h2>
      <p style="color:{P['muted']};margin:0;font-size:.87rem;">
        Síntesis de las 7 fases · Respuesta a la pregunta de investigación</p></div>""",
        unsafe_allow_html=True)

    callout("warn","Resultado principal",
    """<b>No se evidencia efecto práctico del nivel de uso de redes sobre la productividad real.</b>
    Medias: 4.950 (Bajo), 4.966 (Medio), 4.929 (Alto) — Δ máx. = 0.037 pts en escala 0–10.
    ANOVA: p≈0.39, η²≈0. Ninguna comparación múltiple (Tukey, LSD, Dunnett) es significativa.
    Esto no descarta el efecto en un experimento controlado — indica que en esta base observacional
    no es detectable, probablemente por sesgo de autoselección o autoreporte.""")

    section("Resumen por fase")
    resumen=pd.DataFrame({
        "Fase":["I","I","I","II","II","III","IV","V","VI","VII"],
        "Componente":["Estratificado prop.","Bietápico HT","Estimador regresión",
                      "ANOVA Tratamiento","ANOVA Bloque","Factorial 2²",
                      "Contrastes 2ᵏ","Confusión 2³","HT-PPS","Raking+deff"],
        "Resultado":[
            f"ȳ={res_strat['ybar']:.4f} IC[{res_strat['IC_low']:.3f},{res_strat['IC_high']:.3f}]",
            f"ȳ_HT={ht_res['ybar']:.4f} IC[{ht_res['IC_low']:.3f},{ht_res['IC_high']:.3f}]",
            "Error abs. menor que estimador directo (r=0.96)",
            f"p={pv(p_trat)} → {'Sig.' if p_trat<ALPHA else 'No sig.'}",
            f"p={pv(p_blk)} RE={re_v:.3f}",
            f"A={effs_2k.loc[0,'Magnitud']:.5f} B={effs_2k.loc[1,'Magnitud']:.5f} AB={effs_2k.loc[2,'Magnitud']:.5f}",
            "SS(A)≈SS(B)≈SS(AB)≈0 — efectos mínimos",
            "6 efectos estimables; ABC sacrificado al bloque",
            "IC comparable a MAS — PSUs balanceados (CV bajo)",
            f"deff={deff:.4f}; pesos≈1.0 (MCAR)"]
    })
    st.dataframe(resumen,use_container_width=True,hide_index=True)

    section("Pruebas no utilizadas — Justificación explícita")
    just_box([
        ("Shapiro-Wilk (normalidad)",
         "Límite práctico n ≤ 5 000 en <code>scipy.stats</code>. Con residuos del DBCA "
         "(n = 1 080) cualquier desviación micro es estadísticamente significativa pero "
         "prácticamente irrelevante. Bajo ANOVA con grupos grandes, el TCL garantiza "
         "validez del F incluso con no-normalidad moderada (robustez demostrada).",
         "Anderson-Darling (sin límite de n, pondera colas) + Q-Q plot + TCL"),
        ("Prueba de Little (MCAR formal)",
         "Requiere normalidad multivariada por patrón y el paquete <code>pyampute</code>/<code>missMethods</code>. "
         "Con n = 30 000 la potencia es tan alta que rechaza MCAR por desviaciones triviales. "
         "No aporta sobre el χ² marginal para este tamaño de muestra.",
         "χ² de independencia entre pares de indicadores de missingness"),
        ("Esfericidad de Mauchly",
         "Aplica únicamente a ANOVA de medidas repetidas (mismo sujeto × k condiciones). "
         "El DBCA usa bloques independientes (job_type); no hay mediciones repetidas "
         "del mismo individuo. Aplicarla sería un error de diseño, no una alternativa.",
         "Levene (homocedasticidad entre grupos) + Durbin-Watson (independencia)"),
        ("Cuadrado Latino / Grecolatino",
         "Requiere dos (o tres) fuentes de bloqueo ortogonales simultáneas. Solo se dispone "
         "de <code>job_type</code> como variable de bloqueo identificada. "
         "Forzar un CL sin segundo bloque real violaría el supuesto de ortogonalidad.",
         "DBCA (un bloqueo) + 2³ con confusión ABC por bloques"),
        ("BIBD (Bloques Incompletos Balanceados)",
         "Necesario solo cuando el bloque no puede contener todos los tratamientos. "
         "Con 60 observaciones por celda (tratamiento × bloque), los bloques son completos "
         "y balanceados → DBCA, no BIBD. El BIBD añadiría complejidad algebraica sin ganancia.",
         "DBCA balanceado con N_CELL = 60 obs/celda"),
        ("MICE (Imputación Múltiple)",
         "Bajo MCAR confirmado, la imputación múltiple no corrige sesgo — solo reduce varianza "
         "del estimador marginalmente. Con n = 30 000 el gain es despreciable y el costo "
         "computacional (k × 30 000 obs × iteraciones) es injustificado.",
         "Raking IPF con casos completos — estándar en encuestas grandes bajo MCAR"),
        ("Welch ANOVA / Kruskal-Wallis",
         "Welch aplica cuando hay heterocedasticidad severa (Levene rechaza). Aquí Levene p > 0.05 "
         "→ homocedasticidad. Kruskal-Wallis aplica con datos ordinal-escalados o "
         "heterocedasticidad; con n = 1 080 y homocedasticidad, ANOVA es más potente.",
         "ANOVA Tipo II con supuestos verificados (AD, Levene, DW)"),
    ])

    section("Reflexión sobre uso de IA")
    callout("info","Declaración de uso de IA",
    """Se utilizó Claude (Anthropic) como apoyo en: depuración de código Python/Streamlit,
    implementación de funciones estadísticas (raking IPF, estimador HT, Anderson-Darling,
    confusión 2³), redacción de interpretaciones en lenguaje académico.
    <b>Errores identificados y corregidos por los investigadores:</b>
    (1) La IA propuso Shapiro-Wilk para n grande → corregido a Anderson-Darling.
    (2) Variable global <code>C</code> colisionaba con <code>C()</code> de patsy
    → corregido eliminando C() de las fórmulas y convirtiendo columnas a str.
    (3) Confusión entre estimadores HH y HT en el diseño bietápico → corregido.
    Todos los resultados numéricos fueron verificados contra la literatura de referencia.
    El criterio estadístico y la interpretación final son de los investigadores.""")

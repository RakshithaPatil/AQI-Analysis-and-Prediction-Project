"""
AQI Analysis & Prediction — Interactive Dashboard
==================================================
Loads:  data/raw/india_aqi_dataset.csv  (21,920 rows · 20 stations · 11 cities)
Run:    streamlit run dashboard/app.py
URL:    http://localhost:8501
"""

import warnings, sys
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "raw" / "india_aqi_dataset.csv"

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="🌬️ India AQI Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
# THEME & CSS
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg,#0d0d1a 0%,#0f1f3d 50%,#0d0d1a 100%);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: rgba(8,10,30,0.92);
    border-right: 1px solid rgba(255,255,255,0.07);
}
/* glass card */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 8px;
}
.card:hover { border-color: rgba(255,255,255,0.22); }

/* kpi */
.kpi-num  { font-size:2.4rem; font-weight:800; line-height:1.1; }
.kpi-lbl  { font-size:.70rem; letter-spacing:.14em; text-transform:uppercase; opacity:.55; }
.kpi-sub  { font-size:.80rem; opacity:.70; margin-top:3px; }

/* section title */
.sec { font-size:1.05rem; font-weight:700; letter-spacing:.06em;
       border-bottom:2px solid rgba(255,255,255,.08);
       padding-bottom:8px; margin-bottom:12px; }

/* pill */
.pill { display:inline-block; padding:2px 11px; border-radius:20px;
        font-size:.72rem; font-weight:700; margin:2px; }

#MainMenu, footer { visibility:hidden; }
div[data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════
AQI_BANDS = [
    (0,   50,  "Good",                           "#00e676"),
    (51,  100, "Moderate",                       "#ffee58"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ffa726"),
    (151, 200, "Unhealthy",                      "#ef5350"),
    (201, 300, "Very Unhealthy",                 "#ab47bc"),
    (301, 500, "Hazardous",                      "#b71c1c"),
]
CAT_COLOR = {lb: c for _, _, lb, c in AQI_BANDS}
CAT_ORDER = [lb for _, _, lb, _ in AQI_BANDS]

POLS = ["pm25","pm10","no2","so2","co","o3","nh3"]
POL_LABEL = {"pm25":"PM₂.₅","pm10":"PM₁₀","no2":"NO₂",
             "so2":"SO₂","co":"CO","o3":"O₃","nh3":"NH₃"}
POL_COLOR = {"pm25":"#ef5350","pm10":"#ff7043","no2":"#ab47bc",
             "so2":"#ffa726","co":"#42a5f5","o3":"#26c6da","nh3":"#a5d6a7"}

THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(255,255,255,0.02)",
    font=dict(color="#dce1f0", family="Inter, sans-serif", size=12),
    title_font=dict(size=13, color="#ffffff"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,.1)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,.05)", showline=False, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,.05)", showline=False, zeroline=False),
    margin=dict(l=8, r=8, t=38, b=8),
)

def T(fig, h=360):
    fig.update_layout(**THEME, height=h)
    return fig

def aqi_col(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "#90a4ae"
    for lo,hi,_,c in AQI_BANDS:
        if lo<=v<=hi: return c
    return "#b71c1c"

def aqi_cat(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "Unknown"
    for lo,hi,lb,_ in AQI_BANDS:
        if lo<=v<=hi: return lb
    return "Hazardous"

# ════════════════════════════════════════════════════════════════
# DATA LOAD
# ════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["aqi_category"] = df["aqi_category"].fillna("Unknown")
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["quarter"] = df["date"].dt.quarter.map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
    df["dow"] = df["date"].dt.day_name()
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    return df

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
def sidebar(df):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:12px 0 6px'>
          <div style='font-size:2.4rem'>🌬️</div>
          <div style='font-size:1.1rem;font-weight:800'>India AQI Monitor</div>
          <div style='font-size:.68rem;opacity:.45;margin-top:2px'>
            20 Stations · 11 Cities · 2022–2024
          </div>
        </div>
        <hr style='border-color:rgba(255,255,255,.08)'>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎛️ Filters")

        all_cities = sorted(df["city"].unique())
        cities = st.multiselect("🏙️ City", all_cities, default=all_cities)

        all_states = sorted(df[df["city"].isin(cities)]["state"].unique()) if cities else []
        stations_in = sorted(df[df["city"].isin(cities)]["station_name"].unique()) if cities else []
        stations = st.multiselect("📍 Station", stations_in, default=stations_in)

        years = sorted(df["year"].unique())
        sel_years = st.multiselect("📅 Year", years, default=years)

        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
        dr = st.date_input("📆 Date Range",
                           value=(min_d, max_d),
                           min_value=min_d, max_value=max_d)
        d0 = pd.Timestamp(dr[0]) if len(dr)>0 else pd.Timestamp(min_d)
        d1 = pd.Timestamp(dr[1]) if len(dr)>1 else pd.Timestamp(max_d)

        focus_pol = st.selectbox("🧪 Focus Pollutant", POLS,
                                  format_func=lambda x: POL_LABEL[x])

        cats = sorted(df["aqi_category"].unique())
        sel_cats = st.multiselect("🎨 AQI Category", cats, default=cats)

        st.markdown("<hr style='border-color:rgba(255,255,255,.08)'>", unsafe_allow_html=True)
        st.caption("📁 Source: india_aqi_dataset.csv · 21,920 rows")

    return dict(cities=cities, stations=stations, years=sel_years,
                d0=d0, d1=d1, pol=focus_pol, cats=sel_cats)


def filt(df, f):
    return df[
        df["city"].isin(f["cities"]) &
        df["station_name"].isin(f["stations"]) &
        df["year"].isin(f["years"]) &
        (df["date"] >= f["d0"]) & (df["date"] <= f["d1"]) &
        df["aqi_category"].isin(f["cats"])
    ].copy()

# ════════════════════════════════════════════════════════════════
# SECTION 1  —  KPI CARDS
# ════════════════════════════════════════════════════════════════
def s_kpis(df):
    latest = df.sort_values("date").groupby("station_id").last().reset_index()
    avg    = df["aqi"].mean()
    worst  = latest.loc[latest["aqi"].idxmax()] if not latest.empty else None
    best   = latest.loc[latest["aqi"].idxmin()] if not latest.empty else None
    unheal = (df["aqi"] > 150).mean() * 100
    total  = len(df)

    def card(icon, lbl, val, col, sub=""):
        return f"""<div class="card" style="text-align:center;border-top:3px solid {col}">
          <div style="font-size:1.6rem">{icon}</div>
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-num" style="color:{col}">{val}</div>
          <div class="kpi-sub">{sub}</div></div>"""

    c = st.columns(5)
    c[0].markdown(card("🌫️","Avg AQI", f"{avg:.0f}", aqi_col(avg), aqi_cat(avg)),
                  unsafe_allow_html=True)
    c[1].markdown(card("⚠️","Worst Station",
        f"{worst['aqi']:.0f}" if worst is not None else "—",
        "#ef5350",
        worst['station_name'] if worst is not None else ""),
        unsafe_allow_html=True)
    c[2].markdown(card("✅","Best Station",
        f"{best['aqi']:.0f}" if best is not None else "—",
        "#00e676",
        best['station_name'] if best is not None else ""),
        unsafe_allow_html=True)
    c[3].markdown(card("🚨","Unhealthy Days", f"{unheal:.1f}%", "#ffa726","AQI > 150"),
                  unsafe_allow_html=True)
    c[4].markdown(card("📊","Total Records", f"{total:,}", "#64b5f6",
                       f"{df['station_id'].nunique()} stations"),
                  unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# SECTION 2  —  AQI GAUGE + MAP
# ════════════════════════════════════════════════════════════════
def s_gauge_map(df):
    col_g, col_m = st.columns([1, 2])

    # Gauge
    with col_g:
        st.markdown('<div class="sec">🎯 Overall AQI Gauge</div>', unsafe_allow_html=True)
        avg = df["aqi"].mean()
        prev = df[df["date"] < df["date"].max() - pd.Timedelta(days=30)]["aqi"].mean()
        col = aqi_col(avg)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(avg, 1),
            delta={"reference": prev, "valueformat":".1f",
                   "increasing":{"color":"#ef5350"}, "decreasing":{"color":"#00e676"}},
            number={"font":{"size":54,"color":col}},
            gauge={
                "axis":{"range":[0,500],"tickwidth":1,
                         "tickcolor":"rgba(255,255,255,.25)",
                         "tickfont":{"color":"#aaa","size":9}},
                "bar":{"color":col,"thickness":0.28},
                "bgcolor":"rgba(0,0,0,0)",
                "bordercolor":"rgba(255,255,255,.08)",
                "steps":[
                    {"range":[0,50],   "color":"rgba(0,230,118,.12)"},
                    {"range":[51,100], "color":"rgba(255,238,88,.12)"},
                    {"range":[101,150],"color":"rgba(255,167,38,.12)"},
                    {"range":[151,200],"color":"rgba(239,83,80,.12)"},
                    {"range":[201,300],"color":"rgba(171,71,188,.12)"},
                    {"range":[301,500],"color":"rgba(183,28,28,.12)"},
                ],
                "threshold":{"line":{"color":col,"width":5},
                              "thickness":0.85,"value":avg},
            },
            title={"text":f"<b>{aqi_cat(avg)}</b>",
                   "font":{"size":14,"color":col}},
        ))
        T(fig, 300)
        st.plotly_chart(fig, width="stretch")

        # Legend pills
        st.markdown('<div style="display:flex;flex-wrap:wrap;gap:3px;margin-top:4px">' +
            "".join([f'<span class="pill" style="background:{c}18;color:{c};border:1px solid {c}44">{lb}</span>'
                     for _,_,lb,c in AQI_BANDS]) +
            "</div>", unsafe_allow_html=True)

    # Map
    with col_m:
        st.markdown('<div class="sec">🗺️ Station Map — Latest AQI</div>', unsafe_allow_html=True)
        latest = df.sort_values("date").groupby("station_id").last().reset_index()
        latest["cat"] = latest["aqi"].apply(aqi_cat)
        latest["sz"]  = np.clip(latest["aqi"]/4, 8, 55)
        latest["hover_pm25"] = latest["pm25"].round(1)

        fig2 = px.scatter_map(
            latest, lat="latitude", lon="longitude",
            size="sz", color="aqi",
            color_continuous_scale=[
                [0.00,"#00e676"],[0.10,"#ffee58"],
                [0.30,"#ffa726"],[0.55,"#ef5350"],
                [0.75,"#ab47bc"],[1.00,"#7e0023"],
            ],
            range_color=[0,350],
            hover_name="station_name",
            hover_data={"city":True,"aqi":":.0f","cat":True,
                        "hover_pm25":True,"latitude":False,
                        "longitude":False,"sz":False},
            custom_data=["city","cat","hover_pm25"],
            map_style="carto-darkmatter",
            zoom=4.0, center={"lat":22,"lon":79},
        )
        fig2.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>City: %{customdata[0]}<br>"
                          "AQI: %{marker.color:.0f} (%{customdata[1]})<br>"
                          "PM₂.₅: %{customdata[2]} µg/m³<extra></extra>"
        )
        T(fig2, 330)
        fig2.update_layout(
            coloraxis_colorbar=dict(title="AQI",
                tickfont=dict(color="#aaa"), title_font=dict(color="#aaa")),
        )
        st.plotly_chart(fig2, width="stretch")

# ════════════════════════════════════════════════════════════════
# SECTION 3  —  AQI TRENDS
# ════════════════════════════════════════════════════════════════
def s_trend(df):
    st.markdown('<div class="sec">📈 AQI Trends Over Time</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["🏙️ City (7-day MA)","📍 Station Detail","📅 Monthly"])

    with t1:
        cd = df.groupby(["date","city"])["aqi"].mean().reset_index()
        cd["ma7"] = cd.groupby("city")["aqi"].transform(
            lambda x: x.rolling(7, min_periods=1).mean())
        fig = px.line(cd, x="date", y="ma7", color="city",
                      labels={"ma7":"7-day Avg AQI","date":""},
                      color_discrete_sequence=px.colors.qualitative.Set2)
        for lo,hi,lb,c in AQI_BANDS:
            fig.add_hrect(y0=lo, y1=min(hi,420),
                          fillcolor=c, opacity=0.03, line_width=0)
        fig.add_hline(y=100, line_dash="dot",
                      line_color="rgba(255,167,38,.5)",
                      annotation_text="Moderate limit",
                      annotation_font=dict(color="rgba(255,167,38,.7)",size=10))
        T(fig, 380); fig.update_traces(line_width=2.2)
        st.plotly_chart(fig, width="stretch")

    with t2:
        sd = df.groupby(["date","station_name"])["aqi"].mean().reset_index()
        sd["ma7"] = sd.groupby("station_name")["aqi"].transform(
            lambda x: x.rolling(7, min_periods=1).mean())
        fig2 = px.line(sd, x="date", y="ma7", color="station_name",
                       labels={"ma7":"7-day Avg AQI","date":""},
                       color_discrete_sequence=px.colors.qualitative.Alphabet)
        T(fig2, 380); fig2.update_traces(line_width=1.8, opacity=0.85)
        st.plotly_chart(fig2, width="stretch")

    with t3:
        df["ym"] = df["date"].dt.to_period("M").astype(str)
        mo = df.groupby(["ym","city"])["aqi"].mean().reset_index()
        fig3 = px.bar(mo, x="ym", y="aqi", color="city", barmode="group",
                      labels={"aqi":"Avg AQI","ym":"Month"},
                      color_discrete_sequence=px.colors.qualitative.Set2)
        T(fig3, 380)
        fig3.update_layout(xaxis_tickangle=-45,
                           xaxis=dict(tickmode="auto", nticks=18,
                                      gridcolor="rgba(255,255,255,.05)"))
        st.plotly_chart(fig3, width="stretch")

# ════════════════════════════════════════════════════════════════
# SECTION 4  —  POLLUTANT ANALYSIS
# ════════════════════════════════════════════════════════════════
def s_pollutants(df, pol):
    st.markdown('<div class="sec">🧪 Pollutant Deep Dive</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        t1, t2 = st.tabs([f"📊 All Pollutants","🎯 Focus: {POL_LABEL[pol]}"])
        with t1:
            pd2 = df.groupby("date")[[p for p in POLS if p in df.columns]].mean().reset_index()
            fig = go.Figure()
            for p in POLS:
                if p not in pd2.columns: continue
                # Convert hex colour to rgba for fill (Plotly 7 requires valid CSS colour)
                hex_c = POL_COLOR[p]
                r_val = int(hex_c[1:3],16); g_val = int(hex_c[3:5],16); b_val = int(hex_c[5:7],16)
                fill_rgba = f"rgba({r_val},{g_val},{b_val},0.09)"
                fig.add_trace(go.Scatter(
                    x=pd2["date"],
                    y=pd2[p].rolling(7, min_periods=1).mean(),
                    name=POL_LABEL[p], mode="lines",
                    line=dict(color=POL_COLOR[p], width=2),
                    fill="tozeroy" if p==pol else "none",
                    fillcolor=fill_rgba,
                ))
            T(fig, 320)
            fig.update_layout(title="7-day Rolling Avg — All Pollutants")
            st.plotly_chart(fig, width="stretch")

        with t2:
            city_pol = df.groupby(["date","city"])[pol].mean().reset_index()
            city_pol["ma7"] = city_pol.groupby("city")[pol].transform(
                lambda x: x.rolling(7, min_periods=1).mean())
            fig2 = px.line(city_pol, x="date", y="ma7", color="city",
                           labels={"ma7":f"{POL_LABEL[pol]} (µg/m³)","date":""},
                           color_discrete_sequence=px.colors.qualitative.Pastel)
            T(fig2, 320)
            fig2.update_layout(title=f"{POL_LABEL[pol]} Trend by City")
            st.plotly_chart(fig2, width="stretch")

    with col2:
        # Radar — latest avg per city (normalised to moderate threshold)
        lat_city = df.sort_values("date").groupby("city")[[p for p in POLS if p in df.columns]].last().reset_index()
        thresh = {"pm25":35.4,"pm10":154,"no2":188,"so2":196,"co":11495,"o3":140,"nh3":50}
        cats_r  = [POL_LABEL[p] for p in POLS if p in df.columns]
        fig3 = go.Figure()
        pal  = px.colors.qualitative.Set2
        # Pastel fill colours for radar (explicit rgba so Plotly 7 is happy)
        radar_fills = [
            "rgba(102,194,165,0.10)","rgba(252,141,98,0.10)",
            "rgba(141,160,203,0.10)","rgba(231,138,195,0.10)",
            "rgba(166,216,84,0.10)", "rgba(255,217,47,0.10)",
            "rgba(229,196,148,0.10)","rgba(179,179,179,0.10)",
        ]
        for i, row in lat_city.iterrows():
            vals = [min(200, round(row[p]/thresh.get(p,100)*100,1))
                    for p in POLS if p in df.columns]
            vals_closed = vals + [vals[0]]
            fig3.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=cats_r+[cats_r[0]],
                fill="toself", name=row["city"],
                line=dict(color=pal[i%len(pal)]),
                fillcolor=radar_fills[i%len(radar_fills)],
            ))
        fig3.update_layout(
            **{**THEME,"height":360},
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,200],
                                 tickfont=dict(color="#888",size=8),
                                 gridcolor="rgba(255,255,255,.08)"),
                angularaxis=dict(tickfont=dict(color="#ccc",size=10)),
            ),
            title="Pollutant Levels vs Moderate Threshold (%)",
        )
        st.plotly_chart(fig3, width="stretch")

# ════════════════════════════════════════════════════════════════
# SECTION 5  —  DISTRIBUTION & CONTRIBUTION
# ════════════════════════════════════════════════════════════════
def s_distribution(df):
    st.markdown('<div class="sec">🎨 AQI Category Distribution & Pollutant Contribution</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    # Donut
    with c1:
        cnt = df["aqi_category"].value_counts().reset_index()
        cnt.columns = ["cat","n"]
        # sort by cat order
        cnt["order"] = cnt["cat"].map({c:i for i,c in enumerate(CAT_ORDER)})
        cnt = cnt.sort_values("order")
        fig = go.Figure(go.Pie(
            labels=cnt["cat"], values=cnt["n"], hole=0.58,
            marker_colors=[CAT_COLOR.get(c,"#aaa") for c in cnt["cat"]],
            textfont=dict(size=10),
            hovertemplate="<b>%{label}</b><br>Days: %{value:,}<br>%{percent}<extra></extra>",
        ))
        fig.add_annotation(text="AQI<br>Days", x=0.5, y=0.5,
                           font=dict(size=12,color="#aaa"), showarrow=False)
        T(fig, 310); fig.update_layout(showlegend=True,
            legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, width="stretch")

    # Dominant pollutant
    with c2:
        dom = df["dominant_pollutant"].value_counts().reset_index()
        dom.columns = ["pol","n"]
        dom["pct"] = dom["n"]/dom["n"].sum()*100
        dom["label"] = dom["pol"].map(POL_LABEL).fillna(dom["pol"])
        fig2 = px.bar(dom, x="pct", y="label", orientation="h",
                      color="label",
                      color_discrete_map={POL_LABEL[p]:POL_COLOR[p] for p in POLS},
                      labels={"pct":"% Days Dominant","label":""},
                      text=dom["pct"].map("{:.1f}%".format))
        fig2.update_traces(textposition="outside", textfont_size=11)
        T(fig2, 310); fig2.update_layout(showlegend=False,
                          title="Dominant Pollutant (%)")
        st.plotly_chart(fig2, width="stretch")

    # Stacked bar per city
    with c3:
        cc = df.groupby(["city","aqi_category"]).size().reset_index(name="n")
        fig3 = px.bar(cc, x="city", y="n", color="aqi_category",
                      color_discrete_map=CAT_COLOR,
                      category_orders={"aqi_category":CAT_ORDER},
                      labels={"n":"Days","city":"","aqi_category":"Category"},
                      barmode="stack")
        T(fig3, 310)
        fig3.update_layout(title="Category Split by City",
                           legend=dict(font=dict(size=9)),
                           xaxis_tickangle=-30)
        st.plotly_chart(fig3, width="stretch")

# ════════════════════════════════════════════════════════════════
# SECTION 6  —  HEATMAPS
# ════════════════════════════════════════════════════════════════
def s_heatmaps(df):
    st.markdown('<div class="sec">📆 Temporal Heatmaps</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    with c1:
        h1 = df.groupby(["year","month"])["aqi"].mean().reset_index()
        pv = h1.pivot(index="year", columns="month", values="aqi")
        pv.columns = [MONTHS[m-1] for m in pv.columns]
        fig = go.Figure(go.Heatmap(
            z=pv.values, x=list(pv.columns),
            y=[str(y) for y in pv.index],
            colorscale=[[0,"#00e676"],[0.2,"#ffee58"],[0.4,"#ffa726"],
                        [0.6,"#ef5350"],[0.8,"#ab47bc"],[1,"#7e0023"]],
            colorbar=dict(title="AQI",tickfont=dict(color="#aaa"),
                          title_font=dict(color="#aaa")),
            hovertemplate="Year:%{y}  Month:%{x}<br>Avg AQI:%{z:.1f}<extra></extra>",
        ))
        T(fig, 230); fig.update_layout(title="Year × Month AQI Heatmap")
        st.plotly_chart(fig, width="stretch")

    with c2:
        DOWS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        h2 = df.groupby(["dow","month"])["aqi"].mean().reset_index()
        h2["dow"] = pd.Categorical(h2["dow"], categories=DOWS, ordered=True)
        h2 = h2.sort_values("dow")
        pv2 = h2.pivot(index="dow", columns="month", values="aqi")
        pv2.columns = [MONTHS[m-1] for m in pv2.columns]
        fig2 = go.Figure(go.Heatmap(
            z=pv2.values, x=list(pv2.columns), y=list(pv2.index),
            colorscale=[[0,"#00e676"],[0.2,"#ffee58"],[0.4,"#ffa726"],
                        [0.6,"#ef5350"],[0.8,"#ab47bc"],[1,"#7e0023"]],
            colorbar=dict(title="AQI",tickfont=dict(color="#aaa"),
                          title_font=dict(color="#aaa")),
            hovertemplate="Day:%{y}  Month:%{x}<br>Avg AQI:%{z:.1f}<extra></extra>",
        ))
        T(fig2, 230); fig2.update_layout(title="Day-of-Week × Month AQI")
        st.plotly_chart(fig2, width="stretch")

# ════════════════════════════════════════════════════════════════
# SECTION 7  —  WEATHER IMPACT
# ════════════════════════════════════════════════════════════════
def s_weather(df):
    st.markdown('<div class="sec">🌤️ Weather Impact on AQI</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    smp = df.dropna(subset=["aqi","temperature","humidity"]).sample(
        min(2000, len(df)), random_state=42)

    with c1:
        fig = px.scatter(smp, x="temperature", y="humidity",
                         color="aqi", size=np.clip(smp["aqi"]/4, 3, 22),
                         color_continuous_scale=[
                             [0,"#00e676"],[0.2,"#ffee58"],
                             [0.5,"#ffa726"],[0.7,"#ef5350"],
                             [0.9,"#ab47bc"],[1,"#7e0023"]],
                         range_color=[0,350],
                         hover_data={"station_name":True,"city":True,
                                     "aqi":":.0f","temperature":":.1f",
                                     "humidity":":.0f"},
                         labels={"temperature":"Temperature (°C)",
                                 "humidity":"Humidity (%)"})
        T(fig, 350); fig.update_layout(title="Temp vs Humidity  (colour = AQI)")
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig2 = make_subplots(rows=2, cols=2,
            subplot_titles=["Wind Speed vs AQI","Precipitation vs AQI",
                            "Pressure vs AQI","Humidity vs AQI"],
            vertical_spacing=0.18, horizontal_spacing=0.12)
        pairs = [("wind_speed",1,1),("precipitation",1,2),
                 ("pressure",2,1),("humidity",2,2)]
        s2 = df.dropna(subset=["aqi"]).sample(min(800,len(df)), random_state=7)
        for wx,r,c in pairs:
            if wx not in s2.columns: continue
            fig2.add_trace(go.Scatter(
                x=s2[wx], y=s2["aqi"], mode="markers",
                marker=dict(color=s2["aqi"],
                            colorscale=[[0,"#00e676"],[0.5,"#ffa726"],[1,"#b71c1c"]],
                            size=3, opacity=0.55),
                name=wx, showlegend=False), row=r, col=c)
        fig2.update_layout(**{**THEME,"height":350},
                           title="Weather Variables vs AQI")
        fig2.update_annotations(font=dict(color="#aaa",size=10))
        st.plotly_chart(fig2, width="stretch")

# ════════════════════════════════════════════════════════════════
# SECTION 8  —  YEAR-OVER-YEAR
# ════════════════════════════════════════════════════════════════
def s_yoy(df):
    st.markdown('<div class="sec">📊 Year-over-Year Comparison</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        yoy = df.groupby(["year","city"])["aqi"].mean().reset_index()
        fig = px.bar(yoy, x="city", y="aqi", color="year",
                     barmode="group",
                     color_discrete_sequence=["#42a5f5","#66bb6a","#ffa726"],
                     labels={"aqi":"Avg AQI","city":""},
                     text=yoy["aqi"].map("{:.0f}".format))
        T(fig, 320); fig.update_traces(textposition="outside")
        fig.update_layout(title="Avg AQI per City by Year",
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")

    with c2:
        # Box plot per year
        fig2 = px.box(df, x="year", y="aqi", color="year",
                      color_discrete_sequence=["#42a5f5","#66bb6a","#ffa726"],
                      labels={"aqi":"AQI","year":"Year"},
                      points=False)
        T(fig2, 320)
        fig2.update_layout(title="AQI Distribution per Year", showlegend=False)
        st.plotly_chart(fig2, width="stretch")

# ════════════════════════════════════════════════════════════════
# SECTION 9  —  CORRELATION MATRIX
# ════════════════════════════════════════════════════════════════
def s_correlation(df):
    st.markdown('<div class="sec">🔗 Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = [c for c in ["aqi","pm25","pm10","no2","so2","co","o3","nh3",
                             "temperature","humidity","wind_speed","pressure"]
                if c in df.columns]
    corr = df[num_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9, color="#fff"),
        colorbar=dict(title="r", tickfont=dict(color="#aaa"),
                      title_font=dict(color="#aaa")),
        hovertemplate="%{x} × %{y}<br>r = %{z:.2f}<extra></extra>",
    ))
    T(fig, 420)
    fig.update_layout(title="Pearson Correlation — Pollutants + Weather + AQI",
                      xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
                      yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(fig, width="stretch")

# ════════════════════════════════════════════════════════════════
# SECTION 10  —  STATION LEADERBOARD
# ════════════════════════════════════════════════════════════════
def s_leaderboard(df):
    st.markdown('<div class="sec">🏆 Station Leaderboard</div>', unsafe_allow_html=True)
    latest = df.sort_values("date").groupby("station_id").last().reset_index()
    avg    = df.groupby("station_id")["aqi"].mean().reset_index()
    avg.columns = ["station_id","avg_aqi_period"]
    tbl = latest.merge(avg, on="station_id")
    tbl["Rank"] = tbl["avg_aqi_period"].rank(ascending=False).astype(int)
    tbl = tbl.sort_values("Rank")
    show = tbl[["Rank","city","station_name","aqi","avg_aqi_period",
                "pm25","pm10","no2","temperature","humidity"]].copy()
    show.columns = ["Rank","City","Station","Latest AQI","Avg AQI",
                    "PM₂.₅","PM₁₀","NO₂","Temp°C","Humidity%"]

    def col_aqi(v):
        try: return f"color:{aqi_col(float(v))};font-weight:700"
        except: return ""

    styled = (show.style
        .map(col_aqi, subset=["Latest AQI","Avg AQI"])
        .format({"Latest AQI":"{:.0f}","Avg AQI":"{:.1f}",
                 "PM₂.₅":"{:.1f}","PM₁₀":"{:.1f}","NO₂":"{:.1f}",
                 "Temp°C":"{:.1f}","Humidity%":"{:.0f}"})
        .set_properties(**{"background-color":"rgba(255,255,255,.03)",
                           "color":"#dce1f0",
                           "border":"1px solid rgba(255,255,255,.06)"})
        .set_table_styles([{"selector":"th",
            "props":[("background-color","rgba(255,255,255,.08)"),
                     ("color","#fff"),("font-size","11px")]}])
    )
    st.dataframe(styled, width="stretch", height=340)

# ════════════════════════════════════════════════════════════════
# SECTION 11  —  AQI PREDICTION PANEL
# ════════════════════════════════════════════════════════════════
def s_prediction(df):
    st.markdown('<div class="sec">🤖 AQI Prediction (US-EPA Formula)</div>',
                unsafe_allow_html=True)

    # Quick ML model trained on the dataset
    c_form, c_out = st.columns([1,1])

    with c_form:
        st.markdown("##### 🎛️ Input Conditions")
        with st.form("pf"):
            ca, cb = st.columns(2)
            with ca:
                pm25 = st.slider("PM₂.₅ µg/m³", 0.0, 500.0, 85.0, 1.0)
                pm10 = st.slider("PM₁₀ µg/m³",  0.0, 600.0, 130.0, 1.0)
                no2  = st.slider("NO₂ µg/m³",   0.0, 400.0, 40.0, 1.0)
                so2  = st.slider("SO₂ µg/m³",   0.0, 200.0, 12.0, 0.5)
            with cb:
                co   = st.slider("CO µg/m³",    0.0, 5000.0, 800.0, 50.0)
                o3   = st.slider("O₃ µg/m³",   0.0, 300.0, 45.0, 1.0)
                nh3  = st.slider("NH₃ µg/m³",  0.0, 100.0, 5.0, 0.5)
                temp = st.slider("Temperature °C", -5.0, 50.0, 28.0, 0.5)
                hum  = st.slider("Humidity %",   0.0, 100.0, 65.0, 1.0)
            go_btn = st.form_submit_button("🔮 Predict AQI", type="primary",
                                            width="stretch")

    with c_out:
        if go_btn:
            # US-EPA sub-AQI calculation
            bps = {
                "pm25":[(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
                        (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500,301,500)],
                "pm10":[(0,54,0,50),(55,154,51,100),(155,254,101,150),
                        (255,354,151,200),(355,424,201,300),(425,604,301,500)],
                "no2": [(0,100,0,50),(101,188,51,100),(189,677,101,150),
                        (678,1222,151,200),(1223,2349,201,300)],
                "so2": [(0,91,0,50),(92,196,51,100),(197,484,101,150),
                        (485,793,151,200),(794,1584,201,300)],
                "co":  [(0,5175,0,50),(5176,11495,51,100),(11496,13760,101,150),
                        (13761,17225,151,200),(17226,34450,201,300)],
                "o3":  [(0,107,0,50),(108,140,51,100),(141,170,101,150),
                        (171,210,151,200),(211,400,201,300)],
            }
            def sub(pol, c):
                for cl,ch,il,ih in bps.get(pol,[]):
                    if cl<=c<=ch: return round(((ih-il)/(ch-cl))*(c-cl)+il, 2)
                return min(round(c/4, 2), 500)

            subs = {
                "PM₂.₅":sub("pm25",pm25), "PM₁₀":sub("pm10",pm10),
                "NO₂":sub("no2",no2), "SO₂":sub("so2",so2),
                "CO":sub("co",co), "O₃":sub("o3",o3),
            }
            dom_label = max(subs, key=subs.get)
            aqi_val = subs[dom_label]
            cat = aqi_cat(aqi_val)
            col = aqi_col(aqi_val)

            # Result card
            st.markdown(f"""
            <div class="card" style="text-align:center;border-left:6px solid {col}">
              <div style="font-size:.75rem;letter-spacing:.12em;opacity:.55">PREDICTED AQI</div>
              <div style="font-size:4rem;font-weight:900;color:{col};line-height:1.05">{aqi_val:.0f}</div>
              <div style="font-size:1.05rem;font-weight:700;color:{col}">{cat}</div>
              <div style="font-size:.82rem;opacity:.7;margin-top:6px">
                Dominant Pollutant: <b style="color:{POL_COLOR.get(dom_label.lower().replace('₂','2').replace('₁','1').replace('₃','3'),'#fff')}">{dom_label}</b>
              </div>
            </div>""", unsafe_allow_html=True)

            # Sub-AQI bar chart
            sub_df = pd.DataFrame({"Pollutant":list(subs.keys()),
                                   "Sub-AQI":list(subs.values())})
            fig = px.bar(sub_df, x="Pollutant", y="Sub-AQI",
                         color="Pollutant",
                         color_discrete_sequence=list(POL_COLOR.values()),
                         text=sub_df["Sub-AQI"].map("{:.0f}".format))
            fig.add_hline(y=100, line_dash="dot", line_color="#ffa726",
                          annotation_text="Unhealthy threshold",
                          annotation_font_color="#ffa726")
            T(fig, 260); fig.update_layout(showlegend=False,
                title="Sub-AQI Breakdown by Pollutant")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, width="stretch")

            # Health advisory
            advs = {
                "Good":("💚","Air quality is satisfactory — safe for all activities."),
                "Moderate":("💛","Acceptable quality. Sensitive individuals should limit prolonged outdoor activity."),
                "Unhealthy for Sensitive Groups":("🧡","Children, elderly & those with respiratory conditions — reduce outdoor activity."),
                "Unhealthy":("❤️","Everyone may experience health effects. Limit outdoor exertion."),
                "Very Unhealthy":("💜","Health alert! Everyone should avoid outdoor exertion."),
                "Hazardous":("🚨","Emergency conditions — stay indoors, keep windows closed."),
            }
            ico, msg = advs.get(cat, ("ℹ️","Monitor conditions carefully."))
            st.info(f"{ico} **Health Advisory:** {msg}")

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:48px 20px">
              <div style="font-size:3rem">🔮</div>
              <div style="opacity:.6;margin-top:12px;font-size:.95rem">
                Use the sliders to set current<br>pollution levels, then click<br>
                <b>Predict AQI</b>
              </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# SECTION 12  —  ANALYTICS SUMMARY
# ════════════════════════════════════════════════════════════════
def s_summary(df):
    st.markdown('<div class="sec">📋 Analytics Summary</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🚨 Top 5 Worst AQI Days**")
        worst = (df.groupby(["date","city"])["aqi"].mean()
                   .reset_index().nlargest(5,"aqi"))
        for _,r in worst.iterrows():
            col = aqi_col(r["aqi"])
            st.markdown(
                f'<div class="card" style="padding:8px 14px;margin:3px 0;'
                f'border-left:4px solid {col}">'
                f'<span style="color:{col};font-weight:800">{r["aqi"]:.0f}</span> — '
                f'{r["city"]} &nbsp;<span style="opacity:.55;font-size:.8rem">'
                f'{r["date"].strftime("%Y-%m-%d")}</span></div>',
                unsafe_allow_html=True)

    with c2:
        st.markdown("**✅ Top 5 Best AQI Days**")
        best = (df.groupby(["date","city"])["aqi"].mean()
                  .reset_index().nsmallest(5,"aqi"))
        for _,r in best.iterrows():
            col = aqi_col(r["aqi"])
            st.markdown(
                f'<div class="card" style="padding:8px 14px;margin:3px 0;'
                f'border-left:4px solid {col}">'
                f'<span style="color:{col};font-weight:800">{r["aqi"]:.0f}</span> — '
                f'{r["city"]} &nbsp;<span style="opacity:.55;font-size:.8rem">'
                f'{r["date"].strftime("%Y-%m-%d")}</span></div>',
                unsafe_allow_html=True)

    with c3:
        st.markdown("**📊 Pollutant Statistics (all data)**")
        avail_p = [p for p in POLS if p in df.columns]
        stats = df[avail_p].agg(["mean","min","max"]).T.round(2)
        stats.index = [POL_LABEL[p] for p in stats.index]
        st.dataframe(
            stats.style.format("{:.2f}"),
            width="stretch", height=240)

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    with st.spinner("⏳ Loading dataset …"):
        df_full = load()

    f = sidebar(df_full)
    df = filt(df_full, f)

    # ── Header ───────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex;align-items:center;gap:16px;padding:4px 0 14px'>
      <div style='font-size:2.8rem'>🌬️</div>
      <div>
        <div style='font-size:1.65rem;font-weight:900;letter-spacing:.03em'>
          India Air Quality Index Dashboard
        </div>
        <div style='font-size:.82rem;opacity:.50'>
          20 monitoring stations &nbsp;·&nbsp; 11 cities &nbsp;·&nbsp;
          2022 – 2024 &nbsp;·&nbsp; US-EPA AQI Standard
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if df.empty:
        st.warning("⚠️ No data for selected filters. Try widening your selection.")
        return

    # Data info bar
    ca, cb, cc, cd = st.columns(4)
    for col_st, txt in [
        (ca, f"📅 **{df['date'].nunique():,}** days selected"),
        (cb, f"📍 **{df['station_id'].nunique()}** stations"),
        (cc, f"🏙️ **{df['city'].nunique()}** cities"),
        (cd, f"📊 **{len(df):,}** records"),
    ]:
        col_st.markdown(f'<div class="card" style="padding:8px 14px">{txt}</div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    s_kpis(df);         st.markdown("<br>", unsafe_allow_html=True)
    s_gauge_map(df);    st.markdown("<br>", unsafe_allow_html=True)
    s_trend(df);        st.markdown("<br>", unsafe_allow_html=True)
    s_pollutants(df, f["pol"]); st.markdown("<br>", unsafe_allow_html=True)
    s_distribution(df); st.markdown("<br>", unsafe_allow_html=True)
    s_heatmaps(df);     st.markdown("<br>", unsafe_allow_html=True)
    s_weather(df);      st.markdown("<br>", unsafe_allow_html=True)
    s_yoy(df);          st.markdown("<br>", unsafe_allow_html=True)
    s_correlation(df);  st.markdown("<br>", unsafe_allow_html=True)
    s_leaderboard(df);  st.markdown("<br>", unsafe_allow_html=True)
    s_prediction(df);   st.markdown("<br>", unsafe_allow_html=True)
    s_summary(df)

    st.markdown("""
    <hr style='border-color:rgba(255,255,255,.06);margin:28px 0 8px'>
    <div style='text-align:center;font-size:.70rem;opacity:.35;padding-bottom:12px'>
      AQI calculated using US-EPA standard &nbsp;·&nbsp;
      Dataset: india_aqi_dataset.csv (21,920 rows) &nbsp;·&nbsp;
      Built with Streamlit + Plotly
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

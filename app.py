"""
Students' Social Network Profile — Clustering App
Applied Data Science, ML & AI | E&ICT Academy, IIT Guwahati
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage as sp_linkage
import time

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Students' SNS Profile Clustering",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS + Background ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 40%, #112240 70%, #0a192f 100%);
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            radial-gradient(circle at 20% 20%, rgba(46,84,150,0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(31,56,100,0.2)  0%, transparent 50%),
            radial-gradient(circle at 60% 30%, rgba(14,165,233,0.05) 0%, transparent 40%);
        pointer-events: none; z-index: 0;
    }
    [data-testid="stHeader"]         { background: transparent !important; }
    [data-testid="collapsedControl"] { display: none; }
    [data-testid="stDataFrame"]      { overflow-x: auto !important; }
    .block-container {
        padding: 1rem 1.5rem 3rem 1.5rem !important;
        max-width: 100% !important; position: relative; z-index: 1;
    }
    .main-header {
        background: linear-gradient(135deg, rgba(31,56,100,0.95) 0%, rgba(46,84,150,0.95) 100%);
        padding: 1.8rem 2rem; border-radius: 14px; margin-bottom: 1.2rem; text-align: center;
        border: 1px solid rgba(100,150,220,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4); backdrop-filter: blur(10px);
    }
    .main-header h1     { color:#ffffff; font-size:clamp(1.3rem,4vw,2rem); margin:0 0 0.4rem 0; }
    .main-header .course{ color:#ffffff; font-size:clamp(0.85rem,2.5vw,1.05rem); font-weight:700; margin:0 0 0.15rem 0; }
    .main-header .inst  { color:#BDD7EE; font-size:clamp(0.78rem,2.2vw,0.92rem); margin:0 0 0.15rem 0; }
    .main-header .algo  { color:#93B8D8; font-size:clamp(0.7rem,2vw,0.82rem); margin:0; }
    .sec {
        background: rgba(46,84,150,0.25); border-left: 4px solid #4A90D9;
        padding: 0.45rem 0.9rem; border-radius: 5px; font-weight: bold;
        color: #BDD7EE; margin: 1rem 0 0.6rem 0;
        font-size: clamp(0.8rem,2.5vw,0.95rem); backdrop-filter: blur(4px);
    }
    .box-green {
        background: rgba(76,175,80,0.12); border-left: 4px solid #4CAF50;
        padding: 0.7rem 1rem; border-radius: 6px; color: #A5D6A7;
        margin: 0.5rem 0; font-size: clamp(0.75rem,2.2vw,0.88rem);
    }
    .box-blue {
        background: rgba(21,101,192,0.15); border-left: 4px solid #42A5F5;
        padding: 0.7rem 1rem; border-radius: 6px; color: #90CAF9;
        margin: 0.5rem 0; font-size: clamp(0.75rem,2.2vw,0.88rem);
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(100,150,220,0.2);
        border-radius: 10px; padding: 0.6rem 0.8rem; backdrop-filter: blur(4px);
    }
    [data-testid="stMetricLabel"] { color: #93B8D8 !important; }
    [data-testid="stMetricValue"] { color: #ffffff  !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; overflow-x: auto; flex-wrap: nowrap;
        background: rgba(255,255,255,0.03); border-radius: 8px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(46,84,150,0.2); border-radius: 6px; padding: 8px 16px;
        font-weight: 600; color: #93B8D8; white-space: nowrap;
        font-size: clamp(0.72rem,2.5vw,0.88rem); border: 1px solid rgba(100,150,220,0.15);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(46,84,150,0.8) !important; color: white !important;
        border-color: rgba(100,150,220,0.4) !important;
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(100,150,220,0.2) !important;
        border-radius: 10px !important;
    }
    label { color: #BDD7EE !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Students' Social Network Profile Clustering</h1>
    <p class="course">Applied Data Science, ML & AI</p>
    <p class="inst">E&ICT Academy, IIT Guwahati</p>
    <p class="algo">KMeans &nbsp;·&nbsp; Hierarchical &nbsp;·&nbsp; DBSCAN</p>
</div>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
KEYWORD_COLS = [
    'basketball','football','soccer','softball','volleyball','swimming',
    'cheerleading','baseball','tennis','sports','cute','sex','sexy','hot',
    'kissed','dance','band','marching','music','rock','god','church','jesus',
    'bible','hair','dress','blonde','mall','shopping','clothes','hollister',
    'abercrombie','die','death','drunk','drugs'
]
THEMES = {
    'Sports':     ['basketball','football','soccer','softball','volleyball',
                   'swimming','cheerleading','baseball','tennis','sports'],
    'Music/Band': ['band','marching','music','rock','dance'],
    'Religion':   ['god','church','jesus','bible'],
    'Appearance': ['cute','hair','dress','blonde','mall','shopping',
                   'clothes','hollister','abercrombie'],
    'Romance':    ['sex','sexy','hot','kissed'],
    'Risky':      ['die','death','drunk','drugs']
}
THEME_COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860']

# ── File Upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📂 Upload **Clustering_Marketing.csv**", type=["csv"])

if uploaded is None:
    st.markdown('<div class="box-blue">📁 Upload <b>Clustering_Marketing.csv</b> above to begin.</div>',
                unsafe_allow_html=True)
    st.stop()

# ── Model Settings ─────────────────────────────────────────────────────────────
with st.expander("⚙️ Model Settings", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🔵 KMeans**")
        k_max   = st.selectbox("Max K to evaluate",
                               options=[5, 8, 10, 12, 15], index=2)
        pca_var = st.selectbox("PCA variance threshold",
                               options=[85, 90, 95], index=1,
                               format_func=lambda x: f"{x}%")
    with c2:
        st.markdown("**📊 Evaluation**")
        samp_size = st.selectbox("Silhouette sample size",
                                  options=[1000, 2000, 3000, 5000], index=2)
        n_init    = st.selectbox("KMeans n_init runs",
                                  options=[5, 10, 20], index=1)
    with c3:
        st.markdown("**🔵 DBSCAN**")
        eps_pct  = st.selectbox("Eps percentile",
                                 options=[80, 85, 90, 95], index=2,
                                 format_func=lambda x: f"{x}th percentile")
        min_samp = st.number_input("Min samples",
                                    min_value=3, max_value=20, value=5, step=1)

# ── Load & Silent Preprocessing ────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)

    # Replace string "None" with actual NaN
    df = df.replace('None', np.nan)

    # Fix dtypes & missing values
    df['gender'].fillna(df['gender'].mode()[0], inplace=True)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age'] = df['age'].where(df['age'].between(13, 20))
    df['age'].fillna(df['age'].median(), inplace=True)

    # Log1p transform high-skew keyword columns
    numeric_cols   = ['age', 'NumberOffriends'] + KEYWORD_COLS
    skew_vals      = df[numeric_cols].skew()
    high_skew_cols = skew_vals[skew_vals.abs() > 1.0].index.tolist()
    df_t           = df[numeric_cols].copy()
    df_t[high_skew_cols] = np.log1p(df[high_skew_cols])

    # Winsorize outliers
    for col in numeric_cols:
        Q1  = df_t[col].quantile(0.25)
        Q3  = df_t[col].quantile(0.75)
        IQR = Q3 - Q1
        df_t[col] = df_t[col].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)

    # Encode gender + add gradyear back
    le = LabelEncoder()
    df_t['gender_enc'] = le.fit_transform(df['gender'])
    df_t['gradyear']   = df['gradyear'].values

    # Build feature matrix + safety fillna
    features = ['gradyear', 'age', 'gender_enc', 'NumberOffriends'] + KEYWORD_COLS
    X        = df_t[features].copy()
    X        = X.fillna(X.median())

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled

with st.spinner("Loading and preprocessing data..."):
    df, X_scaled_arr = load_and_preprocess(uploaded)

# Add theme columns
for theme, cols in THEMES.items():
    df[f'theme_{theme}'] = df[cols].mean(axis=1)

st.markdown(f"""
<div class="box-green">
✅ <b>Data loaded & preprocessed</b> — {len(df):,} records · {X_scaled_arr.shape[1]} features ·
Missing values filled · log1p applied · Winsorized · StandardScaled
</div>""", unsafe_allow_html=True)

# ── Plot Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1b2a',
    'axes.facecolor':   '#112240',
    'axes.edgecolor':   '#2E5496',
    'axes.labelcolor':  '#BDD7EE',
    'xtick.color':      '#93B8D8',
    'ytick.color':      '#93B8D8',
    'text.color':       '#BDD7EE',
    'grid.color':       '#1F3864',
    'grid.alpha':       0.4,
})

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Data Overview",
    "🤖 Clustering Results",
    "🏆 Method Comparison"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec">📋 Dataset Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records",          f"{df.shape[0]:,}")
    c2.metric("Features",         df.shape[1])
    c3.metric("Keyword Features", 36)
    c4.metric("Grad Years",       df['gradyear'].nunique())

    st.markdown("**First 10 rows**")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("**Descriptive Statistics**")
    st.dataframe(df.describe().round(3), use_container_width=True)

    st.markdown('<div class="sec">🎯 Keyword Themes</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    items  = list(THEMES.items())
    with c1:
        for theme, cols in items[:3]:
            st.markdown(f"- **{theme}**: {', '.join(cols)}")
    with c2:
        for theme, cols in items[3:]:
            st.markdown(f"- **{theme}**: {', '.join(cols)}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTERING RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    @st.cache_data
    def run_pca(arr, vt):
        pf = PCA().fit(arr)
        cv = np.cumsum(pf.explained_variance_ratio_)
        nc = int(np.argmax(cv >= vt/100) + 1)
        Xp = PCA(n_components=nc, random_state=42).fit_transform(arr)
        p2 = PCA(n_components=2,  random_state=42)
        X2 = p2.fit_transform(arr)
        return Xp, X2, nc, cv, p2.explained_variance_ratio_

    X_pca, X_2d, n_comp, cumvar, ev2d = run_pca(X_scaled_arr, pca_var)

    @st.cache_data
    def find_best_k(arr, kmax, ss, ni):
        Kr, ins, sils = list(range(2, kmax+1)), [], []
        for k in Kr:
            km  = KMeans(n_clusters=k, random_state=42, n_init=ni)
            lbl = km.fit_predict(arr)
            ins.append(km.inertia_)
            sils.append(silhouette_score(arr, lbl, sample_size=ss, random_state=42))
        return Kr, ins, sils

    with st.spinner("Finding optimal K..."):
        K_range, inertias, sil_scores = find_best_k(X_pca, k_max, samp_size, n_init)

    best_k    = K_range[int(np.argmax(sil_scores))]
    kmeans    = KMeans(n_clusters=best_k, random_state=42, n_init=n_init)
    km_labels = kmeans.fit_predict(X_pca)
    df['Cluster'] = km_labels

    # Metrics
    st.markdown('<div class="sec">📍 Optimal K Selection</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Best K",           best_k)
    c2.metric("Silhouette Score", f"{max(sil_scores):.4f}")
    c3.metric("PCA Components",   f"{n_comp}  ({pca_var}% var)")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(K_range, inertias, marker='o', color='#4A90D9', linewidth=2, markersize=5)
        ax.set_title('Elbow — Inertia vs K', fontweight='bold', fontsize=9)
        ax.set_xlabel('K'); ax.set_ylabel('Inertia'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(K_range, sil_scores, marker='o', color='#55A868', linewidth=2, markersize=5)
        ax.axvline(x=best_k, color='#FF6B6B', linestyle='--', label=f'Best k={best_k}')
        ax.set_title('Silhouette Score vs K', fontweight='bold', fontsize=9)
        ax.set_xlabel('K'); ax.set_ylabel('Silhouette')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # 2D Scatter
    st.markdown('<div class="sec">🗺️ Cluster Visualization — 2D PCA</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in range(best_k):
        mask = km_labels == c
        ax.scatter(X_2d[mask,0], X_2d[mask,1],
                   label=f'C{c} (n={mask.sum()})',
                   color=plt.cm.tab10.colors[c], alpha=0.6, s=10)
    ax.set_title('Student Clusters — 2D PCA Projection', fontweight='bold')
    ax.set_xlabel(f'PC1 ({ev2d[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({ev2d[1]*100:.1f}% var)')
    ax.legend(markerscale=2, fontsize=8, ncol=2,
              facecolor='#0d1b2a', edgecolor='#2E5496', labelcolor='#BDD7EE')
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Cluster sizes
    st.markdown('<div class="sec">📊 Cluster Distribution</div>', unsafe_allow_html=True)
    dist = df['Cluster'].value_counts().sort_index().reset_index()
    dist.columns = ['Cluster','Count']
    dist['%'] = (dist['Count']/len(df)*100).round(2)
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(dist, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3))
        dist.plot(kind='bar', x='Cluster', y='Count', ax=ax,
                  color=list(plt.cm.tab10.colors[:best_k]),
                  edgecolor='#0d1b2a', legend=False)
        ax.set_title('Cluster Sizes', fontweight='bold', fontsize=9)
        ax.set_xlabel('Cluster'); ax.set_ylabel('Count')
        ax.grid(True, alpha=0.2, axis='y')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Interest heatmap
    st.markdown('<div class="sec">🔥 Interest Profile Heatmap</div>', unsafe_allow_html=True)
    cluster_profile = df.groupby('Cluster')[KEYWORD_COLS].mean()
    fig, ax = plt.subplots(figsize=(12, best_k+4))
    sns.heatmap(cluster_profile.T, annot=True, fmt='.2f', cmap='YlOrRd',
                linewidths=0.4, ax=ax,
                cbar_kws={'label':'Avg Mentions'},
                annot_kws={'size': 8}, linecolor='#0d1b2a')
    ax.set_title('Avg Keyword Mentions per Cluster', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Theme profile
    st.markdown('<div class="sec">🎨 Theme-Level Interest by Cluster</div>', unsafe_allow_html=True)
    theme_profile = df.groupby('Cluster')[[f'theme_{t}' for t in THEMES]].mean()
    theme_profile.columns = list(THEMES.keys())
    fig, ax = plt.subplots(figsize=(10, 4))
    theme_profile.T.plot(kind='bar', ax=ax, colormap='tab10',
                         edgecolor='#0d1b2a', width=0.7)
    ax.set_title('Theme Interest by Cluster', fontweight='bold')
    ax.set_xlabel('Theme'); ax.set_ylabel('Avg Mentions')
    plt.xticks(rotation=15, ha='right')
    ax.legend(title='Cluster', fontsize=8, bbox_to_anchor=(1.01,1),
              facecolor='#0d1b2a', edgecolor='#2E5496', labelcolor='#BDD7EE')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    st.dataframe(theme_profile.round(3), use_container_width=True)

    # Demographic profile
    st.markdown('<div class="sec">👥 Demographic Profile per Cluster</div>', unsafe_allow_html=True)
    demo = df.groupby('Cluster').agg(
        Count       = ('Cluster', 'count'),
        Avg_Age     = ('age', 'mean'),
        Avg_Friends = ('NumberOffriends', 'mean'),
        Pct_Female  = ('gender', lambda x: (x=='F').mean()*100)
    ).round(2)
    st.dataframe(demo, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        bp = ax.boxplot(
            [df[df['Cluster']==c]['age'].values for c in range(best_k)],
            patch_artist=True,
            labels=[f'C{c}' for c in range(best_k)]
        )
        for patch, color in zip(bp['boxes'], plt.cm.tab10.colors[:best_k]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_title('Age by Cluster', fontweight='bold')
        ax.set_xlabel('Cluster'); ax.set_ylabel('Age')
        ax.grid(True, alpha=0.2)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        df.groupby('Cluster')['gender'].value_counts(normalize=True).unstack().plot(
            kind='bar', ax=ax, colormap='Set2', edgecolor='#0d1b2a', width=0.6)
        ax.set_title('Gender Ratio by Cluster', fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(title='Gender', fontsize=8,
                  facecolor='#0d1b2a', edgecolor='#2E5496', labelcolor='#BDD7EE')
        ax.grid(True, alpha=0.2, axis='y')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Trend over graduation year
    st.markdown('<div class="sec">📅 Trend Analysis — Over Graduation Year</div>',
                unsafe_allow_html=True)
    theme_list = list(THEMES.keys())
    for row_themes in [theme_list[:3], theme_list[3:]]:
        cols = st.columns(3)
        for col_widget, theme in zip(cols, row_themes):
            fig, ax = plt.subplots(figsize=(5, 4))
            trend = df.groupby(['gradyear','Cluster'])[f'theme_{theme}'].mean().unstack()
            for i, col_name in enumerate(trend.columns):
                ax.plot(trend.index, trend[col_name], marker='o',
                        linewidth=1.5, color=plt.cm.tab10.colors[i % 10],
                        label=f'C{col_name}', markersize=4)
            ax.set_title(theme, fontweight='bold', fontsize=9)
            ax.set_xlabel('Grad Year', fontsize=8)
            ax.set_ylabel('Avg Mentions', fontsize=8)
            ax.legend(title='C', fontsize=7, ncol=2,
                      facecolor='#0d1b2a', edgecolor='#2E5496', labelcolor='#BDD7EE')
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            col_widget.pyplot(fig, use_container_width=True); plt.close()

    # Download
    st.markdown('<div class="sec">⬇️ Download Cluster Assignments</div>', unsafe_allow_html=True)
    out = df[['gradyear','gender','age','NumberOffriends','Cluster']].copy()
    st.download_button(
        "📥 Download CSV",
        out.to_csv(index=False).encode('utf-8'),
        "student_clusters.csv", "text/csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METHOD COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    t0      = time.time()
    km_sil  = silhouette_score(X_pca, km_labels, sample_size=samp_size, random_state=42)
    km_db   = davies_bouldin_score(X_pca, km_labels)
    km_ch   = calinski_harabasz_score(X_pca, km_labels)
    km_time = time.time() - t0

    # Hierarchical
    st.markdown('<div class="sec">🌿 Hierarchical Clustering</div>', unsafe_allow_html=True)

    @st.cache_data
    def run_hierarchical(arr, k, ss):
        res = {}
        for link in ['ward','complete','average']:
            t0  = time.time()
            lbl = AgglomerativeClustering(n_clusters=k, linkage=link).fit_predict(arr)
            res[link] = {
                'labels': lbl,
                'sil':    silhouette_score(arr, lbl, sample_size=ss, random_state=42),
                'db':     davies_bouldin_score(arr, lbl),
                'ch':     calinski_harabasz_score(arr, lbl),
                'time':   round(time.time()-t0, 2)
            }
        return res

    with st.spinner("Running Hierarchical..."):
        hier = run_hierarchical(X_pca, best_k, samp_size)

    link_df = pd.DataFrame({
        'Linkage':        list(hier.keys()),
        'Silhouette':     [v['sil']  for v in hier.values()],
        'Davies-Bouldin': [v['db']   for v in hier.values()],
        'Calinski-H':     [v['ch']   for v in hier.values()],
        'Time (s)':       [v['time'] for v in hier.values()]
    })
    st.dataframe(link_df.round(4), use_container_width=True)

    best_link = max(hier, key=lambda x: hier[x]['sil'])
    hc_labels = hier[best_link]['labels']
    hc_sil, hc_db, hc_ch, hc_time = (
        hier[best_link]['sil'], hier[best_link]['db'],
        hier[best_link]['ch'],  hier[best_link]['time']
    )
    st.success(f"Best linkage: **{best_link}**  |  Silhouette = {hc_sil:.4f}")

    # Dendrogram
    st.markdown('<div class="sec">🌳 Dendrogram (300-sample)</div>', unsafe_allow_html=True)
    sample_idx = np.random.RandomState(42).choice(len(X_pca), size=300, replace=False)
    linked     = sp_linkage(X_pca[sample_idx], method=best_link)
    fig, ax    = plt.subplots(figsize=(10, 5))
    dendrogram(linked, truncate_mode='lastp', p=30,
               leaf_rotation=90, leaf_font_size=8,
               show_contracted=True, ax=ax,
               color_threshold=linked[-(best_k)+1, 2])
    ax.axhline(y=linked[-(best_k)+1, 2], color='#FF6B6B',
               linestyle='--', label=f'Cut k={best_k}')
    ax.set_title(f'Dendrogram — {best_link} linkage', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # DBSCAN
    st.markdown('<div class="sec">🔵 DBSCAN</div>', unsafe_allow_html=True)

    @st.cache_data
    def dbscan_pipeline(arr, ep, ms, ss):
        nn   = NearestNeighbors(n_neighbors=5).fit(arr)
        d, _ = nn.kneighbors(arr)
        kd   = np.sort(d[:,4])[::-1]
        sug  = round(np.percentile(kd, ep), 3)
        rows = []
        for eps in [round(sug*f, 3) for f in [0.5, 0.75, 1.0, 1.25, 1.5]]:
            lbl     = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit_predict(arr)
            n_cl    = len(set(lbl)) - (1 if -1 in lbl else 0)
            n_noi   = (lbl == -1).sum()
            noi_pct = n_noi / len(lbl) * 100
            sil     = (silhouette_score(arr, lbl, sample_size=ss, random_state=42)
                       if n_cl >= 2 and noi_pct < 50 else np.nan)
            rows.append({'eps': eps, 'min_samples': ms, 'n_clusters': n_cl,
                         'noise_pts': n_noi, 'noise_pct': round(noi_pct, 2),
                         'silhouette': round(sil,4) if not np.isnan(sil) else np.nan})
        return pd.DataFrame(rows), sug, kd

    with st.spinner("Running DBSCAN..."):
        dbscan_df, sug_eps, k_dist = dbscan_pipeline(X_pca, eps_pct, min_samp, samp_size)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(k_dist, color='#4A90D9', linewidth=1.5)
        ax.axhline(y=sug_eps, color='#FF6B6B', linestyle='--', label=f'eps={sug_eps}')
        ax.set_title('K-Distance Plot (k=5)', fontweight='bold', fontsize=9)
        ax.set_xlabel('Points'); ax.set_ylabel('5-NN Distance')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    with c2:
        st.markdown(f"**Suggested eps = {sug_eps}**")
        st.dataframe(dbscan_df.round(4), use_container_width=True)

    valid = dbscan_df.dropna(subset=['silhouette'])
    if len(valid) > 0:
        best_row  = valid.sort_values('silhouette', ascending=False).iloc[0]
        best_eps  = best_row['eps']
        t0        = time.time()
        db_labels = DBSCAN(eps=best_eps, min_samples=min_samp, n_jobs=-1).fit_predict(X_pca)
        db_time   = time.time() - t0
        db_sil    = silhouette_score(X_pca, db_labels, sample_size=samp_size, random_state=42)
        db_db     = davies_bouldin_score(X_pca, db_labels)
        db_ch     = calinski_harabasz_score(X_pca, db_labels)
        db_n_cl   = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        db_noise  = int((db_labels == -1).sum())
        st.success(f"Best DBSCAN: eps={best_eps}, ms={min_samp} → "
                   f"{db_n_cl} clusters | {db_noise} noise pts | Sil={db_sil:.4f}")
    else:
        st.warning("No valid DBSCAN config. Adjust settings above.")
        db_labels = np.zeros(len(X_pca), dtype=int)
        db_sil, db_db, db_ch, db_time = 0.0, 0.0, 0.0, 0.0
        db_n_cl, db_noise, best_eps   = 0, 0, sug_eps

    # Side-by-side scatter
    st.markdown('<div class="sec">🖼️ Cluster Plots — All 3 Methods</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col_w, (labels, title) in zip(cols, [
        (km_labels, f'KMeans (k={best_k})\nSil={km_sil:.4f}'),
        (hc_labels, f'Hierarchical ({best_link})\nSil={hc_sil:.4f}'),
        (db_labels, f'DBSCAN (eps={best_eps})\nSil={db_sil:.4f}'),
    ]):
        fig, ax = plt.subplots(figsize=(5, 4))
        for i, lbl in enumerate(sorted(set(labels))):
            mask  = labels == lbl
            color = '#888888' if lbl == -1 else plt.cm.tab10(i % 10)
            ax.scatter(X_2d[mask,0], X_2d[mask,1], c=[color],
                       label='Noise' if lbl==-1 else f'C{lbl}',
                       alpha=0.5, s=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=8); ax.set_ylabel('PC2', fontsize=8)
        ax.legend(markerscale=2, fontsize=7,
                  facecolor='#0d1b2a', edgecolor='#2E5496', labelcolor='#BDD7EE')
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        col_w.pyplot(fig, use_container_width=True); plt.close()

    # Final comparison table
    st.markdown('<div class="sec">📊 Final Metric Comparison</div>', unsafe_allow_html=True)
    comparison = pd.DataFrame({
        'Method':           [f'KMeans (k={best_k})',
                             f'Hierarchical ({best_link})',
                             f'DBSCAN (eps={best_eps})'],
        'N Clusters':       [best_k,  best_k,  db_n_cl],
        'Silhouette ↑':     [round(km_sil,4), round(hc_sil,4), round(db_sil,4)],
        'Davies-Bouldin ↓': [round(km_db,4),  round(hc_db,4),  round(db_db,4)],
        'Calinski-H ↑':     [round(km_ch,2),  round(hc_ch,2),  round(db_ch,2)],
        'Train Time (s)':   [round(km_time,2), round(hc_time,2), round(db_time,2)],
        'Noise Pts':        [0, 0, db_noise]
    })
    st.dataframe(comparison, use_container_width=True)
    st.caption("↑ higher is better   ↓ lower is better")

    # Bar charts
    st.markdown('<div class="sec">📉 Metric Bar Charts</div>', unsafe_allow_html=True)
    methods_lbl = ['KMeans', f'Hierarchical\n({best_link})', 'DBSCAN']
    colors_m    = ['#4C72B0','#55A868','#C44E52']

    for vals, title in [
        ([km_sil, hc_sil, db_sil], 'Silhouette ↑'),
        ([km_db,  hc_db,  db_db],  'Davies-Bouldin ↓'),
        ([km_ch,  hc_ch,  db_ch],  'Calinski-Harabasz ↑'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.bar(methods_lbl, vals, color=colors_m,
                      edgecolor='#0d1b2a', width=0.4)
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, max(vals)*1.3)
        ax.grid(True, alpha=0.2, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(vals)*0.02,
                    f'{v:.4f}' if v < 1000 else f'{v:.0f}',
                    ha='center', fontsize=11, fontweight='bold', color='#BDD7EE')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Winner
    best_idx    = int(np.argmax([km_sil, hc_sil, db_sil]))
    best_method = comparison['Method'].iloc[best_idx]
    st.markdown(f"""
    <div class="box-green">
    ★ <b>Best method by Silhouette: {best_method}</b><br><br>
    <b>Metric Guide</b><br>
    • Silhouette &gt;0.5 = strong &nbsp;|&nbsp; 0.2–0.5 = moderate &nbsp;|&nbsp; &lt;0.2 = weak<br>
    • Davies-Bouldin: lower = more compact &amp; well-separated<br>
    • Calinski-Harabasz: higher = denser, better-defined clusters<br><br>
    <b>Algorithm Notes</b><br>
    • <b>KMeans</b> — fast, scalable, assumes spherical clusters<br>
    • <b>Hierarchical</b> — dendrogram, no k needed upfront, slower on large data<br>
    • <b>DBSCAN</b> — handles noise natively, arbitrary shapes, eps-sensitive
    </div>""", unsafe_allow_html=True)

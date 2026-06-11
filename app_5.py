import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io
import re
import networkx as nx
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
import shap
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Scopus Dashboard — Lesiones en Fútbol (ML)", layout="wide")

# ---------------------------
# Utilidades de carga y limpieza
# ---------------------------
REQUIRED_COLS = ['Authors', 'Title', 'Year', 'Abstract', 'Cited by', 'DOI', 'Author Keywords', 'Link']

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip().replace('"', '') if isinstance(c, str) else c)
    return df

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def clean_authors(authors_str):
    if pd.isna(authors_str):
        return []
    # Separadores comunes en Scopus export
    parts = re.split(r';|, and | and |,', str(authors_str))
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts

@st.cache_data(show_spinner=False)
def load_and_clean(path_or_buffer) -> pd.DataFrame:
    # path_or_buffer puede ser un file-like o URL
    df = pd.read_csv(path_or_buffer, dtype=str, encoding='utf-8', low_memory=False)
    df = normalize_columns(df)
    df = ensure_columns(df)
    # Convertir tipos
    df['Year'] = pd.to_numeric(df.get('Year', pd.Series()), errors='coerce').astype('Int64')
    df['Cited by'] = pd.to_numeric(df.get('Cited by', 0), errors='coerce').fillna(0).astype(int)
    # Limpiar strings
    for c in ['Title', 'Abstract', 'DOI', 'Author Keywords', 'Link', 'Authors']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({'nan': pd.NA})
    # Lista de autores
    df['Authors_list'] = df['Authors'].apply(clean_authors)
    # Normalizar keywords (lista)
    def split_keywords(k):
        if pd.isna(k): return []
        parts = re.split(r';|,', str(k))
        return [p.strip().lower() for p in parts if p.strip()]
    df['Keywords_list'] = df.get('Author Keywords', pd.Series()).apply(split_keywords)
    # Eliminar filas sin título
    df = df[~df['Title'].isna()]
    # Eliminar duplicados por DOI o título
    if 'DOI' in df.columns:
        df = df.drop_duplicates(subset=['DOI']).reset_index(drop=True)
    df = df.drop_duplicates(subset=['Title']).reset_index(drop=True)
    return df

# ---------------------------
# Funciones de visualización
# ---------------------------
def chart_publications_by_year(df: pd.DataFrame):
    counts = df.groupby('Year').size().reset_index(name='count').dropna()
    if counts.empty:
        return None
    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X('Year:O', sort='-y', title='Año'),
        y=alt.Y('count:Q', title='Publicaciones'),
        tooltip=['Year', 'count']
    ).properties(height=300)
    return chart

def chart_top_authors(df: pd.DataFrame, top_n=15):
    authors = df['Authors_list'].explode().dropna()
    if authors.empty:
        return None
    top = authors.value_counts().head(top_n).reset_index()
    top.columns = ['author', 'count']
    chart = alt.Chart(top).mark_bar().encode(
        x=alt.X('count:Q', title='Publicaciones'),
        y=alt.Y('author:N', sort='-x', title='Autor'),
        tooltip=['author', 'count']
    ).properties(height=350)
    return chart

def plot_citations_distribution(df: pd.DataFrame):
    df_plot = df.dropna(subset=['Year'])
    if df_plot.empty:
        return None
    fig, ax = plt.subplots(figsize=(10,4))
    sns.boxplot(x='Year', y='Cited by', data=df_plot, ax=ax)
    ax.set_xlabel("Año")
    ax.set_ylabel("Citas")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def wordcloud_abstracts(df: pd.DataFrame, max_words=150):
    text = " ".join(df['Abstract'].dropna().astype(str).tolist())
    if not text.strip():
        return None
    wc = WordCloud(width=900, height=300, background_color='white', max_words=max_words).generate(text)
    fig, ax = plt.subplots(figsize=(12,4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    return fig

def coauthorship_graph(df: pd.DataFrame, top_n_authors=30):
    authors = df['Authors_list'].explode().dropna()
    if authors.empty:
        return None
    # Seleccionar autores más frecuentes
    top_auth = authors.value_counts().head(top_n_authors).index.tolist()
    G = nx.Graph()
    for idx, row in df.iterrows():
        a_list = [a for a in row['Authors_list'] if a in top_auth]
        for i in range(len(a_list)):
            for j in range(i+1, len(a_list)):
                a, b = a_list[i], a_list[j]
                if G.has_edge(a, b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a, b, weight=1)
    if G.number_of_nodes() == 0:
        return None
    # Dibujar con matplotlib (layout pequeño)
    pos = nx.spring_layout(G, seed=42, k=0.5)
    fig, ax = plt.subplots(figsize=(8,6))
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue', ax=ax)
    nx.draw_networkx_edges(G, pos, width=[max(0.5, w*0.5) for w in weights], alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_axis_off()
    plt.tight_layout()
    return fig

def top_keywords_bar(df: pd.DataFrame, top_n=20):
    kw = df['Keywords_list'].explode().dropna()
    if kw.empty:
        return None
    top = kw.value_counts().head(top_n).reset_index()
    top.columns = ['keyword', 'count']
    chart = alt.Chart(top).mark_bar().encode(
        x=alt.X('count:Q', title='Frecuencia'),
        y=alt.Y('keyword:N', sort='-x', title='Keyword'),
        tooltip=['keyword', 'count']
    ).properties(height=350)
    return chart

# ---------------------------
# ML rápido (opcional)
# ---------------------------
def quick_ml_example(df: pd.DataFrame, target_keyword='injury'):
    """
    Construye un ejemplo simple: clasificar si el abstract menciona target_keyword.
    Esto es solo demostrativo: crea una etiqueta binaria y entrena RandomForest.
    """
    df_ml = df.copy()
    df_ml['abstract_text'] = df_ml['Abstract'].fillna('').astype(str).str.lower()
    df_ml['label'] = df_ml['abstract_text'].str.contains(target_keyword.lower()).astype(int)
    # Rechazar si muy pocos ejemplos
    if df_ml['label'].sum() < 3 or (len(df_ml) - df_ml['label'].sum()) < 3:
        return None, "No hay suficientes ejemplos positivos/negativos para entrenar un modelo robusto."
    # Vectorizar texto (bag-of-words pequeño)
    vect = CountVectorizer(max_features=200, stop_words='english')
    X = vect.fit_transform(df_ml['abstract_text']).toarray()
    y = df_ml['label'].values
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    clf.fit(X, y)
    # SHAP (explicador)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X) if hasattr(explainer, "shap_values") else explainer(X)
    feature_names = vect.get_feature_names_out()
    return {
        'clf': clf,
        'auc_mean': float(np.mean(scores)),
        'auc_std': float(np.std(scores)),
        'feature_names': feature_names,
        'shap_values': shap_values,
        'X': X
    }, None

# ---------------------------
# Interfaz Streamlit
# ---------------------------
st.title("Scopus Dashboard — Predicción de lesiones en fútbol (Machine Learning)")
st.markdown("Explora tu CSV exportado desde Scopus. El dashboard incluye filtros, visualizaciones y un ejemplo ML opcional.")

# Sidebar: carga y filtros
with st.sidebar:
    st.header("Carga de datos")
    uploaded = st.file_uploader("Sube tu CSV exportado desde Scopus", type=['csv'])
    github_url = st.text_input("O pega la URL raw de GitHub (opcional)")
    st.markdown("---")
    st.header("Filtros globales (aplican a todas las visualizaciones)")
    min_cites = st.number_input("Mínimo de citas", min_value=0, value=0, step=1)
    year_range = st.slider("Rango de años (si aplica)", 2000, 2030, (2018, 2026))
    kw_filter = st.text_input("Filtrar por palabra en abstract o keyword (opcional)")
    st.markdown("---")
    st.markdown("Opciones")
    show_ml = st.checkbox("Incluir ejemplo ML (RandomForest + SHAP)", value=False)
    st.markdown("Licencia sugerida: **MIT** (agregar en GitHub)")

# Cargar datos
data = None
try:
    if uploaded:
        data = load_and_clean(uploaded)
    elif github_url:
        data = load_and_clean(github_url)
except Exception as e:
    st.error(f"Error cargando o parseando el CSV: {e}")
    st.stop()

if data is None:
    st.info("Sube un CSV o pega la URL raw de GitHub para comenzar.")
    st.stop()

# Aplicar filtros básicos
df = data.copy()
# Año
if 'Year' in df.columns:
    df = df[df['Year'].notna()]
    df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
# Citas
df = df[df['Cited by'] >= int(min_cites)]
# Keyword/abstract filter
if kw_filter and kw_filter.strip():
    kw = kw_filter.strip().lower()
    df = df[df['Abstract'].fillna('').str.lower().str.contains(kw) | df['Author Keywords'].fillna('').str.lower().str.contains(kw)]

# Layout principal
col_main, col_side = st.columns([3,1])

with col_main:
    st.subheader("Visualizaciones principales")
    # Row 1: publicaciones por año + top autores
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown("**Publicaciones por año**")
        chart_year = chart_publications_by_year(df)
        if chart_year:
            st.altair_chart(chart_year, use_container_width=True)
        else:
            st.info("No hay datos por año para mostrar.")
    with c2:
        st.markdown("**Top autores (por número de publicaciones)**")
        chart_auth = chart_top_authors(df, top_n=15)
        if chart_auth:
            st.altair_chart(chart_auth, use_container_width=True)
        else:
            st.info("No hay autores para mostrar.")

    # Row 2: distribución de citas + top keywords
    c3, c4 = st.columns([2,1])
    with c3:
        st.markdown("**Distribución de citas por año**")
        fig_cit = plot_citations_distribution(df)
        if fig_cit:
            st.pyplot(fig_cit)
        else:
            st.info("No hay suficientes datos para la distribución de citas.")
    with c4:
        st.markdown("**Top keywords (author keywords)**")
        chart_kw = top_keywords_bar(df, top_n=15)
        if chart_kw:
            st.altair_chart(chart_kw, use_container_width=True)
        else:
            st.info("No hay keywords para mostrar.")

    # Row 3: wordcloud + coauthorship
    c5, c6 = st.columns([2,1])
    with c5:
        st.markdown("**Wordcloud de abstracts**")
        fig_wc = wordcloud_abstracts(df)
        if fig_wc:
            st.pyplot(fig_wc)
        else:
            st.info("No hay abstracts para generar la wordcloud.")
    with c6:
        st.markdown("**Grafo de coautoría (autores más frecuentes)**")
        fig_graph = coauthorship_graph(df, top_n_authors=25)
        if fig_graph:
            st.pyplot(fig_graph)
        else:
            st.info("No hay suficientes datos de coautoría para mostrar grafo.")

    st.markdown("---")
    st.subheader("Tabla interactiva y detalle")
    display_cols = ['Title', 'Authors', 'Year', 'Cited by', 'DOI']
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols].fillna(''), use_container_width=True, height=300)

    st.markdown("**Detalle de artículo**")
    titles = df['Title'].fillna('').tolist()
    sel_title = st.selectbox("Selecciona título para ver detalle", options=[""] + titles)
    if sel_title:
        row = df[df['Title'] == sel_title].iloc[0]
        st.markdown(f"**Título:** {row.get('Title','')}")
        st.markdown(f"**Autores:** {row.get('Authors','')}")
        st.markdown(f"**Año:** {row.get('Year','')}")
        st.markdown(f"**Citas:** {row.get('Cited by',0)}")
        st.markdown(f"**DOI:** {row.get('DOI','')}")
        link = row.get('Link','')
        if link and link != 'nan':
            st.markdown(f"**Link:** {link}")
        st.markdown("**Abstract:**")
        st.write(row.get('Abstract','No disponible'))

with col_side:
    st.subheader("Resumen rápido")
    st.metric("Artículos totales (filtrados)", len(df))
    st.metric("Citas totales (filtradas)", int(df['Cited by'].sum()))
    st.markdown("**Top 5 autores (filtrados)**")
    top5 = df['Authors_list'].explode().value_counts().head(5)
    for a, c in top5.items():
        st.write(f"- **{a}** — {c} publicaciones")
    st.markdown("---")
    st.subheader("Export / Repositorio")
    st.markdown("Sugerencias para GitHub:")
    st.write("- Añade `LICENSE` (MIT recomendado).")
    st.write("- Incluye `README.md` con instrucciones de ejecución.")
    st.write("- Sube `examples/scopus_clean.csv` con una muestra.")

# ---------------------------
# Ejemplo ML (opcional)
# ---------------------------
if show_ml:
    st.markdown("---")
    st.subheader("Ejemplo ML rápido: clasificar abstracts que mencionan 'injury' (demostrativo)")
    with st.spinner("Entrenando modelo de ejemplo..."):
        result, err = quick_ml_example(df, target_keyword='injury')
    if err:
        st.warning(err)
    else:
        st.success(f"Modelo entrenado. AUC (CV 5-fold): {result['auc_mean']:.3f} ± {result['auc_std']:.3f}")
        st.markdown("**Top features (por importancia del modelo)**")
        importances = result['clf'].feature_importances_
        feat_imp = sorted(zip(result['feature_names'], importances), key=lambda x: x[1], reverse=True)[:15]
        for f, imp in feat_imp:
            st.write(f"- {f}: {imp:.4f}")
        st.markdown("**SHAP summary (puede tardar en renderizar)**")
        try:
            shap.initjs()
            # Mostrar summary_plot en matplotlib y renderizar
            fig_shap = plt.figure(figsize=(8,4))
            shap.summary_plot(result['shap_values'][1] if isinstance(result['shap_values'], list) else result['shap_values'], 
                              features=result['X'], feature_names=result['feature_names'], show=False, plot_type="bar")
            st.pyplot(fig_shap)
        except Exception as e:
            st.info(f"SHAP no pudo generarse en este entorno: {e}")

# ---------------------------
# Footer / ayuda
# ---------------------------
st.markdown("---")
st.markdown("**Notas:**")
st.markdown("- El CSV esperado debe contener columnas como: **Authors, Title, Year, Abstract, Cited by, DOI, Author Keywords, Link**.")
st.markdown("- No subas datos sensibles a repositorios públicos.")
st.markdown("- Si tu CSV es muy grande, considera preprocesarlo y subir una muestra para el dashboard.")


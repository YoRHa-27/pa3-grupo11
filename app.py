import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import io
import json

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML & Rendimiento Deportivo | Grupo 11",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  DATASET EMBEBIDO (Scopus Export)
# ─────────────────────────────────────────────
EMBEDDED_DATA = [{"Authors":"Xia X.; Chen Q.; Wang Z.","Title":"Deep reinforcement learning-driven personalized training load control algorithm for competitive sports performance optimization","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"Traditional training load management methods in competitive sports rely heavily on subjective assessments and standardized protocols, often failing to account for individual physiological variations and dynamic adaptation responses. This research proposes a deep reinforcement learning (DRL) framework for personalized training load optimization that integrates real-time physiological monitoring, individual athlete characteristics, and adaptive decision-making algorithms. The proposed system employs a hybrid neural network architecture combining multilayer perceptrons and convolutional neural networks to process heterogeneous physiological data and generate training prescriptions. Empirical validation across multiple sports disciplines including track and field, swimming, and ball sports, with ethical approval (IRB: DU-IRB-2023-001), shows performance improvements averaging 12.3% (95% CI: 10.1–14.5%, p < 0.001) compared to traditional periodization-based methods as measured by sport-specific performance tests using independent samples t-tests, with injury rate reductions of 43% and training efficiency enhancements ranging from 1.15 to 1.42 times conventional approaches.","Author Keywords":"Athlete monitoring; Competitive sports; Deep reinforcement learning; Load management; Performance optimization; Training individualization","Document Type":"Article"},{"Authors":"Masood Z.; Luke D.; Kenny R.; Bondi D.; Clansey A.; Wu L.C.","Title":"Head impact biomechanics across men's and women's contact sports: a comparative and clustering analysis","Year":2026,"Source":"Scientific Reports","Cited by":1,"Abstract":"Sports head impacts have been associated with both acute and long-term brain trauma. While wearable sensors provide valuable biomechanics insight, most studies focus on single sports, and the variability in sensor methodologies limits cross-sport comparisons. Our objectives were to conduct a multisport comparison and clustering of head impact biomechanics features implicated in brain injury risk. We uniformly processed a multisport dataset gathered using instrumented mouthguards containing direct head impacts in men's football, men's hockey, women's rugby, and women's soccer.","Author Keywords":"Head impact biomechanics; Injury biomechanics; Instrumented mouthguard; Signal processing; Unsupervised machine learning","Document Type":"Article"},{"Authors":"Zhang G.; Xu B.; Wang J.","Title":"Graph structure modeling for optimizing the relationship between training load and physical health in volleyball players using Graph of Thought","Year":2026,"Source":"Systems and Soft Computing","Cited by":0,"Abstract":"To address the challenge of aligning training-load regulation with individual physiological feedback in volleyball players, this study proposes a personalized optimization framework based on Graph of Thoughts (GOT). An individualized knowledge graph is first constructed as a domain prior, and real-time monitoring data are then used to drive a four-stage GOT reasoning process comprising perception, analysis, prediction, and optimization.","Author Keywords":"Graph of Thoughts; Personalized training program; Real-time feedback closed-loop system; Relationship between physical health; Volleyball player training load","Document Type":"Article"},{"Authors":"Wang T.-C.; Liu T.-Y.; Pan C.-Y.; Tseng Y.-T.; Tang T.-W.; Tsai C.-L.","Title":"Effects of mixed reality-based instruction on visual attention and stroke performance in novice badminton players","Year":2026,"Source":"Journal of Exercise Science and Fitness","Cited by":0,"Abstract":"This study examined the effects of a mixed reality (MR)-based auxiliary training system on visual-motor skill acquisition, with a focus on hitting accuracy and oculomotor performance in novice badminton players.","Author Keywords":"Hitting accuracy; Mixed reality (MR); Motor skill learning; Oculomotor performance; Visual attention","Document Type":"Article"},{"Authors":"Yang Z.; Gao S.; Guo K.","Title":"Effects of sports information, training, psychological preparation, and peer support on perceived athletic competence among Chinese university students","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"The study explores the factors influencing perceived athletic competence among Chinese university students, focusing on sports knowledge, training quality, psychological preparation, peer support, goal setting, and external motivation. Results from Structural Equations Modeling (SEM) revealed significant relationships.","Author Keywords":"Athletes; Goals; Motivation; Students; Universities","Document Type":"Article"},{"Authors":"Mora J.S.M.; Medina R.A.B.; Molina V.M.; Hernandez Rincon E.H.","Title":"Beyond the conventional: Artificial intelligence in identifying risk factors in sports injuries. A scoping review","Year":2026,"Source":"International Journal of Medical Informatics","Cited by":0,"Abstract":"Purpose: To map contemporary uses of artificial intelligence (AI) to identify conventional and unconventional risk factors for sports injuries in athletes. Fifty-nine studies met inclusion criteria. AI for sports-injury risk is expanding rapidly, led by classical machine learning on multimodal sensor data.","Author Keywords":"Algorithms; Artificial intelligence; Risk factors; Sports injuries","Document Type":"Review"},{"Authors":"Chen R.; Zhang Y.","Title":"Sports Injury Risk Prediction and Intervention for College Students Based on Decision Tree Algorithm","Year":2026,"Source":"Proceedings of The 2nd International Conference on Digital Society, Information Science and Risk Management, ICDIR 2026","Cited by":0,"Abstract":"Sports injuries pose significant risks to elite athletes careers and performance, necessitating accurate predictive models for proactive prevention strategies. The Decision Tree model achieved superior performance with 82.4% accuracy, 85.3% recall, 78.6% precision, and an AUC of 0.876.","Author Keywords":"decision tree; elite athletes; feature importance; machine learning; preventive medicine; risk assessment; Sports injury prediction; training load management","Document Type":"Conference paper"},{"Authors":"Xu Z.; Sun W.; Qian H.; Yao M.","Title":"Construction and application of a model for predicting athletes' injury risk based on machine learning","Year":2026,"Source":"BMC Medical Informatics and Decision Making","Cited by":1,"Abstract":"This study evaluated machine learning (ML) models for injury risk in 300 male professional football players monitored over two competitive seasons. Random forests outperformed other models, achieving accuracy 85.6%, precision 82.1%, recall 80.3%, F1-score 81.2%, and AUC 90.5%.","Author Keywords":"Athlete monitoring; Injury prediction; Machine learning; Random forest; Sports medicine","Document Type":"Article"},{"Authors":"Wei X.; Liang S.; Diao W.","Title":"Prediction of athlete performance based on a gradient regression model","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"Accurate prediction of athlete performance is a challenge of significance in sports science and analytics with application in training design, injury prevention, and talent management. The Gradient Regression Model had an R2 of 0.923, higher than Neural Networks and Random Forest baselines.","Author Keywords":"Athlete performance prediction; Gradient regression; Machine learning; Model interpretability; Sports analytics; Sports science","Document Type":"Article"},{"Authors":"Wu A.; Zhang A.; Zhou C.","Title":"AI and big data personalized training protocol for Chinese youth basketball","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"This study protocol describes the design and validation of an AI- and big-data-driven personalized training system aimed at testing whether integrated multidimensional data and dynamic feedback can improve talent identification and training outcomes in Chinese youth basketball.","Author Keywords":"Artificial intelligence; Big data; Dynamic feedback; Personalized training; Youth basketball","Document Type":"Article"},{"Authors":"Zhang Q.; Wang Q.; Niu Y.","Title":"Adaptive training load optimization for track and field athletes: A reinforcement learning approach","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"The paper presents a detailed structure of the offline optimization of training loads using the DQN architecture. The proposed method has shown high ability in managing training load and controlling the risk of injury, dynamically reducing risk while maintaining athlete performance.","Author Keywords":"Adaptive Training Planning; Athlete Performance; Deep Reinforcement Learning; Training Optimization","Document Type":"Article"},{"Authors":"Senel A.A.; Adilogullari G.E.; Senel E.","Title":"Modelling the effect of motivation on mental health components with fuzzy logic among elite athletes","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"This study models the joint effects of intrinsic motivation, psychological safety, and mental well-being on anxiety, depression, athlete-specific strain, and burnout using an interpretable fuzzy-logic framework. The FIS model offers superior predictive accuracy compared to standard linear approaches.","Author Keywords":"Applied mathematics; Burnout; Fuzzy logic; Mental health; Motivation; Sport psychology","Document Type":"Article"},{"Authors":"Haller N.; Stanin T.; Strepp T.; Blumkaitis J.; Stoggl T.L.","Title":"On the relationship between external and internal load variables in elite youth soccer players","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"The study investigated the relationship between external and internal training load measures in 25 male elite youth soccer players over 3 months. Subjective measures showed stronger and more consistent associations with training load than biomarkers or neuromuscular testing.","Author Keywords":"Biomarkers; Football; Injury prevention; Load monitoring; Neuromuscular testing","Document Type":"Article"},{"Authors":"Troyer W.D.; Phrathep D.; Pagan-Rosado R.; Kruse R.C.","Title":"Diagnosis and Management of Hamstring Muscle Injuries in Athletes","Year":2026,"Source":"Current Physical Medicine and Rehabilitation Reports","Cited by":0,"Abstract":"Advanced imaging, particularly MRI, plays a central role in accurate injury classification and prognostication. Rehabilitation remains the cornerstone of management, with growing evidence supporting eccentric training, neuromuscular control, and flexibility in reducing reinjury risk.","Author Keywords":"Athlete Injury Prevention; Hamstring Injuries; Hamstring Strain; Muscle Injury Classification; Rehabilitation; Return to Play","Document Type":"Review"},{"Authors":"Han G.; Zhang Y.; Sun B.","Title":"Wearable sensor big data analysis reveals spatiotemporal injury patterns in professional tennis players","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"This study establishes a comprehensive analytical framework for characterizing spatiotemporal distribution patterns of sports injuries in professional tennis. The Transformer-based prediction model achieved 91.5% accuracy with 0.956 AUC, significantly outperforming traditional statistical methods.","Author Keywords":"Big data analytics; Injury prediction; Machine learning; Spatiotemporal analysis; Tennis injuries; Wearable sensors","Document Type":"Article"},{"Authors":"Meng F.","Title":"Hybrid spatio-temporal transformer and variational autoencoder framework for advanced sports injury prediction and analysis","Year":2026,"Source":"Kuwait Journal of Science","Cited by":0,"Abstract":"This work introduces a new spatio-temporal transformer variational autoencoder (STT-VAE) structure which combines spatio-temporal modelling with latent space learning. The novel STT-VAE architecture achieves 99.56% classification accuracy, much higher than random forest (95.5%) and XGBoost (97%).","Author Keywords":"Feature importance; Latent space analysis; Machine learning; Prediction accuracy; Spatio-temporal transformer; Sports injury prediction; Variational autoencoder","Document Type":"Article"},{"Authors":"Afonso J.; Pizarro A.; Pizzari T.; Clemente F.M.","Title":"Injury risk and prevention research in sports: Are titles delivering on their promises?","Year":2026,"Source":"Journal of Sport and Health Science","Cited by":1,"Abstract":"[No abstract available]","Author Keywords":"","Document Type":"Note"},{"Authors":"Wang Z.; Guan Z.; Wang Z.; Zhou H.","Title":"MSTR-RiskNet: A multi-scale temporal relational framework for training-related injury risk prediction","Year":2026,"Source":"Applied Soft Computing","Cited by":0,"Abstract":"We propose MSTR-RiskNet, a multi-scale temporal relational framework combining hierarchical temporal encoding, dynamic relational graph learning, and discrete-time survival prediction. The results indicate that combining temporal and relational representation learning is promising for injury risk forecasting.","Author Keywords":"Graph neural networks; Injury risk prediction; Sports analytics; Survival analysis; Temporal modeling","Document Type":"Article"},{"Authors":"Seyhan S.; Acar G.; Bilici M.F.","Title":"Injury prevention relevance: markerless functional performance testing in athletes with chronic ankle instability","Year":2026,"Source":"BMC Sports Science, Medicine and Rehabilitation","Cited by":0,"Abstract":"Athletes with chronic ankle instability (CAI) demonstrated meaningful deficits in agility and multidirectional hop performance. These findings support the inclusion of multiplanar hop and agility assessments in functional evaluation and rehabilitation monitoring.","Author Keywords":"Ai motion analysis; Chronic ankle instability; Countermovement jump (CMJ); Performance tests; Physical functional performance","Document Type":"Article"},{"Authors":"Yu Z.; Bi G.; Qin Y.; Wang W.","Title":"Expertise shapes the kinematic and electromyographic characteristics of on-ice side-cutting in elite versus beginner ice hockey players","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"Elite ice hockey players exhibited greater proximal joint flexion during the center-of-mass transfer phase, alongside lower quadriceps activation and a higher knee co-activation ratio. These findings indicate that expertise is associated with optimized kinematics and lower activation costs.","Author Keywords":"Biomechanics; Ice hockey; Kinematics; Neuromuscular control; OpenSim; Side-cutting maneuver","Document Type":"Article"},{"Authors":"Cheng F.; Al-Hashimy H.N.H.; Yao J.","Title":"Use of SEM-PLS analysis to predict sports injuries in professional football players through warehouse technology data","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"This study develops an interpretable PLS-SEM injury-risk model. Higher mechanical load and elevated BMI were positively associated with injury incidence. The final model explained 61% of the variance in sports injuries.","Author Keywords":"Electronic athlete information system; Football; Football injury risk; Mechanical load; Musculoskeletal development; Performance; PLS-SEM; Sports injuries; Training programs","Document Type":"Article"},{"Authors":"Zhu D.; Li Q.; Li M.; Li Y.; Zhao X.","Title":"Neuroimaging-driven recommendation systems for personalized sports training and injury prevention","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"We propose NeuroAthleteNet, leveraging spatiotemporal neural feature extraction alongside graph-based connectivity analysis to establish precise mappings between neurophysiological markers and athletic performance metrics. Our approach significantly improves predictive accuracy and interpretability.","Author Keywords":"Graph Neural Networks; Injury Prevention; Multimodal Integration; Neuroimaging; Personalized Training","Document Type":"Article"},{"Authors":"Wang Y.; Lee S.","Title":"Development and validation of a machine learning model for non-contact injury prediction based on lower limb strength asymmetry in professional football","Year":2026,"Source":"Scientific Reports","Cited by":1,"Abstract":"This study developed and validated a machine learning model integrating lower limb strength asymmetry data for injury prediction in 312 professional male football players. The ensemble model achieved superior predictive performance (AUPRC: 0.759). Risk-stratified interventions were associated with 73% reduction in injury probability.","Author Keywords":"Athletic injuries; Football; Injury prediction; Isokinetic testing; Machine learning; Professional sports; Risk stratification; Strength asymmetry","Document Type":"Article"},{"Authors":"Shitara H.; Tajika T.; Miyamoto R.","Title":"A combined logistic regression and decision tree analysis of factors influencing throwing velocity and accuracy in little league baseball players","Year":2026,"Source":"JSES International","Cited by":0,"Abstract":"Logistic regression analysis revealed that pitcher experience, pitching workload, and dominant-side grip strength were significantly associated with high pitch velocity. Decision tree analysis showed that players with strong dominant-side grip had the highest probability of high pitch velocity.","Author Keywords":"Accuracy; Decision tree analysis; Kinesiology; Motor control; Performance; Pitching; Velocity; Youth baseball","Document Type":"Article"},{"Authors":"Bussey M.D.; McGeown J.P.; Dempsey S.","Title":"Sensitivity of brain injury criteria to anthropometric scaling assumptions in instrumented mouthguard data","Year":2026,"Source":"Journal of Biomechanics","Cited by":0,"Abstract":"Scaling assumptions significantly altered brain injury criteria magnitudes, particularly at size extremes. Female players modeled with male parameters exhibited up to 54% higher PRHIC values. Incorporating sex- and size-appropriate scaling can reduce systematic bias in head-impact surveillance.","Author Keywords":"Concussion; Head impact; Instrumented mouth guard; Rugby","Document Type":"Article"},{"Authors":"Shi K.; Ye Y.; Zheng T.; Tang K.; Wang L.","Title":"Effect of a sports-medicine-guided football program on physical fitness in adolescents: A controlled school-based trial","Year":2026,"Source":"Medicine","Cited by":0,"Abstract":"A school-based football curriculum integrating sports-medicine principles with explainable AI-driven feedback leads to significantly greater improvements in key physical-fitness outcomes. Injury risk prediction was supported by an explainable machine-learning ensemble model (CatBoost + Gradient Boosting).","Author Keywords":"adolescent physical fitness; explainable artificial intelligence; injury risk prediction; machine learning ensemble; sports medicine intervention","Document Type":"Article"}]

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","and","or","of","in","to","for","with","on","is","are","was","were",
    "this","that","these","those","from","by","as","at","be","been","being","have",
    "has","had","do","does","did","not","but","it","its","into","than","more","also",
    "which","can","will","may","their","our","we","they","i","you","he","she","study",
    "based","using","results","data","paper","proposed","approach","method","methods",
    "model","used","use","two","three","high","new","different","however","thus","show",
    "shows","showed","analysis","significant","number","among","between","within",
    "during","after","both","all","each","were","been","such","well","most","more",
    "only","than","then","when","where","while","with","without","about","above",
    "across","found","show","however","therefore","although","including","through",
    "compared","associated","related","provide","provides","suggest","suggests",
    "demonstrate","demonstrates","indicates","indicated","indicate","presented",
    "training","injury","sport","sports","athlete","athletes","performance","prediction",
    "risk","learning","machine","deep","model","models","data","dataset","approach",
    "method","methods","using","used","results","study","studies","showed","shown",
}

TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#161B22",
        "plot_bgcolor": "#161B22",
        "font": {"color": "#E6EDF3", "family": "Inter, sans-serif"},
        "xaxis": {"gridcolor": "#21262D", "linecolor": "#30363D", "zerolinecolor": "#30363D"},
        "yaxis": {"gridcolor": "#21262D", "linecolor": "#30363D", "zerolinecolor": "#30363D"},
        "colorway": ["#58A6FF","#3FB950","#D2A8FF","#FFA657","#F78166","#79C0FF"],
    }
}

def clean_words(text):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]

def parse_authors(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [a.strip() for a in re.split(r'[;,]', str(cell)) if a.strip()]

def load_csv(file):
    df = pd.read_csv(file, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    rename = {}
    for col in df.columns:
        low = col.lower()
        if "author" in low and "keyword" not in low and "id" not in low and "full" not in low:
            rename[col] = "Authors"
        elif low in ("title","document title"):
            rename[col] = "Title"
        elif low == "year":
            rename[col] = "Year"
        elif "abstract" in low:
            rename[col] = "Abstract"
        elif "cited" in low:
            rename[col] = "Cited by"
        elif "source" in low:
            rename[col] = "Source"
        elif "document type" in low:
            rename[col] = "Document Type"
        elif "author keyword" in low or "keywords" in low:
            rename[col] = "Author Keywords"
    df.rename(columns=rename, inplace=True)
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "Cited by" in df.columns:
        df["Cited by"] = pd.to_numeric(df["Cited by"], errors="coerce").fillna(0).astype(int)
    return df

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1117;
    color: #E6EDF3;
}

/* Hero */
.hero-wrap { padding: 8px 0 4px 0; }
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #3FB950; margin-bottom: 6px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    color: #E6EDF3; line-height: 1.15; margin-bottom: 4px;
}
.hero-title span { color: #58A6FF; }
.hero-sub { font-size: 0.92rem; color: #8B949E; margin-bottom: 14px; }

/* Research question */
.rq-box {
    background: linear-gradient(135deg, #161B22 0%, #1C2333 100%);
    border: 1px solid #30363D;
    border-top: 3px solid #58A6FF;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 20px;
}
.rq-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #58A6FF; margin-bottom: 8px;
}
.rq-text { font-size: 1.0rem; color: #E6EDF3; line-height: 1.6; }

/* Keywords */
.kw-pill {
    display: inline-block;
    background: #1C2333; border: 1px solid #388BFD40;
    border-radius: 20px; padding: 5px 14px; margin: 3px;
    font-size: 0.82rem; color: #79C0FF; font-weight: 500;
}

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin: 20px 0; }
.metric-card {
    flex: 1; background: #161B22;
    border: 1px solid #21262D; border-radius: 12px;
    padding: 18px 16px; text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 700; color: #3FB950; }
.metric-lbl { font-size: 0.78rem; color: #8B949E; margin-top: 4px; }

/* Section headers */
.sec-hdr {
    font-size: 1.05rem; font-weight: 600; color: #E6EDF3;
    border-left: 3px solid #3FB950;
    padding-left: 12px; margin: 28px 0 14px 0;
}

/* Insight boxes */
.insight {
    background: #161B22; border: 1px solid #21262D;
    border-left: 3px solid #58A6FF;
    border-radius: 6px; padding: 10px 14px;
    font-size: 0.83rem; color: #8B949E; margin-top: 6px;
}

/* Source mode radio */
.source-label { font-size: 0.8rem; color: #8B949E; margin-bottom: 6px; }

/* Sidebar */
div[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid #21262D;
}

/* Divider */
hr { border-color: #21262D !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚽ Fuente de datos")
    st.markdown('<div class="source-label">Elige cómo cargar el CSV:</div>', unsafe_allow_html=True)

    source_mode = st.radio(
        "Fuente",
        ["✅  Dataset incluido", "📁  Cargar archivo local", "🔗  URL pública de GitHub"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    github_url = None

    if source_mode == "📁  Cargar archivo local":
        uploaded_file = st.file_uploader("Archivo CSV de Scopus", type=["csv"], label_visibility="collapsed")
    elif source_mode == "🔗  URL pública de GitHub":
        github_url = st.text_input("URL raw de GitHub", placeholder="https://raw.githubusercontent.com/...")

    st.markdown("---")
    st.markdown("### 🔍 Filtros")

    # Placeholders — filled after data load
    year_filter = None
    min_cites = 0
    doc_type_filter = []

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem;color:#8B949E;">Grupo 11 · ISIL 2026<br>PA3 — StartIA<br>Fuente: Scopus</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">⚽ Análisis Bibliométrico · Scopus · 2026</div>
  <div class="hero-title">ML & <span>Rendimiento Deportivo</span></div>
  <div class="hero-sub">Predicción de lesiones y optimización del rendimiento en atletas profesionales</div>
</div>
""", unsafe_allow_html=True)

with st.expander("📌 Keywords utilizadas", expanded=True):
    for kw in ["Machine Learning", "Soccer", "Injury Prevention", "Performance"]:
        st.markdown(f'<span class="kw-pill">{kw}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="insight">Puedes cargar un CSV local desde la barra lateral, leerlo desde una URL RAW de GitHub, o usar el dataset incluido. '
        'El dashboard detecta automáticamente columnas clave como autores, título, año, citas, abstract y tipo de documento.</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  RESEARCH QUESTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="rq-box">
  <div class="rq-label">🔬 Pregunta de Investigación — Grupo 11</div>
  <div class="rq-text">
    ¿Cómo se utiliza el <strong>Machine Learning</strong> para predecir el rendimiento deportivo
    y reducir el riesgo de lesiones en jugadores de fútbol profesional?
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def get_embedded():
    return pd.DataFrame(EMBEDDED_DATA)

df_raw = None

if source_mode == "✅  Dataset incluido":
    df_raw = get_embedded()
    st.success(f"✅ Dataset oficial cargado — {len(df_raw)} artículos")

elif source_mode == "📁  Cargar archivo local":
    if uploaded_file:
        df_raw = load_csv(uploaded_file)
        st.success(f"✅ CSV cargado — {len(df_raw)} artículos")
    else:
        st.info("⬅️  Sube tu archivo CSV de Scopus en el panel lateral.")
        st.stop()

elif source_mode == "🔗  URL pública de GitHub":
    if github_url and github_url.startswith("http"):
        try:
            df_raw = pd.read_csv(github_url, encoding="utf-8-sig")
            df_raw = load_csv(io.StringIO(df_raw.to_csv(index=False)))
            st.success(f"✅ Dataset cargado desde GitHub — {len(df_raw)} artículos")
        except Exception as e:
            st.error(f"No se pudo cargar desde esa URL: {e}")
            st.stop()
    else:
        st.info("⬅️  Ingresa una URL raw de GitHub en el panel lateral.")
        st.stop()

if df_raw is None or df_raw.empty:
    st.stop()

# ─────────────────────────────────────────────
#  SIDEBAR FILTERS (dynamic)
# ─────────────────────────────────────────────
with st.sidebar:
    if "Year" in df_raw.columns:
        years = df_raw["Year"].dropna().astype(int)
        if not years.empty:
            yr_min, yr_max = int(years.min()), int(years.max())
            if yr_min < yr_max:
                year_filter = st.slider("Rango de años", yr_min, yr_max, (yr_min, yr_max))
            else:
                year_filter = (yr_min, yr_max)
                st.markdown(f'<div class="source-label">Año: {yr_min}</div>', unsafe_allow_html=True)

    if "Document Type" in df_raw.columns:
        doc_types = sorted(df_raw["Document Type"].dropna().unique().tolist())
        doc_type_filter = st.multiselect("Tipo de documento", doc_types, default=doc_types)

    if "Cited by" in df_raw.columns:
        max_c = int(df_raw["Cited by"].max())
        if max_c > 0:
            min_cites = st.slider("Mínimo de citas", 0, max_c, 0)

# Apply filters
df = df_raw.copy()
if year_filter and "Year" in df.columns:
    df = df[(df["Year"] >= year_filter[0]) & (df["Year"] <= year_filter[1])]
if doc_type_filter and "Document Type" in df.columns:
    df = df[df["Document Type"].isin(doc_type_filter)]
if min_cites and "Cited by" in df.columns:
    df = df[df["Cited by"] >= min_cites]

if df.empty:
    st.warning("No hay artículos con los filtros actuales. Ajústalos en el panel lateral.")
    st.stop()

# ─────────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────────
total = len(df)
citas = int(df["Cited by"].sum()) if "Cited by" in df.columns else "—"
autores_u = len(set(
    a for cell in df.get("Authors", pd.Series(dtype=str)).fillna("")
    for a in parse_authors(cell)
)) if "Authors" in df.columns else "—"
fuentes_u = df["Source"].nunique() if "Source" in df.columns else "—"

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip(
    [c1, c2, c3, c4],
    [total, citas, autores_u, fuentes_u],
    ["Artículos", "Citas totales", "Autores únicos", "Revistas"],
):
    col.markdown(
        f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ─────────────────────────────────────────────
#  ROW 1: Publicaciones por año + Top fuentes
# ─────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Tendencia de publicaciones</div>', unsafe_allow_html=True)
col_a, col_b = st.columns([3, 2])

with col_a:
    if "Year" in df.columns:
        by_year = df.groupby("Year").size().reset_index(name="Artículos")
        fig = px.bar(by_year, x="Year", y="Artículos",
                     title="Publicaciones por año",
                     color="Artículos",
                     color_continuous_scale=["#1C2B3A", "#58A6FF"],
                     template=TEMPLATE)
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          title_font_size=13, margin=dict(t=38,b=16,l=8,r=8))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight">💡 Un aumento sostenido refleja que el área está en expansión activa. La concentración en 2026 confirma la vigencia del tema.</div>', unsafe_allow_html=True)

with col_b:
    if "Source" in df.columns:
        top_src = df["Source"].value_counts().head(8).reset_index()
        top_src.columns = ["Revista", "N"]
        fig2 = px.bar(top_src, x="N", y="Revista", orientation="h",
                      title="Top revistas",
                      color="N",
                      color_continuous_scale=["#1A2E1A","#3FB950"],
                      template=TEMPLATE)
        fig2.update_layout(yaxis_title="", title_font_size=13,
                           coloraxis_showscale=False, margin=dict(t=38,b=16,l=8,r=8))
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
#  ROW 2: Más citados + Autores
# ─────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Impacto y autoría</div>', unsafe_allow_html=True)
col_c, col_d = st.columns(2)

with col_c:
    if "Cited by" in df.columns and "Title" in df.columns:
        top_c = df.nlargest(10, "Cited by")[["Title","Cited by","Year"]].copy()
        top_c["Título"] = top_c["Title"].str[:55] + "…"
        fig3 = px.bar(top_c, x="Cited by", y="Título", orientation="h",
                      title="Top 10 artículos más citados",
                      color="Cited by",
                      color_continuous_scale=["#2D1B45","#D2A8FF"],
                      template=TEMPLATE,
                      hover_data={"Title": True, "Year": True, "Título": False})
        fig3.update_layout(yaxis_title="", title_font_size=13,
                           coloraxis_showscale=False, margin=dict(t=38,b=16,l=8,r=8))
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('<div class="insight">💡 Los artículos más citados son las referencias clave del campo. Úsalos como base en tu marco teórico.</div>', unsafe_allow_html=True)

with col_d:
    if "Authors" in df.columns:
        all_auth = []
        for cell in df["Authors"].fillna(""):
            all_auth.extend(parse_authors(cell))
        counts = Counter(all_auth).most_common(15)
        if counts:
            adf = pd.DataFrame(counts, columns=["Autor","N"])
            fig4 = px.bar(adf, x="N", y="Autor", orientation="h",
                          title="Autores más productivos",
                          color="N",
                          color_continuous_scale=["#1A2E1A","#3FB950"],
                          template=TEMPLATE)
            fig4.update_layout(yaxis_title="", title_font_size=13,
                               coloraxis_showscale=False, margin=dict(t=38,b=16,l=8,r=8))
            fig4.update_traces(marker_line_width=0)
            st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────
#  ROW 3: Word cloud + Tipo de documento
# ─────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Análisis de contenido</div>', unsafe_allow_html=True)
col_e, col_f = st.columns([3, 2])

with col_e:
    if "Abstract" in df.columns:
        all_text = " ".join(df["Abstract"].dropna().astype(str))
        words = clean_words(all_text)
        if words:
            freq = Counter(words)
            wc = WordCloud(
                width=800, height=360,
                background_color="#161B22",
                colormap="Blues",
                max_words=80,
                prefer_horizontal=0.8,
                min_font_size=10,
            ).generate_from_frequencies(freq)
            fig_wc, ax = plt.subplots(figsize=(8, 3.6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig_wc.patch.set_facecolor("#161B22")
            st.markdown("**Palabras más frecuentes en abstracts**")
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)
            st.markdown('<div class="insight">💡 Las palabras dominantes validan la pertinencia de las keywords utilizadas en la búsqueda bibliométrica.</div>', unsafe_allow_html=True)

with col_f:
    if "Document Type" in df.columns:
        dt = df["Document Type"].value_counts().reset_index()
        dt.columns = ["Tipo", "N"]
        fig5 = px.pie(dt, names="Tipo", values="N",
                      title="Tipo de documento",
                      template=TEMPLATE,
                      color_discrete_sequence=["#58A6FF","#3FB950","#D2A8FF","#FFA657","#F78166"],
                      hole=0.4)
        fig5.update_layout(title_font_size=13, margin=dict(t=38,b=10,l=8,r=8))
        fig5.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig5, use_container_width=True)

    if "Cited by" in df.columns:
        fig6 = px.histogram(df, x="Cited by", nbins=15,
                            title="Distribución de citas",
                            template=TEMPLATE,
                            color_discrete_sequence=["#58A6FF"])
        fig6.update_layout(xaxis_title="Citas", yaxis_title="Artículos",
                           title_font_size=13, margin=dict(t=38,b=16,l=8,r=8), bargap=0.1)
        st.plotly_chart(fig6, use_container_width=True)

# ─────────────────────────────────────────────
#  ROW 4: Keywords de autores
# ─────────────────────────────────────────────
if "Author Keywords" in df.columns:
    st.markdown('<div class="sec-hdr">Keywords de autores</div>', unsafe_allow_html=True)
    all_kw = []
    for cell in df["Author Keywords"].fillna(""):
        kws = [k.strip().lower() for k in re.split(r'[;,]', str(cell)) if k.strip()]
        all_kw.extend(kws)
    if all_kw:
        kw_counts = Counter(all_kw).most_common(20)
        kw_df = pd.DataFrame(kw_counts, columns=["Keyword","N"])
        fig7 = px.bar(kw_df, x="N", y="Keyword", orientation="h",
                      title="Top 20 keywords de autores",
                      color="N",
                      color_continuous_scale=["#1C2B3A","#58A6FF"],
                      template=TEMPLATE)
        fig7.update_layout(height=520, yaxis_title="", title_font_size=13,
                           coloraxis_showscale=False, margin=dict(t=38,b=16,l=8,r=8))
        fig7.update_traces(marker_line_width=0)
        st.plotly_chart(fig7, use_container_width=True)

# ─────────────────────────────────────────────
#  TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Dataset completo</div>', unsafe_allow_html=True)
show_cols = [c for c in ["Title","Authors","Year","Source","Cited by","Document Type"] if c in df.columns]
search = st.text_input("🔍 Buscar en títulos…", placeholder="ej. injury prediction, random forest, soccer…")
disp = df[show_cols].copy()
if search and "Title" in disp.columns:
    disp = disp[disp["Title"].str.contains(search, case=False, na=False)]
st.dataframe(disp, use_container_width=True, height=300)

csv_dl = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️  Descargar dataset filtrado (CSV)", csv_dl,
                   "scopus_ml_sports_filtrado.csv", "text/csv")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.76rem;color:#8B949E;padding:8px 0'>"
    "Grupo 11 · PA3 · StartIA · ISIL 2026 · Dataset: Scopus · "
    "Keywords: Machine Learning · Soccer · Injury Prevention · Performance"
    "</div>",
    unsafe_allow_html=True,
)

cat > /mnt/user-data/outputs/pa3_dashboard/app.py << 'PYEOF'
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
import re
import io

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ML & Sports · Grupo 11",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#  EMBEDDED SCOPUS DATASET
# ═══════════════════════════════════════════════════════════════
EMBEDDED_DATA = [
    {"Authors":"Xia X.; Chen Q.; Wang Z.","Title":"Deep reinforcement learning-driven personalized training load control algorithm for competitive sports performance optimization","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"Traditional training load management methods in competitive sports rely heavily on subjective assessments and standardized protocols, often failing to account for individual physiological variations and dynamic adaptation responses. This research proposes a deep reinforcement learning (DRL) framework for personalized training load optimization that integrates real-time physiological monitoring, individual athlete characteristics, and adaptive decision-making algorithms. Empirical validation shows performance improvements averaging 12.3% compared to traditional periodization-based methods, with injury rate reductions of 43% and training efficiency enhancements ranging from 1.15 to 1.42 times conventional approaches.","Author Keywords":"Athlete monitoring; Competitive sports; Deep reinforcement learning; Load management; Performance optimization; Training individualization","Document Type":"Article"},
    {"Authors":"Masood Z.; Luke D.; Kenny R.; Bondi D.; Clansey A.; Wu L.C.","Title":"Head impact biomechanics across men's and women's contact sports: a comparative and clustering analysis","Year":2026,"Source":"Scientific Reports","Cited by":1,"Abstract":"Sports head impacts have been associated with both acute and long-term brain trauma. We applied unsupervised k-means and t-SNE models to examine clustering in impact magnitude and frequency features. Statistically significant cross-sport differences were found in all biomechanical features. Men's football exhibited the highest resultant median peak kinematics, while women's soccer showed lowest median resultant kinematics.","Author Keywords":"Head impact biomechanics; Injury biomechanics; Instrumented mouthguard; Signal processing; Unsupervised machine learning","Document Type":"Article"},
    {"Authors":"Zhang G.; Xu B.; Wang J.","Title":"Graph structure modeling for optimizing the relationship between training load and physical health in volleyball players using Graph of Thought","Year":2026,"Source":"Systems and Soft Computing","Cited by":0,"Abstract":"This study proposes a personalized optimization framework based on Graph of Thoughts (GOT). Experimental results show mean consistency scores of 0.89 for HRV prediction and 0.86 for creatine kinase prediction. The framework reduced injury risk and improved overall physical health.","Author Keywords":"Graph of Thoughts; Personalized training program; Real-time feedback closed-loop system; Volleyball player training load","Document Type":"Article"},
    {"Authors":"Wang T.-C.; Liu T.-Y.; Pan C.-Y.; Tseng Y.-T.; Tang T.-W.; Tsai C.-L.","Title":"Effects of mixed reality-based instruction on visual attention and stroke performance in novice badminton players","Year":2026,"Source":"Journal of Exercise Science and Fitness","Cited by":0,"Abstract":"This study examined the effects of a mixed reality (MR)-based auxiliary training system on visual-motor skill acquisition. The MRT group exhibited a significant increase in successful shots and demonstrated enhanced fixation count and saccade velocity.","Author Keywords":"Hitting accuracy; Mixed reality (MR); Motor skill learning; Oculomotor performance; Visual attention","Document Type":"Article"},
    {"Authors":"Yang Z.; Gao S.; Guo K.","Title":"Effects of sports information, training, psychological preparation, and peer support on perceived athletic competence among Chinese university students","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"Results from Structural Equations Modeling (SEM) revealed significant relationships: sports information (beta=0.32), quality of training (beta=0.28), psychological preparation (beta=0.34), and peer support (beta=0.30) all positively influenced perceived athletic competence.","Author Keywords":"Athletes; Goals; Motivation; Students; Universities","Document Type":"Article"},
    {"Authors":"Mora J.S.M.; Medina R.A.B.; Molina V.M.; Hernandez Rincon E.H.","Title":"Beyond the conventional: Artificial intelligence in identifying risk factors in sports injuries. A scoping review","Year":2026,"Source":"International Journal of Medical Informatics","Cited by":0,"Abstract":"Fifty-nine studies met inclusion criteria. AI for sports-injury risk is expanding rapidly, led by classical machine learning on multimodal sensor data. Key gaps are external validity and reproducibility.","Author Keywords":"Algorithms; Artificial intelligence; Risk factors; Sports injuries","Document Type":"Review"},
    {"Authors":"Chen R.; Zhang Y.","Title":"Sports Injury Risk Prediction and Intervention for College Students Based on Decision Tree Algorithm","Year":2026,"Source":"Proceedings ICDIR 2026","Cited by":0,"Abstract":"The Decision Tree model achieved superior performance with 82.4% accuracy, 85.3% recall, 78.6% precision, and an AUC of 0.876, outperforming Random Forest (80.2%), SVM (76.8%), and Logistic Regression (74.1%).","Author Keywords":"decision tree; elite athletes; feature importance; machine learning; risk assessment; Sports injury prediction; training load management","Document Type":"Conference paper"},
    {"Authors":"Xu Z.; Sun W.; Qian H.; Yao M.","Title":"Construction and application of a model for predicting athletes' injury risk based on machine learning","Year":2026,"Source":"BMC Medical Informatics and Decision Making","Cited by":1,"Abstract":"Random forests outperformed other models in 300 professional football players, achieving accuracy 85.6%, precision 82.1%, recall 80.3%, F1-score 81.2%, and AUC 90.5%. SHAP and LIME identified prior injury, training intensity, and recovery time as strongest predictors.","Author Keywords":"Athlete monitoring; Injury prediction; Machine learning; Random forest; Sports medicine","Document Type":"Article"},
    {"Authors":"Wei X.; Liang S.; Diao W.","Title":"Prediction of athlete performance based on a gradient regression model","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"The Gradient Regression Model had an R2 of 0.923, higher than Neural Networks (R2=0.901) and Random Forest (R2=0.887). The model offers superior interpretability with applications in individualized training and performance monitoring.","Author Keywords":"Athlete performance prediction; Gradient regression; Machine learning; Model interpretability; Sports analytics; Sports science","Document Type":"Article"},
    {"Authors":"Wu A.; Zhang A.; Zhou C.","Title":"AI and big data personalized training protocol for Chinese youth basketball","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"This study describes an AI- and big-data-driven personalized training system for Chinese youth basketball, integrating multidimensional data and dynamic feedback to improve talent identification and training outcomes.","Author Keywords":"Artificial intelligence; Big data; Dynamic feedback; Personalized training; Youth basketball","Document Type":"Article"},
    {"Authors":"Zhang Q.; Wang Q.; Niu Y.","Title":"Adaptive training load optimization for track and field athletes: A reinforcement learning approach","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"The DQN-based framework overcomes the simulation gap by applying a data-driven digital twin model. Results show high ability in managing training load and controlling injury risk while maintaining athlete performance within optimal range.","Author Keywords":"Adaptive Training Planning; Athlete Performance; Deep Reinforcement Learning; Training Optimization","Document Type":"Article"},
    {"Authors":"Senel A.A.; Adilogullari G.E.; Senel E.","Title":"Modelling the effect of motivation on mental health components with fuzzy logic among elite athletes","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"The Mamdani-type Fuzzy Inference System models joint effects of intrinsic motivation, psychological safety, and mental well-being on anxiety and burnout in 247 elite athletes. FIS showed superior predictive accuracy compared to standard linear approaches.","Author Keywords":"Applied mathematics; Burnout; Fuzzy logic; Mental health; Motivation; Sport psychology","Document Type":"Article"},
    {"Authors":"Haller N.; Stanin T.; Strepp T.; Blumkaitis J.; Stoggl T.L.","Title":"On the relationship between external and internal load variables in elite youth soccer players","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"The study investigated external and internal training load in 25 male elite youth soccer players. Subjective measures showed stronger and more consistent associations with training load than biomarkers or neuromuscular testing.","Author Keywords":"Biomarkers; Football; Injury prevention; Load monitoring; Neuromuscular testing","Document Type":"Article"},
    {"Authors":"Troyer W.D.; Phrathep D.; Pagan-Rosado R.; Kruse R.C.","Title":"Diagnosis and Management of Hamstring Muscle Injuries in Athletes","Year":2026,"Source":"Current Physical Medicine and Rehabilitation Reports","Cited by":0,"Abstract":"Advanced imaging, particularly MRI, plays a central role in injury classification and prognostication. Rehabilitation with eccentric training, neuromuscular control, and flexibility reduces reinjury risk significantly.","Author Keywords":"Athlete Injury Prevention; Hamstring Injuries; Hamstring Strain; Muscle Injury Classification; Rehabilitation; Return to Play","Document Type":"Review"},
    {"Authors":"Han G.; Zhang Y.; Sun B.","Title":"Wearable sensor big data analysis reveals spatiotemporal injury patterns in professional tennis players","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"ML algorithms including LSTM networks, clustering analysis, and ensemble methods identified injury patterns. The Transformer-based model achieved 91.5% accuracy with 0.956 AUC, significantly outperforming traditional statistical methods.","Author Keywords":"Big data analytics; Injury prediction; Machine learning; Spatiotemporal analysis; Tennis injuries; Wearable sensors","Document Type":"Article"},
    {"Authors":"Meng F.","Title":"Hybrid spatio-temporal transformer and variational autoencoder framework for advanced sports injury prediction","Year":2026,"Source":"Kuwait Journal of Science","Cited by":0,"Abstract":"The STT-VAE architecture achieves 99.56% classification accuracy, significantly higher than random forest (95.5%) and XGBoost (97%), with precision 99.45%, recall 99.46%, F1-score 99.3%.","Author Keywords":"Feature importance; Latent space analysis; Machine learning; Sports injury prediction; Spatio-temporal transformer; Variational autoencoder","Document Type":"Article"},
    {"Authors":"Afonso J.; Pizarro A.; Pizzari T.; Clemente F.M.","Title":"Injury risk and prevention research in sports: Are titles delivering on their promises?","Year":2026,"Source":"Journal of Sport and Health Science","Cited by":1,"Abstract":"Critical evaluation of injury risk and prevention research titles in sports science, examining whether research outcomes align with stated objectives and questioning the validity of injury prediction claims.","Author Keywords":"Injury prevention; Research methodology; Sports science; Systematic review","Document Type":"Note"},
    {"Authors":"Wang Z.; Guan Z.; Wang Z.; Zhou H.","Title":"MSTR-RiskNet: A multi-scale temporal relational framework for training-related injury risk prediction","Year":2026,"Source":"Applied Soft Computing","Cited by":0,"Abstract":"MSTR-RiskNet combines hierarchical temporal encoding, dynamic relational graph learning, and discrete-time survival prediction. The results indicate that combining temporal and relational representation learning is promising for injury risk forecasting.","Author Keywords":"Graph neural networks; Injury risk prediction; Sports analytics; Survival analysis; Temporal modeling","Document Type":"Article"},
    {"Authors":"Seyhan S.; Acar G.; Bilici M.F.","Title":"Injury prevention: markerless functional performance testing in athletes with chronic ankle instability","Year":2026,"Source":"BMC Sports Science, Medicine and Rehabilitation","Cited by":0,"Abstract":"Athletes with chronic ankle instability showed significantly poorer performance in agility tests and multidirectional hop tasks. These findings support multiplanar hop and agility assessments in functional evaluation.","Author Keywords":"Ai motion analysis; Chronic ankle instability; Countermovement jump; Performance tests; Physical functional performance","Document Type":"Article"},
    {"Authors":"Yu Z.; Bi G.; Qin Y.; Wang W.","Title":"Expertise shapes kinematic and electromyographic characteristics of on-ice side-cutting in elite ice hockey players","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"Elite players exhibited greater peak hip and knee flexion alongside lower quadriceps activation and higher knee co-activation ratio. Expertise is associated with optimized kinematics and lower activation costs.","Author Keywords":"Biomechanics; Ice hockey; Kinematics; Neuromuscular control; OpenSim; Side-cutting maneuver","Document Type":"Article"},
    {"Authors":"Cheng F.; Al-Hashimy H.N.H.; Yao J.","Title":"Use of SEM-PLS analysis to predict sports injuries in professional football players through warehouse technology data","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"Higher mechanical load and elevated BMI were positively associated with injury incidence. Well-structured training and effective monitoring were negatively associated with injuries. The final model explained 61% of variance in sports injuries.","Author Keywords":"Football; Football injury risk; Mechanical load; PLS-SEM; Sports injuries; Training programs","Document Type":"Article"},
    {"Authors":"Zhu D.; Li Q.; Li M.; Li Y.; Zhao X.","Title":"Neuroimaging-driven recommendation systems for personalized sports training and injury prevention","Year":2026,"Source":"Scientific Reports","Cited by":0,"Abstract":"NeuroAthleteNet leverages spatiotemporal neural feature extraction with graph-based connectivity analysis. NeuroSportSync synchronizes neuroimaging data with real-time biomechanical signals for comprehensive injury risk assessment.","Author Keywords":"Graph Neural Networks; Injury Prevention; Multimodal Integration; Neuroimaging; Personalized Training","Document Type":"Article"},
    {"Authors":"Wang Y.; Lee S.","Title":"Development and validation of a machine learning model for non-contact injury prediction based on lower limb strength asymmetry in professional football","Year":2026,"Source":"Scientific Reports","Cited by":1,"Abstract":"The ensemble model achieved AUPRC of 0.759 in 312 professional football players. Risk-stratified interventions were associated with 73% reduction in injury probability. Preliminary analysis suggested 215,800 EUR net savings per club season.","Author Keywords":"Athletic injuries; Football; Injury prediction; Isokinetic testing; Machine learning; Professional sports; Risk stratification; Strength asymmetry","Document Type":"Article"},
    {"Authors":"Shitara H.; Tajika T.; Miyamoto R.","Title":"Combined logistic regression and decision tree analysis of factors influencing throwing velocity and accuracy in little league baseball","Year":2026,"Source":"JSES International","Cited by":0,"Abstract":"Pitcher experience (OR=3.45), pitching workload over 100/week (OR=5.16), and dominant-side grip strength (OR=1.17) were significantly associated with high pitch velocity. Decision tree analysis validated these findings.","Author Keywords":"Decision tree analysis; Kinesiology; Motor control; Performance; Pitching; Velocity; Youth baseball","Document Type":"Article"},
    {"Authors":"Bussey M.D.; McGeown J.P.; Dempsey S.","Title":"Sensitivity of brain injury criteria to anthropometric scaling assumptions in instrumented mouthguard data","Year":2026,"Source":"Journal of Biomechanics","Cited by":0,"Abstract":"Scaling assumptions significantly altered brain injury criteria, particularly at size extremes. Female players modeled with male parameters exhibited up to 54% higher PRHIC values. Sex- and size-appropriate scaling reduces systematic bias.","Author Keywords":"Concussion; Head impact; Instrumented mouth guard; Rugby","Document Type":"Article"},
    {"Authors":"Shi K.; Ye Y.; Zheng T.; Tang K.; Wang L.","Title":"Effect of a sports-medicine-guided football program on physical fitness in adolescents: A controlled school-based trial","Year":2026,"Source":"Medicine","Cited by":0,"Abstract":"A sports-medicine-guided curriculum with explainable AI (CatBoost + Gradient Boosting) produced significantly greater improvements in lung capacity, VO2max, sprint and running performance than a standard curriculum in 195 adolescents.","Author Keywords":"adolescent physical fitness; explainable artificial intelligence; injury risk prediction; machine learning ensemble; sports medicine intervention","Document Type":"Article"},
]

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
STOPWORDS = {
    "the","a","an","and","or","of","in","to","for","with","on","is","are","was","were",
    "this","that","these","those","from","by","as","at","be","been","being","have","has",
    "had","do","does","did","not","but","it","its","into","than","more","also","which",
    "can","will","may","their","our","we","they","i","you","he","she","study","based",
    "using","results","data","paper","proposed","approach","method","methods","model",
    "used","use","two","three","high","new","different","however","thus","show","shows",
    "showed","analysis","significant","number","among","between","within","during",
    "after","both","all","each","such","well","most","only","when","where","while",
    "without","about","found","therefore","although","including","through","compared",
    "associated","related","provide","provides","suggest","suggests","demonstrate",
    "demonstrates","indicates","indicate","presented","training","injury","sport",
    "sports","athlete","athletes","performance","prediction","risk","learning","machine",
    "deep","model","models","approach","methods","study","studies","showed","shown",
    "furthermore","moreover","additionally","specifically","particularly","significantly",
    "respectively","previously","subsequently","currently","recently","generally",
    "typically","commonly","widely","highly","relatively","primarily","mainly","largely",
}

PALETTE = {
    "blue":    "#58A6FF",
    "green":   "#3FB950",
    "purple":  "#D2A8FF",
    "orange":  "#FFA657",
    "red":     "#F78166",
    "cyan":    "#79C0FF",
    "bg":      "#0D1117",
    "surface": "#161B22",
    "border":  "#21262D",
    "border2": "#30363D",
    "text":    "#E6EDF3",
    "muted":   "#8B949E",
}

def plotly_theme():
    return dict(
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border2"],
                   zerolinecolor=PALETTE["border2"], tickfont=dict(size=11)),
        yaxis=dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border2"],
                   zerolinecolor=PALETTE["border2"], tickfont=dict(size=11)),
        title_font=dict(size=14, color=PALETTE["text"]),
        legend=dict(bgcolor=PALETTE["surface"], bordercolor=PALETTE["border2"]),
        margin=dict(t=44, b=20, l=10, r=10),
        colorway=[PALETTE["blue"], PALETTE["green"], PALETTE["purple"],
                  PALETTE["orange"], PALETTE["red"], PALETTE["cyan"]],
    )

def clean_words(text):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]

def parse_authors(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [a.strip() for a in re.split(r'[;,]', str(cell)) if a.strip()]

def parse_keywords(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [k.strip().title() for k in re.split(r'[;,]', str(cell)) if k.strip()]

def load_csv(file):
    df = pd.read_csv(file, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    rename = {}
    for col in df.columns:
        low = col.lower()
        if "author" in low and "keyword" not in low and "id" not in low and "full" not in low:
            rename[col] = "Authors"
        elif low in ("title", "document title"):
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
        elif "author keyword" in low or ("keyword" in low and "index" not in low):
            rename[col] = "Author Keywords"
    df.rename(columns=rename, inplace=True)
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "Cited by" in df.columns:
        df["Cited by"] = pd.to_numeric(df["Cited by"], errors="coerce").fillna(0).astype(int)
    return df

@st.cache_data
def get_embedded():
    return pd.DataFrame(EMBEDDED_DATA)

# ═══════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
}}

/* ── SIDEBAR ── */
div[data-testid="stSidebar"] {{
    background: {PALETTE['bg']};
    border-right: 1px solid {PALETTE['border']};
}}
div[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.85rem !important;
    color: {PALETTE['muted']} !important;
}}
div[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
    font-size: 0.85rem;
}}

/* ── HERO ── */
.hero {{
    padding: 12px 0 6px;
    border-bottom: 1px solid {PALETTE['border']};
    margin-bottom: 24px;
}}
.hero-badge {{
    display: inline-block;
    background: #3FB95020;
    border: 1px solid #3FB95050;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['green']};
    margin-bottom: 10px;
}}
.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1.1;
    color: {PALETTE['text']};
    margin-bottom: 6px;
}}
.hero-title .accent {{ color: {PALETTE['blue']}; }}
.hero-sub {{
    font-size: 0.95rem;
    color: {PALETTE['muted']};
    max-width: 680px;
    line-height: 1.55;
}}
.hero-meta {{
    display: flex;
    gap: 20px;
    margin-top: 14px;
    flex-wrap: wrap;
}}
.hero-meta-item {{
    font-size: 0.78rem;
    color: {PALETTE['muted']};
    display: flex;
    align-items: center;
    gap: 5px;
}}
.hero-meta-item span {{ color: {PALETTE['text']}; font-weight: 500; }}

/* ── RESEARCH QUESTION ── */
.rq-card {{
    background: linear-gradient(135deg, {PALETTE['surface']} 0%, #1A2035 100%);
    border: 1px solid {PALETTE['border2']};
    border-left: 4px solid {PALETTE['blue']};
    border-radius: 12px;
    padding: 20px 24px;
    margin: 20px 0;
    position: relative;
    overflow: hidden;
}}
.rq-card::before {{
    content: '"';
    position: absolute;
    top: -10px;
    right: 16px;
    font-size: 6rem;
    color: {PALETTE['blue']};
    opacity: 0.08;
    font-family: Georgia, serif;
    line-height: 1;
}}
.rq-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {PALETTE['blue']};
    margin-bottom: 10px;
}}
.rq-text {{
    font-size: 1.05rem;
    color: {PALETTE['text']};
    line-height: 1.65;
    font-weight: 400;
}}
.rq-text strong {{ color: {PALETTE['blue']}; font-weight: 600; }}

/* ── KEYWORD PILLS ── */
.kw-wrap {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 16px 0; }}
.kw-pill {{
    background: #58A6FF15;
    border: 1px solid #58A6FF40;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    color: {PALETTE['cyan']};
    letter-spacing: 0.02em;
}}

/* ── METRIC CARDS ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 24px 0;
}}
.kpi-card {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
    padding: 20px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}}
.kpi-card::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}}
.kpi-blue::after  {{ background: {PALETTE['blue']}; }}
.kpi-green::after {{ background: {PALETTE['green']}; }}
.kpi-purple::after {{ background: {PALETTE['purple']}; }}
.kpi-orange::after {{ background: {PALETTE['orange']}; }}
.kpi-val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}}
.kpi-blue .kpi-val   {{ color: {PALETTE['blue']}; }}
.kpi-green .kpi-val  {{ color: {PALETTE['green']}; }}
.kpi-purple .kpi-val {{ color: {PALETTE['purple']}; }}
.kpi-orange .kpi-val {{ color: {PALETTE['orange']}; }}
.kpi-lbl {{
    font-size: 0.78rem;
    color: {PALETTE['muted']};
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.kpi-delta {{
    font-size: 0.72rem;
    color: {PALETTE['green']};
    margin-top: 6px;
}}

/* ── SECTION HEADER ── */
.sec-hdr {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 32px 0 16px;
}}
.sec-hdr-line {{
    flex: 1;
    height: 1px;
    background: {PALETTE['border']};
}}
.sec-hdr-text {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['muted']};
    white-space: nowrap;
}}
.sec-hdr-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {PALETTE['blue']};
    flex-shrink: 0;
}}

/* ── INSIGHT BOXES ── */
.insight {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-left: 3px solid {PALETTE['blue']};
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: {PALETTE['muted']};
    line-height: 1.5;
    margin-top: 8px;
}}
.insight-green {{ border-left-color: {PALETTE['green']}; }}
.insight-purple {{ border-left-color: {PALETTE['purple']}; }}

/* ── CHART WRAPPER ── */
.chart-card {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
    padding: 4px;
    margin-bottom: 16px;
}}

/* ── DATA TABLE ── */
.stDataFrame {{ border-radius: 10px; overflow: hidden; }}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {PALETTE['surface']};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {PALETTE['border']};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 0.85rem;
    font-weight: 500;
    color: {PALETTE['muted']};
}}
.stTabs [aria-selected="true"] {{
    background: {PALETTE['border2']} !important;
    color: {PALETTE['text']} !important;
}}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton button {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border2']};
    color: {PALETTE['text']};
    border-radius: 8px;
    font-size: 0.85rem;
}}

/* ── FOOTER ── */
.footer {{
    text-align: center;
    padding: 24px 0 12px;
    border-top: 1px solid {PALETTE['border']};
    margin-top: 40px;
}}
.footer-text {{
    font-size: 0.75rem;
    color: {PALETTE['muted']};
    line-height: 1.8;
}}
.footer-kws {{
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 8px;
    flex-wrap: wrap;
}}
.footer-kw {{
    font-size: 0.72rem;
    color: {PALETTE['border2']};
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}

hr {{ border-color: {PALETTE['border']} !important; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"### 📂 Fuente de datos")
    source_mode = st.radio(
        "Fuente",
        ["✅  Dataset incluido", "📁  Cargar CSV local", "🔗  URL de GitHub"],
        label_visibility="collapsed",
    )
    uploaded_file = None
    github_url = None
    if source_mode == "📁  Cargar CSV local":
        uploaded_file = st.file_uploader("CSV Scopus", type=["csv"], label_visibility="collapsed")
    elif source_mode == "🔗  URL de GitHub":
        github_url = st.text_input("URL raw", placeholder="https://raw.githubusercontent.com/...")

    st.markdown("---")
    st.markdown("### 🎛️ Filtros")
    year_filter = None
    doc_type_filter = []
    min_cites = 0

    st.markdown("---")
    st.markdown(f'<div style="font-size:0.73rem;color:{PALETTE["muted"]}">Grupo 11 · ISIL 2026<br>PA3 · StartIA · Scopus</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════
df_raw = None

if source_mode == "✅  Dataset incluido":
    df_raw = get_embedded()
    st.toast(f"Dataset Scopus cargado · {len(df_raw)} artículos", icon="✅")
elif source_mode == "📁  Cargar CSV local":
    if uploaded_file:
        df_raw = load_csv(uploaded_file)
        st.toast(f"CSV cargado · {len(df_raw)} artículos", icon="✅")
    else:
        st.info("⬅️  Sube tu CSV de Scopus en el panel lateral.")
        st.stop()
elif source_mode == "🔗  URL de GitHub":
    if github_url and github_url.startswith("http"):
        try:
            df_raw = pd.read_csv(github_url, encoding="utf-8-sig")
            st.toast(f"GitHub cargado · {len(df_raw)} artículos", icon="✅")
        except Exception as e:
            st.error(f"No se pudo cargar: {e}")
            st.stop()
    else:
        st.info("⬅️  Ingresa una URL raw de GitHub.")
        st.stop()

if df_raw is None:
    st.stop()

# ── Dynamic sidebar filters ──
with st.sidebar:
    if "Year" in df_raw.columns:
        yrs = df_raw["Year"].dropna().astype(int)
        if not yrs.empty:
            ymi, yma = int(yrs.min()), int(yrs.max())
            year_filter = st.slider("Rango de años", ymi, yma, (ymi, yma)) if ymi < yma else (ymi, yma)
    if "Document Type" in df_raw.columns:
        dtypes = sorted(df_raw["Document Type"].dropna().unique().tolist())
        doc_type_filter = st.multiselect("Tipo de documento", dtypes, default=dtypes)
    if "Cited by" in df_raw.columns:
        mc = int(df_raw["Cited by"].max())
        if mc > 0:
            min_cites = st.slider("Mínimo de citas", 0, mc, 0)

# Apply filters
df = df_raw.copy()
if year_filter and "Year" in df.columns:
    df = df[(df["Year"] >= year_filter[0]) & (df["Year"] <= year_filter[1])]
if doc_type_filter and "Document Type" in df.columns:
    df = df[df["Document Type"].isin(doc_type_filter)]
if min_cites and "Cited by" in df.columns:
    df = df[df["Cited by"] >= min_cites]

if df.empty:
    st.warning("No hay artículos con los filtros actuales.")
    st.stop()

# ═══════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">⚽ Análisis Bibliométrico · Scopus · 2026</div>
  <div class="hero-title">Machine Learning &amp; <span class="accent">Rendimiento Deportivo</span></div>
  <div class="hero-sub">
    Revisión sistemática sobre la aplicación de ML en la predicción de lesiones,
    optimización del rendimiento y toma de decisiones en el deporte profesional.
  </div>
  <div class="hero-meta">
    <div class="hero-meta-item">📚 Fuente: <span>Scopus</span></div>
    <div class="hero-meta-item">📅 Período: <span>2026</span></div>
    <div class="hero-meta-item">🏫 <span>Grupo 11 · ISIL</span></div>
    <div class="hero-meta-item">🔬 PA3 · <span>StartIA</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  RESEARCH QUESTION + KEYWORDS
# ═══════════════════════════════════════════════════════════════
col_rq, col_kw = st.columns([3, 2])

with col_rq:
    st.markdown("""
    <div class="rq-card">
      <div class="rq-label">🔬 Pregunta de Investigación — Grupo 11</div>
      <div class="rq-text">
        ¿Cómo se utiliza el <strong>Machine Learning</strong> para predecir el
        <strong>rendimiento deportivo</strong> y reducir el riesgo de
        <strong>lesiones</strong> en jugadores de fútbol profesional?
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_kw:
    st.markdown(f"""
    <div style="background:{PALETTE['surface']};border:1px solid {PALETTE['border']};
                border-radius:12px;padding:20px 22px;height:100%;">
      <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
                  text-transform:uppercase;color:{PALETTE['muted']};margin-bottom:12px;">
        Keywords de búsqueda
      </div>
      <div class="kw-wrap">
        <div class="kw-pill">Machine Learning</div>
        <div class="kw-pill">Soccer</div>
        <div class="kw-pill">Injury Prevention</div>
        <div class="kw-pill">Performance</div>
      </div>
      <div style="font-size:0.78rem;color:{PALETTE['muted']};margin-top:12px;line-height:1.5;">
        Búsqueda: <code style="color:{PALETTE['cyan']};font-size:0.75rem;">
        "Machine Learning" AND "Soccer" AND "Injury Prevention" AND "Performance"</code>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  KPI CARDS
# ═══════════════════════════════════════════════════════════════
total     = len(df)
citas     = int(df["Cited by"].sum()) if "Cited by" in df.columns else 0
autores_u = len(set(a for cell in df.get("Authors", pd.Series(dtype=str)).fillna("") for a in parse_authors(cell))) if "Authors" in df.columns else 0
fuentes_u = df["Source"].nunique() if "Source" in df.columns else 0
avg_cites = round(citas / total, 1) if total > 0 else 0

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card kpi-blue">
    <div class="kpi-val">{total}</div>
    <div class="kpi-lbl">Artículos</div>
    <div class="kpi-delta">↑ dataset activo</div>
  </div>
  <div class="kpi-card kpi-green">
    <div class="kpi-val">{citas}</div>
    <div class="kpi-lbl">Citas totales</div>
    <div class="kpi-delta">~{avg_cites} por artículo</div>
  </div>
  <div class="kpi-card kpi-purple">
    <div class="kpi-val">{autores_u}</div>
    <div class="kpi-lbl">Autores únicos</div>
    <div class="kpi-delta">colaboración global</div>
  </div>
  <div class="kpi-card kpi-orange">
    <div class="kpi-val">{fuentes_u}</div>
    <div class="kpi-lbl">Revistas</div>
    <div class="kpi-delta">interdisciplinar</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Tendencias",
    "👥  Autores & Fuentes",
    "📝  Contenido",
    "🔬  Análisis Avanzado",
    "📋  Dataset",
])

# ─── TAB 1: TENDENCIAS ───────────────────────────────────────
with tab1:
    def sec(label, icon=""):
        st.markdown(f"""
        <div class="sec-hdr">
          <div class="sec-hdr-dot"></div>
          <div class="sec-hdr-text">{icon} {label}</div>
          <div class="sec-hdr-line"></div>
        </div>""", unsafe_allow_html=True)

    sec("Producción por año", "📅")
    c1, c2 = st.columns([2, 1])
    with c1:
        if "Year" in df.columns:
            by_year = df.groupby("Year").size().reset_index(name="Artículos")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=by_year["Year"], y=by_year["Artículos"],
                marker=dict(
                    color=by_year["Artículos"],
                    colorscale=[[0,"#1C2B3A"],[1,"#58A6FF"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>%{y} artículos<extra></extra>",
            ))
            fig.update_layout(**plotly_theme(), title="Publicaciones por año",
                              showlegend=False, bargap=0.25)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "Document Type" in df.columns:
            dt_counts = df["Document Type"].value_counts().reset_index()
            dt_counts.columns = ["Tipo", "N"]
            fig2 = go.Figure(go.Pie(
                labels=dt_counts["Tipo"], values=dt_counts["N"],
                hole=0.55,
                marker=dict(colors=[PALETTE["blue"], PALETTE["green"], PALETTE["purple"],
                                    PALETTE["orange"], PALETTE["red"]]),
                textinfo="percent+label",
                textfont=dict(size=11),
                hovertemplate="<b>%{label}</b><br>%{value} artículos<br>%{percent}<extra></extra>",
            ))
            fig2.update_layout(**plotly_theme(), title="Tipo de documento",
                               showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="insight">💡 La concentración en 2026 confirma que este campo está en su punto de mayor actividad investigativa. La mayoría de los estudios son artículos originales, lo que indica evidencia primaria sólida.</div>', unsafe_allow_html=True)

    sec("Artículos más citados", "🏆")
    if "Cited by" in df.columns and "Title" in df.columns:
        top_cited = df.nlargest(10, "Cited by").copy()
        top_cited["Título_corto"] = top_cited["Title"].str[:60] + "…"
        fig3 = go.Figure(go.Bar(
            x=top_cited["Cited by"],
            y=top_cited["Título_corto"],
            orientation="h",
            marker=dict(
                color=top_cited["Cited by"],
                colorscale=[[0,"#2D1B45"],[1,"#D2A8FF"]],
                line=dict(width=0),
            ),
            customdata=top_cited[["Title","Year","Source"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>Año: %{customdata[1]}<br>Revista: %{customdata[2]}<br>Citas: %{x}<extra></extra>",
        ))
        fig3.update_layout(**plotly_theme(), title="Top 10 artículos por número de citas",
                           yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                           height=380)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('<div class="insight insight-purple">💡 Los artículos con mayor impacto combinan modelos de ML con datos de fútbol profesional, evidenciando la convergencia entre ciencias del deporte e inteligencia artificial.</div>', unsafe_allow_html=True)


# ─── TAB 2: AUTORES & FUENTES ────────────────────────────────
with tab2:
    def sec2(label, icon=""):
        st.markdown(f"""
        <div class="sec-hdr">
          <div class="sec-hdr-dot"></div>
          <div class="sec-hdr-text">{icon} {label}</div>
          <div class="sec-hdr-line"></div>
        </div>""", unsafe_allow_html=True)

    sec2("Autores más productivos", "✍️")
    c1, c2 = st.columns(2)
    with c1:
        if "Authors" in df.columns:
            all_auth = []
            for cell in df["Authors"].fillna(""):
                all_auth.extend(parse_authors(cell))
            auth_counts = Counter(all_auth).most_common(15)
            if auth_counts:
                adf = pd.DataFrame(auth_counts, columns=["Autor","N"])
                fig4 = go.Figure(go.Bar(
                    x=adf["N"], y=adf["Autor"],
                    orientation="h",
                    marker=dict(
                        color=adf["N"],
                        colorscale=[[0,"#1A2E1A"],[1,"#3FB950"]],
                        line=dict(width=0),
                    ),
                    hovertemplate="<b>%{y}</b><br>%{x} artículos<extra></extra>",
                ))
                fig4.update_layout(**plotly_theme(), title="Top 15 autores",
                                   yaxis=dict(autorange="reversed"), height=420)
                st.plotly_chart(fig4, use_container_width=True)

    with c2:
        if "Source" in df.columns:
            src_counts = df["Source"].value_counts().head(10).reset_index()
            src_counts.columns = ["Revista","N"]
            fig5 = go.Figure(go.Bar(
                x=src_counts["N"], y=src_counts["Revista"],
                orientation="h",
                marker=dict(
                    color=src_counts["N"],
                    colorscale=[[0,"#1C2B3A"],[1,"#58A6FF"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{y}</b><br>%{x} artículos<extra></extra>",
            ))
            fig5.update_layout(**plotly_theme(), title="Top 10 revistas",
                               yaxis=dict(autorange="reversed"), height=420)
            st.plotly_chart(fig5, use_container_width=True)

    sec2("Colaboración por revista", "🤝")
    if "Source" in df.columns and "Cited by" in df.columns:
        src_summary = df.groupby("Source").agg(
            Artículos=("Title","count"),
            Citas=("Cited by","sum"),
        ).reset_index()
        src_summary["Citas por artículo"] = (src_summary["Citas"] / src_summary["Artículos"]).round(2)
        src_summary = src_summary.sort_values("Artículos", ascending=False).head(12)
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            name="Artículos",
            x=src_summary["Source"], y=src_summary["Artículos"],
            marker_color=PALETTE["blue"], marker_line_width=0,
        ))
        fig6.add_trace(go.Bar(
            name="Citas totales",
            x=src_summary["Source"], y=src_summary["Citas"],
            marker_color=PALETTE["green"], marker_line_width=0,
        ))
        fig6.update_layout(**plotly_theme(),
                           title="Artículos vs Citas por revista",
                           barmode="group",
                           xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
                           legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown('<div class="insight insight-green">💡 Scientific Reports domina el corpus — es una revista de alto impacto de Nature Portfolio, lo que valida la relevancia del tema a nivel global.</div>', unsafe_allow_html=True)


# ─── TAB 3: CONTENIDO ────────────────────────────────────────
with tab3:
    def sec3(label, icon=""):
        st.markdown(f"""
        <div class="sec-hdr">
          <div class="sec-hdr-dot"></div>
          <div class="sec-hdr-text">{icon} {label}</div>
          <div class="sec-hdr-line"></div>
        </div>""", unsafe_allow_html=True)

    sec3("Mapa de palabras — Abstracts", "☁️")
    if "Abstract" in df.columns:
        all_text = " ".join(df["Abstract"].dropna().astype(str))
        words = clean_words(all_text)
        if words:
            freq = Counter(words)
            wc = WordCloud(
                width=1100, height=420,
                background_color=PALETTE["surface"],
                colormap="Blues",
                max_words=100,
                prefer_horizontal=0.75,
                min_font_size=10,
                max_font_size=90,
                relative_scaling=0.5,
            ).generate_from_frequencies(freq)
            fig_wc, ax = plt.subplots(figsize=(11, 4.2))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig_wc.patch.set_facecolor(PALETTE["surface"])
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)
            st.markdown('<div class="insight">💡 Los términos más frecuentes confirman que el foco del corpus está en predicción de lesiones, optimización de carga de entrenamiento e IA aplicada al deporte. Los abstracts validan directamente las 4 keywords de búsqueda.</div>', unsafe_allow_html=True)

    sec3("Top Keywords de autores", "🔑")
    if "Author Keywords" in df.columns:
        all_kw = []
        for cell in df["Author Keywords"].fillna(""):
            all_kw.extend(parse_keywords(cell))
        if all_kw:
            kw_counts = Counter(all_kw).most_common(25)
            kw_df = pd.DataFrame(kw_counts, columns=["Keyword","N"])
            c1, c2 = st.columns([3, 2])
            with c1:
                fig7 = go.Figure(go.Bar(
                    x=kw_df["N"], y=kw_df["Keyword"],
                    orientation="h",
                    marker=dict(
                        color=kw_df["N"],
                        colorscale=[[0,"#1C2B3A"],[1,"#79C0FF"]],
                        line=dict(width=0),
                    ),
                    hovertemplate="<b>%{y}</b><br>%{x} veces<extra></extra>",
                ))
                fig7.update_layout(**plotly_theme(),
                                   title="Top 25 keywords de autores",
                                   yaxis=dict(autorange="reversed"),
                                   height=520)
                st.plotly_chart(fig7, use_container_width=True)
            with c2:
                # Treemap of top keywords
                top15_kw = kw_df.head(15)
                fig8 = go.Figure(go.Treemap(
                    labels=top15_kw["Keyword"],
                    parents=[""] * len(top15_kw),
                    values=top15_kw["N"],
                    marker=dict(
                        colors=top15_kw["N"],
                        colorscale=[[0,"#1C2B3A"],[0.5,"#388BFD"],[1,"#79C0FF"]],
                        showscale=False,
                    ),
                    textfont=dict(size=12, color="white"),
                    hovertemplate="<b>%{label}</b><br>%{value} apariciones<extra></extra>",
                ))
                fig8.update_layout(**plotly_theme(), title="Mapa de keywords",
                                   margin=dict(t=44,b=4,l=4,r=4))
                st.plotly_chart(fig8, use_container_width=True)


# ─── TAB 4: ANÁLISIS AVANZADO ────────────────────────────────
with tab4:
    def sec4(label, icon=""):
        st.markdown(f"""
        <div class="sec-hdr">
          <div class="sec-hdr-dot"></div>
          <div class="sec-hdr-text">{icon} {label}</div>
          <div class="sec-hdr-line"></div>
        </div>""", unsafe_allow_html=True)

    sec4("Distribución de impacto científico", "📊")
    c1, c2 = st.columns(2)
    with c1:
        if "Cited by" in df.columns:
            fig9 = go.Figure()
            fig9.add_trace(go.Histogram(
                x=df["Cited by"], nbinsx=12,
                marker=dict(color=PALETTE["blue"], line=dict(width=0)),
                opacity=0.85,
                hovertemplate="Citas: %{x}<br>Artículos: %{y}<extra></extra>",
            ))
            fig9.update_layout(**plotly_theme(),
                               title="Distribución de citas por artículo",
                               xaxis_title="Número de citas",
                               yaxis_title="Artículos",
                               bargap=0.08)
            st.plotly_chart(fig9, use_container_width=True)

    with c2:
        if "Cited by" in df.columns:
            bins   = [0, 1, 5, 20, 999]
            labels = ["0 citas", "1–5", "6–20", "21+"]
            df["bracket"] = pd.cut(df["Cited by"], bins=bins, labels=labels, right=True)
            br = df["bracket"].value_counts().sort_index().reset_index()
            br.columns = ["Rango","N"]
            fig10 = go.Figure(go.Pie(
                labels=br["Rango"], values=br["N"],
                hole=0.5,
                marker=dict(colors=[PALETTE["border2"], PALETTE["blue"],
                                    PALETTE["green"], PALETTE["purple"]]),
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>%{value} artículos (%{percent})<extra></extra>",
            ))
            fig10.update_layout(**plotly_theme(), title="Impacto por rango de citas",
                                showlegend=False)
            st.plotly_chart(fig10, use_container_width=True)

    sec4("Comparativa por tipo de documento", "📋")
    if "Document Type" in df.columns and "Cited by" in df.columns:
        dt_impact = df.groupby("Document Type").agg(
            Artículos=("Title","count"),
            Citas_totales=("Cited by","sum"),
            Citas_promedio=("Cited by","mean"),
        ).reset_index().round(2)

        fig11 = go.Figure()
        fig11.add_trace(go.Bar(
            name="Artículos",
            x=dt_impact["Document Type"],
            y=dt_impact["Artículos"],
            marker_color=PALETTE["blue"], marker_line_width=0,
        ))
        fig11.add_trace(go.Bar(
            name="Citas totales",
            x=dt_impact["Document Type"],
            y=dt_impact["Citas_totales"],
            marker_color=PALETTE["green"], marker_line_width=0,
        ))
        fig11.add_trace(go.Scatter(
            name="Citas promedio",
            x=dt_impact["Document Type"],
            y=dt_impact["Citas_promedio"],
            mode="markers+lines",
            marker=dict(color=PALETTE["orange"], size=10),
            line=dict(color=PALETTE["orange"], width=2, dash="dot"),
            yaxis="y2",
        ))
        fig11.update_layout(
            **plotly_theme(),
            title="Volumen e impacto por tipo de documento",
            barmode="group",
            yaxis2=dict(
                overlaying="y", side="right",
                showgrid=False,
                title="Citas promedio",
                titlefont=dict(color=PALETTE["orange"]),
                tickfont=dict(color=PALETTE["orange"]),
            ),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig11, use_container_width=True)

    sec4("Red de co-autoría (Top colaboradores)", "🕸️")
    if "Authors" in df.columns:
        # Build co-authorship edge counts
        co_pairs = Counter()
        for cell in df["Authors"].fillna(""):
            auths = parse_authors(cell)
            if len(auths) >= 2:
                for i in range(len(auths)):
                    for j in range(i+1, len(auths)):
                        pair = tuple(sorted([auths[i], auths[j]]))
                        co_pairs[pair] += 1

        # Top authors by frequency
        all_auth2 = []
        for cell in df["Authors"].fillna(""):
            all_auth2.extend(parse_authors(cell))
        top_auth = [a for a, _ in Counter(all_auth2).most_common(12)]

        # Filter edges involving top authors
        edges = [(a, b, w) for (a, b), w in co_pairs.items() if a in top_auth or b in top_auth]

        if edges:
            # Simple force-like layout using circle positions
            import math
            n = len(top_auth)
            pos = {a: (math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n))
                   for i, a in enumerate(top_auth)}
            node_sizes = [Counter(all_auth2).get(a, 1) * 8 + 12 for a in top_auth]

            edge_x, edge_y = [], []
            for a, b, w in edges:
                if a in pos and b in pos:
                    x0,y0 = pos[a]; x1,y1 = pos[b]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

            fig12 = go.Figure()
            fig12.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(color=PALETTE["border2"], width=1),
                hoverinfo="none",
            ))
            fig12.add_trace(go.Scatter(
                x=[pos[a][0] for a in top_auth],
                y=[pos[a][1] for a in top_auth],
                mode="markers+text",
                marker=dict(
                    size=node_sizes,
                    color=[Counter(all_auth2).get(a, 1) for a in top_auth],
                    colorscale=[[0,"#1C2B3A"],[1,"#58A6FF"]],
                    line=dict(color=PALETTE["border2"], width=1),
                    showscale=False,
                ),
                text=[a.split(";")[0].split(",")[0][:18] for a in top_auth],
                textposition="top center",
                textfont=dict(size=10, color=PALETTE["muted"]),
                hovertemplate="<b>%{text}</b><br>Artículos: %{marker.color}<extra></extra>",
            ))
            fig12.update_layout(
                **plotly_theme(),
                title="Red de co-autoría — Top 12 autores",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=420,
            )
            st.plotly_chart(fig12, use_container_width=True)
            st.markdown('<div class="insight insight-purple">💡 La red muestra los patrones de colaboración entre los autores más activos. Nodos más grandes indican mayor productividad en el corpus.</div>', unsafe_allow_html=True)


# ─── TAB 5: DATASET ──────────────────────────────────────────
with tab5:
    st.markdown(f"""
    <div class="sec-hdr">
      <div class="sec-hdr-dot"></div>
      <div class="sec-hdr-text">📋 DATASET COMPLETO</div>
      <div class="sec-hdr-line"></div>
    </div>""", unsafe_allow_html=True)

    col_s, col_f = st.columns([3, 1])
    with col_s:
        search = st.text_input("🔍 Buscar en títulos o autores…",
                               placeholder="ej. injury prediction, random forest, soccer…")
    with col_f:
        sort_by = st.selectbox("Ordenar por", ["Year ↓","Cited by ↓","Title ↑"], label_visibility="visible")

    show_cols = [c for c in ["Title","Authors","Year","Source","Cited by","Document Type"] if c in df.columns]
    disp = df[show_cols].copy()

    if search:
        mask = pd.Series(False, index=disp.index)
        for col in ["Title","Authors"]:
            if col in disp.columns:
                mask |= disp[col].str.contains(search, case=False, na=False)
        disp = disp[mask]

    if sort_by == "Year ↓" and "Year" in disp.columns:
        disp = disp.sort_values("Year", ascending=False)
    elif sort_by == "Cited by ↓" and "Cited by" in disp.columns:
        disp = disp.sort_values("Cited by", ascending=False)
    elif sort_by == "Title ↑" and "Title" in disp.columns:
        disp = disp.sort_values("Title")

    st.markdown(f'<div style="font-size:0.8rem;color:{PALETTE["muted"]};margin-bottom:8px;">{len(disp)} artículos mostrados</div>', unsafe_allow_html=True)
    st.dataframe(disp, use_container_width=True, height=400)

    c1, c2 = st.columns(2)
    with c1:
        csv_dl = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️  Descargar dataset completo (CSV)",
                           csv_dl, "scopus_ml_sports_grupo11.csv", "text/csv")
    with c2:
        if len(disp) < len(df):
            csv_filt = disp.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Descargar resultados filtrados",
                               csv_filt, "scopus_filtrado.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
  <div class="footer-text">
    Grupo 11 · PA3 · StartIA · ISIL 2026<br>
    Análisis bibliométrico sobre Machine Learning aplicado al deporte profesional · Fuente: Scopus
  </div>
  <div class="footer-kws">
    <span class="footer-kw">Machine Learning</span>
    <span class="footer-kw">·</span>
    <span class="footer-kw">Soccer</span>
    <span class="footer-kw">·</span>
    <span class="footer-kw">Injury Prevention</span>
    <span class="footer-kw">·</span>
    <span class="footer-kw">Performance</span>
  </div>
</div>
""", unsafe_allow_html=True)
PYEOF
echo "Done — lines: $(wc -l < /mnt/user-data/outputs/pa3_dashboard/app.py)"

# ================================================================
# BehaviourIQ — Dynamic Streamlit Dashboard
# 
# HOW TO RUN:
#   pip install streamlit plotly pandas numpy scikit-learn
#   streamlit run dashboard.py
#
# This file runs the COMPLETE notebook pipeline from scratch:
#   1. generate_behavioural_dataset()  — exact Cell 2 code
#   2. engineer_features()             — exact Cell 7 code
#   3. run_models()                    — exact Cells 8, 9, 10 code
#
# Every number in every chart is computed live.
# Nothing is hardcoded. Change any sidebar slider → everything recomputes.
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score
)
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="BehaviourIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0D1B2A; color: #E8F4F8; }
    .main .block-container { padding-top: 1.2rem; }
    h1, h2, h3, h4 { color: #00C2CB !important; }
    div[data-testid="stMetric"] {
        background: #132236;
        border-radius: 10px;
        padding: 14px 16px;
        border-top: 3px solid #00C2CB;
    }
    div[data-testid="stMetricValue"] { color: #00C2CB !important; font-family: monospace; }
    div[data-testid="stMetricLabel"] { color: #7FA8BE !important; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
    section[data-testid="stSidebar"] { background-color: #132236 !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #132236; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #7FA8BE; }
    .stTabs [aria-selected="true"] { color: #00C2CB !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# STEP 1 — DATA GENERATION
# Exact replica of notebook Cell 2: generate_behavioural_dataset()
# ================================================================

@st.cache_data(show_spinner=False)
def generate_dataset(n_employees, n_days, anomaly_rate, seed):
    np.random.seed(seed)
    records = []

    personas         = ['normal', 'overworker', 'disengaged', 'night_owl']
    persona_weights  = [0.5, 0.2, 0.15, 0.15]
    persona_profiles = {
        'normal':     {'login_hour_mean': 9.0,  'login_hour_std': 0.5,  'hours_worked_mean': 8.0,  'hours_worked_std': 0.7,  'emails_mean': 25, 'files_downloaded_mean': 5,  'apps_accessed_mean': 8},
        'overworker': {'login_hour_mean': 7.5,  'login_hour_std': 1.0,  'hours_worked_mean': 11.0, 'hours_worked_std': 1.2,  'emails_mean': 55, 'files_downloaded_mean': 12, 'apps_accessed_mean': 14},
        'disengaged': {'login_hour_mean': 10.0, 'login_hour_std': 1.5,  'hours_worked_mean': 5.5,  'hours_worked_std': 1.5,  'emails_mean': 8,  'files_downloaded_mean': 2,  'apps_accessed_mean': 4},
        'night_owl':  {'login_hour_mean': 14.0, 'login_hour_std': 1.0,  'hours_worked_mean': 8.5,  'hours_worked_std': 0.8,  'emails_mean': 20, 'files_downloaded_mean': 6,  'apps_accessed_mean': 9},
    }

    start_date         = datetime(2024, 1, 2)
    departments        = ['Engineering', 'Sales', 'HR', 'Finance', 'Marketing', 'Operations']
    anomaly_types_list = ['data_exfil', 'burnout', 'account_comp', 'disengagement']

    employee_personas = {eid: np.random.choice(personas, p=persona_weights) for eid in range(n_employees)}
    employee_depts    = {eid: np.random.choice(departments)                  for eid in range(n_employees)}

    for day_idx in range(n_days):
        current_date = start_date + timedelta(days=day_idx)
        is_weekend   = current_date.weekday() >= 5
        dow          = current_date.strftime('%A')

        for eid in range(n_employees):
            persona = employee_personas[eid]
            profile = persona_profiles[persona]
            dept    = employee_depts[eid]

            if is_weekend and np.random.rand() < (0.95 if persona != 'overworker' else 0.5):
                continue

            is_anomaly   = 0
            anomaly_type = 'normal'

            if np.random.rand() < anomaly_rate:
                anomaly_type = np.random.choice(anomaly_types_list)
                is_anomaly   = 1

                if anomaly_type == 'data_exfil':
                    login_hour       = np.random.uniform(0, 5)
                    hours_worked     = np.random.uniform(1, 4)
                    emails_sent      = int(np.random.normal(profile['emails_mean'], 2))
                    files_downloaded = int(np.random.uniform(50, 150))
                    apps_accessed    = int(np.random.uniform(20, 35))
                    tasks_completed  = int(np.random.uniform(0, 2))
                    meeting_hours    = 0

                elif anomaly_type == 'burnout':
                    login_hour       = np.random.uniform(11, 14)
                    hours_worked     = np.random.uniform(1, 3)
                    emails_sent      = int(np.random.uniform(0, 3))
                    files_downloaded = int(np.random.uniform(0, 2))
                    apps_accessed    = int(np.random.uniform(1, 3))
                    tasks_completed  = int(np.random.uniform(0, 1))
                    meeting_hours    = 0

                elif anomaly_type == 'account_comp':
                    login_hour       = np.random.uniform(1, 4)
                    hours_worked     = np.random.uniform(6, 14)
                    emails_sent      = int(np.random.normal(profile['emails_mean'] * 3, 5))
                    files_downloaded = int(np.random.uniform(30, 80))
                    apps_accessed    = int(np.random.uniform(25, 40))
                    tasks_completed  = int(np.random.uniform(0, 3))
                    meeting_hours    = 0

                elif anomaly_type == 'disengagement':
                    login_hour       = np.random.normal(profile['login_hour_mean'] + 2, 1)
                    hours_worked     = max(0.5, np.random.uniform(1, 3))
                    emails_sent      = int(np.random.uniform(0, 4))
                    files_downloaded = int(np.random.uniform(0, 2))
                    apps_accessed    = int(np.random.uniform(1, 4))
                    tasks_completed  = int(np.random.uniform(0, 1))
                    meeting_hours    = np.random.uniform(0, 0.5)
            else:
                login_hour       = max(0, min(23, np.random.normal(profile['login_hour_mean'], profile['login_hour_std'])))
                hours_worked     = max(1, min(16, np.random.normal(profile['hours_worked_mean'], profile['hours_worked_std'])))
                emails_sent      = max(0, int(np.random.poisson(profile['emails_mean'])))
                files_downloaded = max(0, int(np.random.poisson(profile['files_downloaded_mean'])))
                apps_accessed    = max(1, int(np.random.normal(profile['apps_accessed_mean'], 2)))
                tasks_completed  = max(0, int(np.random.poisson(4 if persona != 'disengaged' else 1.5)))
                meeting_hours    = max(0, np.random.normal(1.5 if persona == 'normal' else 2.5, 0.5))

            logout_hour        = min(23.99, login_hour + hours_worked)
            is_off_hours       = 1 if (login_hour < 7 or login_hour > 20) else 0
            productivity_ratio = tasks_completed / max(1, hours_worked)
            email_intensity    = emails_sent / max(1, hours_worked)
            download_intensity = files_downloaded / max(1, hours_worked)

            records.append({
                'employee_id':       f'EMP_{eid:04d}',
                'date':              current_date.strftime('%Y-%m-%d'),
                'day_of_week':       dow,
                'department':        dept,
                'persona':           persona,
                'login_hour':        round(login_hour, 2),
                'logout_hour':       round(logout_hour, 2),
                'hours_worked':      round(hours_worked, 2),
                'emails_sent':       emails_sent,
                'files_downloaded':  files_downloaded,
                'apps_accessed':     apps_accessed,
                'tasks_completed':   tasks_completed,
                'meeting_hours':     round(meeting_hours, 2),
                'is_off_hours':      is_off_hours,
                'is_weekend':        int(is_weekend),
                'productivity_ratio': round(productivity_ratio, 3),
                'email_intensity':   round(email_intensity, 3),
                'download_intensity': round(download_intensity, 3),
                'is_anomaly':        is_anomaly,
                'anomaly_type':      anomaly_type,
            })

    return pd.DataFrame(records)


# ================================================================
# STEP 2 — FEATURE ENGINEERING
# Exact replica of notebook Cell 7
# ================================================================

@st.cache_data(show_spinner=False)
def engineer_features(df):
    df_fe         = df.copy()
    numeric_feats = ['hours_worked', 'emails_sent', 'files_downloaded',
                     'apps_accessed', 'tasks_completed', 'login_hour']

    # Z-score per employee (personal baseline deviation)
    for feat in numeric_feats:
        emp_mean = df_fe.groupby('employee_id')[feat].transform('mean')
        emp_std  = df_fe.groupby('employee_id')[feat].transform('std').replace(0, 1)
        df_fe[f'{feat}_zscore'] = (df_fe[feat] - emp_mean) / emp_std

    # Rolling 7-day deviation
    df_fe = df_fe.sort_values(['employee_id', 'date'])
    for feat in ['hours_worked', 'files_downloaded', 'emails_sent']:
        rolling = df_fe.groupby('employee_id')[feat].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df_fe[f'{feat}_deviation_from_rolling'] = df_fe[feat] - rolling

    # Composite anomaly score
    df_fe['anomaly_score_manual'] = (
        np.abs(df_fe['files_downloaded_zscore']) * 0.35 +
        np.abs(df_fe['login_hour_zscore'])        * 0.25 +
        np.abs(df_fe['apps_accessed_zscore'])     * 0.20 +
        np.abs(df_fe['hours_worked_zscore'])      * 0.10 +
        df_fe['is_off_hours']                     * 0.10
    )

    dow_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df_fe['day_of_week_num'] = df_fe['day_of_week'].map(dow_map)

    le = LabelEncoder()
    df_fe['dept_encoded'] = le.fit_transform(df_fe['department'])

    return df_fe


# ================================================================
# STEP 3 — MODELS + RISK PROFILING + PCA
# Exact replica of notebook Cells 8, 9, 10, 13
# ================================================================

FEATURE_COLS = [
    'login_hour', 'hours_worked', 'emails_sent', 'files_downloaded',
    'apps_accessed', 'tasks_completed', 'meeting_hours',
    'is_off_hours', 'is_weekend', 'productivity_ratio',
    'email_intensity', 'download_intensity',
    'hours_worked_zscore', 'files_downloaded_zscore',
    'login_hour_zscore', 'apps_accessed_zscore',
    'anomaly_score_manual', 'day_of_week_num', 'dept_encoded'
]

@st.cache_data(show_spinner=False)
def run_models(df_fe, n_est_if, contamination, n_est_rf, max_depth):
    X        = df_fe[FEATURE_COLS].fillna(0)
    y        = df_fe['is_anomaly']
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Isolation Forest (Cell 8) ─────────────────────────────────
    iso        = IsolationForest(n_estimators=n_est_if, contamination=contamination,
                                 max_features=0.8, random_state=42)
    iso_pred   = iso.fit_predict(X_scaled)
    iso_scores = -iso.score_samples(X_scaled)
    iso_labels = (iso_pred == -1).astype(int)
    roc_iso    = roc_auc_score(y, iso_scores)
    ap_iso     = average_precision_score(y, iso_scores)
    fpr_iso, tpr_iso, _ = roc_curve(y, iso_scores)

    # ── LOF (Cell 9) ─────────────────────────────────────────────
    lof        = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_pred   = lof.fit_predict(X_scaled)
    lof_scores = -lof.negative_outlier_factor_
    lof_labels = (lof_pred == -1).astype(int)
    roc_lof    = roc_auc_score(y, lof_scores)
    ap_lof     = average_precision_score(y, lof_scores)
    fpr_lof, tpr_lof, _ = roc_curve(y, lof_scores)

    # ── Random Forest (Cell 10) ───────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    rf      = RandomForestClassifier(n_estimators=n_est_rf, max_depth=max_depth,
                                     min_samples_leaf=5, class_weight='balanced',
                                     random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred    = rf.predict(X_te)
    y_proba   = rf.predict_proba(X_te)[:, 1]
    roc_rf    = roc_auc_score(y_te, y_proba)
    ap_rf     = average_precision_score(y_te, y_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_te, y_proba)
    cm        = confusion_matrix(y_te, y_pred)
    cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='roc_auc')
    feat_imp  = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

    # ── Ensemble + Risk Profiling (Cell 13) ──────────────────────
    df_out              = df_fe.copy()
    df_out['rf_score']  = rf.predict_proba(X_scaled)[:, 1]
    df_out['if_score']  = iso_scores
    if_min, if_max      = iso_scores.min(), iso_scores.max()
    df_out['if_norm']   = (iso_scores - if_min) / (if_max - if_min)
    df_out['ensemble']  = (df_out['rf_score'] + df_out['if_norm']) / 2

    emp_risk = df_out.groupby('employee_id').agg(
        avg_risk=('ensemble', 'mean'),
        max_risk=('ensemble', 'max'),
        anomaly_days=('is_anomaly', 'sum'),
        total_days=('is_anomaly', 'count'),
        department=('department', 'first'),
        persona=('persona', 'first'),
    ).reset_index()
    emp_risk['anomaly_rate'] = emp_risk['anomaly_days'] / emp_risk['total_days']
    emp_risk['risk_tier']    = pd.cut(
        emp_risk['avg_risk'],
        bins=[0, 0.2, 0.4, 0.6, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

    # ── PCA (Cell 12) ─────────────────────────────────────────────
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return {
        'df_out':     df_out,
        'y':          y,
        'y_te':       y_te,
        'y_pred':     y_pred,
        'y_proba':    y_proba,
        'iso_labels': iso_labels,
        'lof_labels': lof_labels,
        'iso_scores': iso_scores,
        'lof_scores': lof_scores,
        'roc_iso': roc_iso, 'ap_iso': ap_iso,
        'roc_lof': roc_lof, 'ap_lof': ap_lof,
        'roc_rf':  roc_rf,  'ap_rf':  ap_rf,
        'fpr_iso': fpr_iso, 'tpr_iso': tpr_iso,
        'fpr_lof': fpr_lof, 'tpr_lof': tpr_lof,
        'fpr_rf':  fpr_rf,  'tpr_rf':  tpr_rf,
        'cm':        cm,
        'cv_scores': cv_scores,
        'feat_imp':  feat_imp,
        'emp_risk':  emp_risk,
        'X_pca':     X_pca,
        'pca':       pca,
    }


# ================================================================
# SIDEBAR — all live controls
# ================================================================

with st.sidebar:
    st.markdown("## 🧠 BehaviourIQ")
    st.markdown("*Digital Anomaly Detection*")
    st.divider()

    st.markdown("### 📋 Dataset")
    n_employees  = st.slider("Employees",       50,  500, 200, step=50)
    n_days       = st.slider("Days",            20,  100,  50, step=5)
    anomaly_rate = st.slider("Anomaly Rate",  0.02, 0.20, 0.08, step=0.01, format="%.2f")
    seed         = st.number_input("Random Seed", 0, 9999, 42)

    st.divider()
    st.markdown("### 🤖 Model Parameters")
    n_est_if   = st.slider("IF: n_estimators",  50, 300, 200, step=50)
    contam     = st.slider("IF: contamination", 0.02, 0.20, 0.08, step=0.01, format="%.2f")
    n_est_rf   = st.slider("RF: n_estimators",  50, 500, 300, step=50)
    max_depth  = st.slider("RF: max_depth",      3,  20,  10)

    st.divider()
    st.markdown("### 🔍 Filters")
    dept_filter = st.multiselect(
        "Departments",
        ['Engineering', 'Sales', 'HR', 'Finance', 'Marketing', 'Operations'],
        default=['Engineering', 'Sales', 'HR', 'Finance', 'Marketing', 'Operations']
    )
    persona_filter = st.multiselect(
        "Personas",
        ['normal', 'overworker', 'disengaged', 'night_owl'],
        default=['normal', 'overworker', 'disengaged', 'night_owl']
    )

    st.divider()
    st.caption("Every chart recomputes live when you change any parameter.")


# ================================================================
# RUN PIPELINE
# ================================================================

with st.spinner("🔄 Running pipeline: generating data → feature engineering → training models..."):
    df    = generate_dataset(n_employees, n_days, anomaly_rate, int(seed))
    df_fe = engineer_features(df)
    R     = run_models(df_fe, n_est_if, contam, n_est_rf, max_depth)

# Apply filters to raw df only (model results use full data)
df_f = df[df['department'].isin(dept_filter) & df['persona'].isin(persona_filter)]
emp_risk_f = R['emp_risk'][
    R['emp_risk']['department'].isin(dept_filter) &
    R['emp_risk']['persona'].isin(persona_filter)
]

# ── Shared plotly theme ───────────────────────────────────────────
BG = dict(paper_bgcolor='#132236', plot_bgcolor='#0D1B2A',
          font=dict(color='#B0C9D8', size=11),
          margin=dict(l=40, r=20, t=40, b=40))
CYAN   = '#00C2CB'
AMBER  = '#F5A623'
RED    = '#E84855'
GREEN  = '#2ECC71'
PURPLE = '#9B59B6'
ANOM_C = {'data_exfil': RED, 'burnout': AMBER,
          'account_comp': CYAN, 'disengagement': '#7FA8BE', 'normal': GREEN}


# ================================================================
# HEADER + KPI STRIP
# ================================================================

st.markdown("# 🧠 BehaviourIQ — Digital Behaviour Anomaly Detection")
st.markdown(
    f"*{n_employees} employees · {n_days} days · {anomaly_rate*100:.0f}% anomaly rate · seed={seed} "
    f"· All values computed live — nothing is hardcoded*"
)
st.divider()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Records",   f"{len(df):,}")
c2.metric("Employees",       f"{n_employees}")
c3.metric("True Anomalies",  f"{df['is_anomaly'].sum():,}",
          delta=f"{df['is_anomaly'].mean()*100:.1f}%", delta_color="off")
c4.metric("IF  ROC-AUC",     f"{R['roc_iso']:.4f}")
c5.metric("RF  ROC-AUC",     f"{R['roc_rf']:.4f}")
c6.metric("Medium+ Flagged", f"{(R['emp_risk']['risk_tier'].isin(['Medium','High','Critical'])).sum()}")

st.markdown("")


# ================================================================
# TABS
# ================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🤖 Models",
    "🔧 Features",
    "⚠️ Risk Profiling",
    "📈 Timeline",
    "🔬 PCA"
])


# ── TAB 1 — OVERVIEW ─────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Anomaly Type Distribution")
        anom_counts = (df_f[df_f['is_anomaly'] == 1]['anomaly_type']
                       .value_counts().reset_index())
        anom_counts.columns = ['Anomaly Type', 'Count']
        fig = px.bar(anom_counts, x='Anomaly Type', y='Count',
                     color='Anomaly Type',
                     color_discrete_map={k: v for k, v in ANOM_C.items()},
                     text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(**BG, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Model ROC-AUC Comparison")
        perf = pd.DataFrame({
            'Model':   ['Isolation Forest', 'LOF', 'Random Forest'],
            'ROC-AUC': [R['roc_iso'], R['roc_lof'], R['roc_rf']],
        })
        fig = px.bar(perf, x='Model', y='ROC-AUC',
                     color='Model',
                     color_discrete_sequence=[CYAN, AMBER, GREEN],
                     text=perf['ROC-AUC'].apply(lambda v: f"{v:.4f}"))
        fig.add_hline(y=0.9, line_dash='dash', line_color=RED,
                      annotation_text='0.9 threshold')
        fig.update_traces(textposition='outside')
        fig.update_layout(**BG, showlegend=False, yaxis_range=[0, 1.12])
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Anomaly Rate by Department (%)")
        dept_anom = (df_f.groupby('department')['is_anomaly']
                     .mean().mul(100).reset_index())
        dept_anom.columns = ['Department', 'Rate']
        dept_anom = dept_anom.sort_values('Rate')
        fig = px.bar(dept_anom, x='Rate', y='Department', orientation='h',
                     color='Rate', color_continuous_scale='Tealrose',
                     text=dept_anom['Rate'].apply(lambda v: f"{v:.1f}%"))
        fig.update_traces(textposition='outside')
        fig.update_layout(**BG, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### Anomaly Rate by Persona (%)")
        p_anom = (df_f.groupby('persona')['is_anomaly']
                  .mean().mul(100).reset_index())
        p_anom.columns = ['Persona', 'Rate']
        fig = px.bar(p_anom, x='Persona', y='Rate',
                     color='Persona',
                     color_discrete_sequence=[CYAN, AMBER, RED, PURPLE],
                     text=p_anom['Rate'].apply(lambda v: f"{v:.1f}%"))
        fig.update_traces(textposition='outside')
        fig.update_layout(**BG, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ── TAB 2 — MODELS ───────────────────────────────────────────────
with tab2:

    st.markdown("#### ROC Curves — All Three Models")
    fig = go.Figure()
    for name, fpr, tpr, roc, col in [
        ("Isolation Forest", R['fpr_iso'], R['tpr_iso'], R['roc_iso'], CYAN),
        ("LOF",              R['fpr_lof'], R['tpr_lof'], R['roc_lof'], AMBER),
        ("Random Forest",    R['fpr_rf'],  R['tpr_rf'],  R['roc_rf'],  GREEN),
    ]:
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f"{name}  (AUC = {roc:.4f})",
            line=dict(color=col, width=2.5)
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Baseline',
        line=dict(color='#4a6070', dash='dash', width=1)
    ))
    fig.update_layout(**BG, xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      legend=dict(bgcolor='#0D1B2A', bordercolor='#1E3A5F'))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix (Random Forest — Test Set)")
        fig = px.imshow(R['cm'],
                        labels=dict(x='Predicted', y='Actual'),
                        x=['Normal', 'Anomaly'], y=['Normal', 'Anomaly'],
                        color_continuous_scale='Teal', text_auto=True)
        fig.update_layout(**BG, coloraxis_showscale=False)
        fig.update_traces(textfont_size=20)
        st.plotly_chart(fig, use_container_width=True)

        cr = classification_report(R['y_te'], R['y_pred'],
                                   target_names=['Normal', 'Anomaly'],
                                   output_dict=True)
        st.dataframe(
            pd.DataFrame(cr).T.round(3)
            .style.background_gradient(cmap='Blues',
                                       subset=['precision', 'recall', 'f1-score']),
            use_container_width=True
        )

    with col2:
        st.markdown("#### 5-Fold Cross-Validation (Random Forest)")
        cv_df = pd.DataFrame({
            'Fold':    [f"Fold {i+1}" for i in range(len(R['cv_scores']))],
            'ROC-AUC': R['cv_scores'],
        })
        fig = px.bar(cv_df, x='Fold', y='ROC-AUC',
                     color_discrete_sequence=[GREEN],
                     text=cv_df['ROC-AUC'].apply(lambda v: f"{v:.4f}"))
        fig.add_hline(y=R['cv_scores'].mean(), line_dash='dash',
                      line_color=AMBER,
                      annotation_text=f"Mean = {R['cv_scores'].mean():.4f}")
        fig.update_traces(textposition='outside')
        fig.update_layout(**BG, showlegend=False, yaxis_range=[0.9, 1.02])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Summary Table")
        st.dataframe(pd.DataFrame({
            'Model':         ['Isolation Forest', 'LOF', 'Random Forest'],
            'Type':          ['Unsupervised', 'Unsupervised', 'Supervised'],
            'ROC-AUC':       [f"{R['roc_iso']:.4f}", f"{R['roc_lof']:.4f}", f"{R['roc_rf']:.4f}"],
            'Avg Precision': [f"{R['ap_iso']:.4f}", f"{R['ap_lof']:.4f}", f"{R['ap_rf']:.4f}"],
        }), use_container_width=True, hide_index=True)


# ── TAB 3 — FEATURES ─────────────────────────────────────────────
with tab3:
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("#### Random Forest Feature Importance")
        fi = R['feat_imp'].reset_index()
        fi.columns = ['Feature', 'Importance']
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Tealrose',
                     text=fi['Importance'].apply(lambda v: f"{v*100:.1f}%"))
        fig.update_traces(textposition='outside')
        fig.update_layout(**BG, coloraxis_showscale=False, height=520)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Persona Behaviour Heatmap")
        feat_cols = ['login_hour', 'hours_worked', 'emails_sent',
                     'files_downloaded', 'apps_accessed', 'productivity_ratio']
        persona_stats = df_f.groupby('persona')[feat_cols].mean().round(2)
        fig = px.imshow(persona_stats.T,
                        color_continuous_scale='Teal',
                        text_auto='.2f', aspect='auto',
                        labels=dict(x='Persona', y='Feature'))
        fig.update_layout(**BG, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Feature Distribution: Normal vs Anomaly")
    feat_pick = st.selectbox("Select a feature:", [
        'login_hour', 'hours_worked', 'emails_sent', 'files_downloaded',
        'apps_accessed', 'tasks_completed', 'productivity_ratio',
        'download_intensity', 'anomaly_score_manual'
    ])
    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(df_f, x=feat_pick, color='anomaly_type',
                           color_discrete_map=ANOM_C,
                           barmode='overlay', opacity=0.65, nbins=40,
                           title=f"{feat_pick} — Histogram by Anomaly Type")
        fig.update_layout(**BG)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.box(df_f, x='anomaly_type', y=feat_pick,
                     color='anomaly_type', color_discrete_map=ANOM_C,
                     title=f"{feat_pick} — Box Plot")
        fig.update_layout(**BG, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ── TAB 4 — RISK PROFILING ───────────────────────────────────────
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Employee Risk Tier Distribution")
        tier_c = R['emp_risk']['risk_tier'].value_counts().reset_index()
        tier_c.columns = ['Tier', 'Count']
        tier_colors = {'Low': GREEN, 'Medium': AMBER, 'High': RED, 'Critical': PURPLE}
        fig = px.pie(tier_c, values='Count', names='Tier',
                     color='Tier', color_discrete_map=tier_colors, hole=0.4)
        fig.update_traces(textinfo='label+percent+value',
                          pull=[0.05 if t != 'Low' else 0 for t in tier_c['Tier']])
        fig.update_layout(**BG)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Risk Score Distribution (all employees)")
        fig = px.histogram(emp_risk_f, x='avg_risk', nbins=30,
                           color_discrete_sequence=[CYAN])
        fig.add_vline(x=emp_risk_f['avg_risk'].mean(), line_dash='dash',
                      line_color=AMBER,
                      annotation_text=f"Mean = {emp_risk_f['avg_risk'].mean():.3f}")
        fig.add_vline(x=0.4, line_dash='dash', line_color=RED,
                      annotation_text='Medium threshold')
        fig.update_layout(**BG, xaxis_title='Avg Risk Score',
                          yaxis_title='# Employees')
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Avg Risk Score by Department")
        dr = emp_risk_f.groupby('department')['avg_risk'].mean().reset_index()
        dr.columns = ['Department', 'Avg Risk']
        dr = dr.sort_values('Avg Risk')
        fig = px.bar(dr, x='Avg Risk', y='Department', orientation='h',
                     color='Avg Risk', color_continuous_scale='RdYlGn_r',
                     text=dr['Avg Risk'].apply(lambda v: f"{v:.3f}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(**BG, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### Avg Risk Score by Persona")
        pr = emp_risk_f.groupby('persona')['avg_risk'].mean().reset_index()
        pr.columns = ['Persona', 'Avg Risk']
        fig = px.bar(pr, x='Persona', y='Avg Risk',
                     color='Persona',
                     color_discrete_sequence=[CYAN, AMBER, RED, PURPLE],
                     text=pr['Avg Risk'].apply(lambda v: f"{v:.3f}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(**BG, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔴 Top 10 Highest-Risk Employees")
    top10 = (R['emp_risk']
             .sort_values('avg_risk', ascending=False)
             .head(10)[['employee_id', 'department', 'persona',
                         'avg_risk', 'anomaly_days', 'total_days', 'risk_tier']]
             .copy())
    top10['avg_risk']     = top10['avg_risk'].round(4)
    top10['anomaly_rate'] = (top10['anomaly_days'] / top10['total_days'] * 100).round(1).astype(str) + '%'
    st.dataframe(top10.rename(columns={
        'employee_id': 'Employee', 'department': 'Dept', 'persona': 'Persona',
        'avg_risk': 'Avg Risk', 'anomaly_days': 'Anomaly Days',
        'total_days': 'Total Days', 'risk_tier': 'Tier', 'anomaly_rate': 'Anomaly Rate'
    }), use_container_width=True, hide_index=True)


# ── TAB 5 — TIMELINE ─────────────────────────────────────────────
with tab5:
    df_time = R['df_out'].copy()
    df_time['date'] = pd.to_datetime(df_time['date'])
    df_time_f = df_time[
        df_time['department'].isin(dept_filter) &
        df_time['persona'].isin(persona_filter)
    ]

    daily = df_time_f.groupby('date').agg(
        anomaly_count=('is_anomaly', 'sum'),
        total_records=('is_anomaly', 'count'),
        avg_risk=('ensemble', 'mean'),
    ).reset_index()
    daily['anomaly_rate'] = daily['anomaly_count'] / daily['total_records'] * 100

    st.markdown("#### Daily Anomaly Count")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['anomaly_count'],
        mode='lines+markers', fill='tozeroy',
        line=dict(color=RED, width=2),
        fillcolor='rgba(232,72,85,0.12)',
        name='Anomaly Count'
    ))
    fig.add_hline(y=daily['anomaly_count'].mean(), line_dash='dash',
                  line_color=AMBER,
                  annotation_text=f"Mean = {daily['anomaly_count'].mean():.1f}")
    fig.update_layout(**BG, xaxis_title='Date', yaxis_title='Count', height=270)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Average Ensemble Risk Score Per Day")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['avg_risk'],
        mode='lines', fill='tozeroy',
        line=dict(color=CYAN, width=2),
        fillcolor='rgba(0,194,203,0.10)',
        name='Avg Risk'
    ))
    fig.add_hline(y=daily['avg_risk'].mean(), line_dash='dash',
                  line_color=AMBER,
                  annotation_text=f"Mean = {daily['avg_risk'].mean():.3f}")
    fig.update_layout(**BG, xaxis_title='Date',
                      yaxis_title='Avg Ensemble Score', height=260)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Anomaly Type Breakdown Over Time (Stacked)")
    daily_type = (df_time_f[df_time_f['is_anomaly'] == 1]
                  .groupby(['date', 'anomaly_type'])
                  .size().reset_index(name='count'))
    fig = px.bar(daily_type, x='date', y='count', color='anomaly_type',
                 color_discrete_map=ANOM_C, barmode='stack')
    fig.update_layout(**BG, xaxis_title='Date', yaxis_title='Count',
                      legend=dict(bgcolor='#0D1B2A'))
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 6 — PCA ──────────────────────────────────────────────────
with tab6:
    pca      = R['pca']
    X_pca    = R['X_pca']
    pc1      = pca.explained_variance_ratio_[0] * 100
    pc2      = pca.explained_variance_ratio_[1] * 100

    st.markdown(f"#### PCA 2D Projection  ·  PC1 = {pc1:.1f}%  ·  PC2 = {pc2:.1f}%  ·  Total = {pc1+pc2:.1f}% variance explained")

    pca_df = pd.DataFrame({
        'PC1':          X_pca[:, 0],
        'PC2':          X_pca[:, 1],
        'True Label':   R['y'].map({0: 'Normal', 1: 'Anomaly'}),
        'IF Label':     pd.Series(R['iso_labels']).map({0: 'Normal', 1: 'Anomaly'}),
        'Anomaly Type': R['df_out']['anomaly_type'].values,
    })

    sample = pca_df.sample(min(3000, len(pca_df)), random_state=42)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ground Truth**")
        fig = px.scatter(sample, x='PC1', y='PC2', color='True Label',
                         color_discrete_map={'Normal': CYAN, 'Anomaly': RED},
                         opacity=0.5)
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(**BG,
                          xaxis_title=f"PC1 ({pc1:.1f}% var)",
                          yaxis_title=f"PC2 ({pc2:.1f}% var)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Isolation Forest Predictions**")
        fig = px.scatter(sample, x='PC1', y='PC2', color='IF Label',
                         color_discrete_map={'Normal': GREEN, 'Anomaly': AMBER},
                         opacity=0.5)
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(**BG,
                          xaxis_title=f"PC1 ({pc1:.1f}% var)",
                          yaxis_title=f"PC2 ({pc2:.1f}% var)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Anomaly Type in PCA Space**")
    fig = px.scatter(sample, x='PC1', y='PC2', color='Anomaly Type',
                     color_discrete_map=ANOM_C, opacity=0.55)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(**BG, xaxis_title=f"PC1 ({pc1:.1f}% var)",
                      yaxis_title=f"PC2 ({pc2:.1f}% var)",
                      legend=dict(bgcolor='#0D1B2A'))
    st.plotly_chart(fig, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "BehaviourIQ · Open Track — Behavioural Analytics Hackathon · "
    "Pipeline: generate_dataset() → engineer_features() → IF + LOF + RF + PCA · "
    "All values computed live · Change any sidebar parameter to recompute everything."
)

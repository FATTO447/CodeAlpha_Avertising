import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
st.set_page_config(
    page_title="📊 Advertising Budget Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border-left: 5px solid #667eea;
        text-align: center;
        margin-bottom: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_assets():
    import os

    def find(names):
        for n in names:
            if os.path.exists(n):
                return n
        raise FileNotFoundError(f"Not found: {names}")

    lr = joblib.load(find([
        "linear_advertising_model.pkl",
        "/mnt/user-data/uploads/1776429244795_linear_advertising_model.pkl"
    ]))
    rf = joblib.load(find([
        "rf_advertising_model.pkl",
        "/mnt/user-data/uploads/1776429244796_rf_advertising_model.pkl"
    ]))
    df = pd.read_csv(find([
        "AD.csv",
        "/mnt/user-data/uploads/1776429232445_AD.csv"
    ]))

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    features = [
        'TV', 'Radio', 'Newspaper', 'Tv_radio_intecreation', 'Tv_Newspaper_interaction',
        'Radio_newspaper_interaction', 'TV_log', 'Radio_log', 'Newspaper_log',
        'TV_squared', 'Radio_squared'
    ]
    X = df[features]
    y = df['Sales']
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)

    metrics_data = {'LR': {'R2': 0.995}, 'RF': {'R2': 0.99}}
    return lr, rf, df, features, scaler, metrics_data

lr_model, rf_model, df_final, features_list, scaler, metrics = load_assets()
def prepare_user_data(tv, radio, news):
    tv, radio, news = float(tv), float(radio), float(news)
    input_data = {
        'TV': [tv], 'Radio': [radio], 'Newspaper': [news],
        'Tv_radio_intecreation': [tv * radio],
        'Tv_Newspaper_interaction': [tv * news],
        'Radio_newspaper_interaction': [radio * news],
        'TV_log': [np.log1p(tv)], 'Radio_log': [np.log1p(radio)], 'Newspaper_log': [np.log1p(news)],
        'TV_squared': [tv**2], 'Radio_squared': [radio**2]
    }
    return pd.DataFrame(input_data)[features_list]

with st.sidebar:
    st.markdown('## 🛠️ Global Settings')
    tv_val    = st.slider("TV Budget ($)",        0, 296, 150, key="sidebar_tv")
    radio_val = st.slider("Radio Budget ($)",      0,  49,  25, key="sidebar_radio")
    news_val  = st.slider("Newspaper Budget ($)",  0, 114,  20, key="sidebar_news")

    total_budget = tv_val + radio_val + news_val
    model_choice = st.radio("Predict Using:", ["Linear Regression", "Random Forest", "Both"], index=2, key="model_choice")

    st.markdown("---")
    st.markdown(f"**Total Budget: ${total_budget}**")

user_df = prepare_user_data(tv_val, radio_val, news_val)
pred_rf = float(rf_model.predict(user_df)[0])
pred_lr = float(lr_model.predict(scaler.transform(user_df))[0])

st.markdown("""<div class="hero"><h1>📊 Advertising Budget Optimizer</h1></div>""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if model_choice in ["Random Forest", "Both"]:
        st.markdown(f'<div class="metric-card"><h4>🌲 RF Forecast</h4><h2>{pred_rf:.2f}</h2><p>Units Sold</p></div>', unsafe_allow_html=True)
with col2:
    if model_choice in ["Linear Regression", "Both"]:
        st.markdown(f'<div class="metric-card"><h4>📈 LR Forecast</h4><h2>{pred_lr:.2f}</h2><p>Units Sold</p></div>', unsafe_allow_html=True)
with col3:
    if model_choice == "Both":
        avg = (pred_lr + pred_rf) / 2
        st.markdown(f'<div class="metric-card" style="border-left-color:#28a745"><h4>⚖️ Combined</h4><h2>{avg:.2f}</h2><p>Best Estimate</p></div>', unsafe_allow_html=True)

st.markdown("### 🔍 Insights & Decision Support")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💸 Current Allocation Analysis", "🔗 Market Synergy", "📉 Saturation Points",
    "🤖 Model Comparison", "⚡ AI Optimizer"
])

with tab1:
    st.markdown("#### 💸 Current Allocation Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Last 10 Records (Reference):")
        st.dataframe(df_final[['TV', 'Radio', 'Newspaper', 'Sales']].tail(10))
    with c2:
        fig_pie = px.pie(values=[tv_val, radio_val, news_val], names=['TV', 'Radio', 'News'],
                        hole=0.4, title="Your Current Allocation Mix")
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.markdown("#### 🔗 Market Synergy (TV × Radio)")
    fig_syn = px.scatter(df_final, x='Tv_radio_intecreation', y='Sales', color='Sales', size='Sales')
    st.plotly_chart(fig_syn, use_container_width=True)

with tab3:
    st.write("After a certain point, spending more gives less and less return.")
    c1, c2 = st.columns(2)
    with c1:
        try:
            fig_tv = px.scatter(
                df_final, x='TV_log', y='Sales',
                trendline='lowess', color='Sales',
                color_continuous_scale="Plasma",
                title="TV Saturation Curve",
                labels={"TV_log": "Log(TV Budget)"}
            )
        except Exception:
            fig_tv = px.scatter(
                df_final, x="TV_log", y="Sales",
                trendline="ols",
                title="TV Saturation Curve",
                labels={"TV_log": "Log(TV Budget)"}
            )
        if tv_val > 0:
            fig_tv.add_vline(
                x=np.log1p(tv_val), line_dash="dash",
                line_color="red", annotation_text=f"You: {tv_val}"
            )
        fig_tv.update_layout(title_x=0.5)
        st.plotly_chart(fig_tv, use_container_width=True)

    with c2:
        try:
            fig_rad = px.scatter(
                df_final, x='Radio_log', y='Sales',
                trendline='lowess', color='Sales',
                color_continuous_scale="Plasma",
                title="Radio Saturation Curve",
                labels={"Radio_log": "Log(Radio Budget)"}
            )
        except Exception:
            fig_rad = px.scatter(
                df_final, x="Radio_log", y="Sales",
                trendline="ols",
                title="Radio Saturation Curve",
                labels={"Radio_log": "Log(Radio Budget)"}
            )
        if radio_val > 0:
            fig_rad.add_vline(
                x=np.log1p(radio_val), line_dash="dash",
                line_color="red", annotation_text=f"You: {radio_val}"
            )
        fig_rad.update_layout(title_x=0.5)
        st.plotly_chart(fig_rad, use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    c1.metric("📈 Linear Regression R²", "99.5%", "MAE: 0.31 units")
    c2.metric("🌲 Random Forest R²",     "99.0%", "MAE: 0.43 units")

    fig_cmp = go.Figure()
    fig_cmp.add_bar(
        x=['Linear Regression', 'Random Forest'],
        y=[pred_lr, pred_rf],
        marker_color=["#667eea", "#16a34a"],
        text=[f"{pred_lr:.2f}", f"{pred_rf:.2f}"],
        textposition="outside"
    )
    fig_cmp.update_layout(
        title=f"Current Predictions — TV={tv_val}, Radio={radio_val}, News={news_val}",
        yaxis_title="Predicted Sales (Units)",
        title_x=0.5, showlegend=False,
        yaxis_range=[0, max(pred_lr, pred_rf) * 1.3]
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    importances = pd.Series(rf_model.feature_importances_, index=features_list).sort_values()
    fig_fi = px.bar(
        importances,
        x=importances.values,
        y=importances.index,
        orientation="h",
        title="Feature Importance (Random Forest)",
        color=importances.values,
        color_continuous_scale='Viridis',
        labels={"x": "Importance", "y": ""}
    )
    fig_fi.update_layout(title_x=0.5, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.info("💡 For final decisions, trust the **Random Forest** — it captures the TV×Radio synergy better.")

with tab5:
    st.markdown("#### ⚡ AI Optimization Engine")
    st.write(f"Finding best way to spend your **${total_budget}**...")
    if st.button("🚀 Run AI Optimization", key="opt_btn"):
        def goal(x):
            d = prepare_user_data(x[0], x[1], x[2])
            return -float(rf_model.predict(d)[0])

        init_guess = [total_budget / 3, total_budget / 3, total_budget / 3]
        bounds = ((0, 296), (0, 49), (0, 114))

        res = minimize(goal, init_guess, method='SLSQP', bounds=bounds,
                    constraints={'type': 'eq', 'fun': lambda x: sum(x) - total_budget})

        opt_sales = -res.fun
        improvement = ((opt_sales - pred_rf) / pred_rf * 100) if pred_rf > 0 else 0

        if improvement > 0.01:
            st.balloons()
            st.success(f"✅ AI found a better split! Potential Increase: **{improvement:.2f}%**")
        else:
            st.info("✨ Your current mix is already optimal.")

        comp_df = pd.DataFrame({
            'Channel': ['TV', 'Radio', 'News', 'TV', 'Radio', 'News'],
            'Amount': [tv_val, radio_val, news_val, res.x[0], res.x[1], res.x[2]],
            'Type': ['Current', 'Current', 'Current', 'AI Optimized', 'AI Optimized', 'AI Optimized']
        })
        fig_bar = px.bar(comp_df, x='Channel', y='Amount', color='Type', barmode='group')
        st.plotly_chart(fig_bar, use_container_width=True)
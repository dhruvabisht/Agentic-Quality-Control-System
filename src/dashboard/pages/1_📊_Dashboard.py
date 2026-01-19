"""
Dashboard Page - KPI Overview

Main analytics dashboard showing audit metrics, trends, and KPIs.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Dashboard | Sentinel-AI",
    page_icon="📊",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3d 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .big-metric {
        font-size: 3rem;
        font-weight: 700;
        color: #0866FF;
    }
    .metric-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard")
st.markdown("Real-time metrics and KPI tracking for content quality audits.")

# Time period selector
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    time_period = st.selectbox(
        "Time Period",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"],
        index=1
    )
with col2:
    if time_period == "Custom":
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
with col3:
    if time_period == "Custom":
        end_date = st.date_input("End Date", datetime.now())

st.markdown("---")

# Key Metrics Row
st.subheader("🎯 Key Performance Indicators")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        label="Total Audits",
        value="1,247",
        delta="+156 from last week",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="Pass Rate",
        value="87.3%",
        delta="+2.1%",
        delta_color="normal"
    )

with col3:
    st.metric(
        label="Fail Rate",
        value="8.2%",
        delta="-1.5%",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="Escalation Rate",
        value="4.5%",
        delta="-0.6%",
        delta_color="inverse"
    )

with col5:
    st.metric(
        label="Avg Confidence",
        value="84.7%",
        delta="+1.2%",
        delta_color="normal"
    )

with col6:
    st.metric(
        label="Accuracy",
        value="94.2%",
        delta="+0.5%",
        delta_color="normal"
    )

st.markdown("---")

# Charts Row
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Audit Volume Trend")
    
    # Generate sample trend data
    dates = [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(6, -1, -1)]
    
    trend_df = pd.DataFrame({
        "Date": dates,
        "Pass": [125, 142, 138, 156, 148, 167, 162],
        "Fail": [12, 15, 11, 18, 14, 21, 17],
        "Escalate": [8, 6, 9, 7, 11, 8, 10]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trend_df["Date"], y=trend_df["Pass"], name="Pass", marker_color="#31A24C"))
    fig.add_trace(go.Bar(x=trend_df["Date"], y=trend_df["Fail"], name="Fail", marker_color="#FA383E"))
    fig.add_trace(go.Bar(x=trend_df["Date"], y=trend_df["Escalate"], name="Escalate", marker_color="#F7B928"))
    
    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🥧 Violation Category Distribution")
    
    category_df = pd.DataFrame({
        "Category": ["Hate Speech", "Harassment", "Misinformation", "Spam", "Violence", "Other"],
        "Count": [45, 32, 28, 22, 15, 8]
    })
    
    fig = px.pie(
        category_df,
        values="Count",
        names="Category",
        color_discrete_sequence=["#FA383E", "#F7B928", "#0866FF", "#9333EA", "#31A24C", "#666"]
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Second Charts Row
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌐 Language Distribution")
    
    lang_df = pd.DataFrame({
        "Language": ["English", "Hindi", "Hinglish (Mixed)"],
        "Audits": [723, 412, 112],
        "Violations": [48, 42, 12]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Total Audits",
        x=lang_df["Language"],
        y=lang_df["Audits"],
        marker_color="#0866FF"
    ))
    fig.add_trace(go.Bar(
        name="Violations",
        x=lang_df["Language"],
        y=lang_df["Violations"],
        marker_color="#FA383E"
    ))
    
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("⏱️ Confidence Score Distribution")
    
    confidence_data = pd.DataFrame({
        "Score Range": ["90-100%", "80-89%", "70-79%", "60-69%", "<60%"],
        "Count": [456, 387, 245, 112, 47]
    })
    
    fig = px.bar(
        confidence_data,
        x="Score Range",
        y="Count",
        color="Count",
        color_continuous_scale=["#FA383E", "#F7B928", "#0866FF", "#31A24C"]
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Recent Activity Table
st.subheader("🕐 Recent Audit Activity")

activity_df = pd.DataFrame({
    "Time": ["2 min ago", "5 min ago", "8 min ago", "12 min ago", "15 min ago"],
    "Content ID": ["CNT-8847", "CNT-8846", "CNT-8845", "CNT-8844", "CNT-8843"],
    "Language": ["Hindi", "English", "Hinglish", "English", "Hindi"],
    "Verdict": ["✅ Pass", "❌ Fail", "⚠️ Escalate", "✅ Pass", "❌ Fail"],
    "Confidence": ["92.3%", "87.5%", "68.2%", "95.1%", "82.7%"],
    "Category": ["-", "Hate Speech", "Harassment", "-", "Misinformation"]
})

st.dataframe(
    activity_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Time": st.column_config.TextColumn("Time", width="small"),
        "Content ID": st.column_config.TextColumn("Content ID", width="medium"),
        "Language": st.column_config.TextColumn("Language", width="small"),
        "Verdict": st.column_config.TextColumn("Verdict", width="medium"),
        "Confidence": st.column_config.TextColumn("Confidence", width="small"),
        "Category": st.column_config.TextColumn("Violation Category", width="medium")
    }
)

# Footer
st.markdown("---")
st.caption("Data refreshes automatically every 60 seconds. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

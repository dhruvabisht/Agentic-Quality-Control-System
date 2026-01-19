"""
Market Insights Page

Generate and view reports on policy violation trends,
with special focus on Hindi-language content patterns.
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
    page_title="Market Insights | Sentinel-AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Market Insights")
st.markdown("Hindi-language content analysis and violation trend reports for the Indian market.")

# Report generation section
st.subheader("📊 Generate New Report")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    report_period = st.selectbox(
        "Report Period",
        ["Last 7 Days", "Last 14 Days", "Last 30 Days", "Custom Range"]
    )

with col2:
    if report_period == "Custom Range":
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        end_date = st.date_input("End Date", datetime.now())

with col3:
    st.write("")  # Spacer
    st.write("")
    if st.button("🔄 Generate Report", type="primary"):
        with st.spinner("Generating market insights report..."):
            import time
            time.sleep(2)  # Simulate processing
            st.session_state.report_generated = True
            st.success("✅ Report generated successfully!")

st.markdown("---")

# Report display
st.subheader("📋 Hindi Market Analysis Report")

# Executive Summary
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Hindi Audits", "524", "+18% vs last week")

with col2:
    st.metric("Hindi Violation Rate", "12.4%", "+2.1%", delta_color="inverse")

with col3:
    st.metric("Code-Mixed Content", "142", "27% of Hindi")

with col4:
    st.metric("Caste-Related Flags", "23", "+8", delta_color="inverse")

st.markdown("---")

# Visualization tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trends",
    "🗂️ Categories",
    "🔤 Language Patterns",
    "💡 Insights"
])

with tab1:
    st.subheader("Violation Trends in Hindi Content")
    
    # Generate trend data
    dates = [(datetime.now() - timedelta(days=x)).strftime("%m/%d") for x in range(6, -1, -1)]
    
    trend_df = pd.DataFrame({
        "Date": dates,
        "Hate Speech": [8, 12, 9, 15, 11, 18, 14],
        "Harassment": [5, 7, 6, 9, 8, 12, 10],
        "Misinformation": [4, 6, 5, 8, 6, 9, 7],
        "Caste-Related": [2, 3, 4, 5, 3, 6, 5]
    })
    
    fig = go.Figure()
    
    colors = ["#FA383E", "#F7B928", "#0866FF", "#9333EA"]
    for i, col in enumerate(["Hate Speech", "Harassment", "Misinformation", "Caste-Related"]):
        fig.add_trace(go.Scatter(
            x=trend_df["Date"],
            y=trend_df[col],
            name=col,
            mode="lines+markers",
            line=dict(color=colors[i], width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Week over week comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Week-over-Week Change")
        wow_df = pd.DataFrame({
            "Category": ["Hate Speech", "Harassment", "Misinformation", "Caste-Related"],
            "This Week": [77, 52, 45, 28],
            "Last Week": [65, 48, 41, 20],
            "Change %": ["+18.5%", "+8.3%", "+9.8%", "+40.0%"]
        })
        st.dataframe(wow_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Peak Violation Hours (IST)")
        hours_df = pd.DataFrame({
            "Time": ["18:00-21:00", "12:00-14:00", "21:00-23:00", "09:00-11:00"],
            "Violations": [89, 67, 54, 42],
            "% of Total": ["35.2%", "26.5%", "21.3%", "16.6%"]
        })
        st.dataframe(hours_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Violation Categories in Hindi Content")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category pie chart
        cat_df = pd.DataFrame({
            "Category": ["Hate Speech", "Harassment", "Misinformation", "Spam", "Violence", "Others"],
            "Count": [77, 52, 45, 32, 18, 12]
        })
        
        fig = px.pie(
            cat_df,
            values="Count",
            names="Category",
            color_discrete_sequence=["#FA383E", "#F7B928", "#0866FF", "#9333EA", "#31A24C", "#666"]
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity breakdown
        st.markdown("#### Severity Distribution")
        
        sev_df = pd.DataFrame({
            "Severity": ["Critical", "High", "Medium", "Low"],
            "Count": [28, 85, 98, 25],
            "Action": ["Immediate Removal", "Review & Remove", "Warning/Reduce", "Monitor"]
        })
        
        fig = px.bar(
            sev_df,
            x="Severity",
            y="Count",
            color="Severity",
            color_discrete_map={
                "Critical": "#FA383E",
                "High": "#F7B928",
                "Medium": "#0866FF",
                "Low": "#31A24C"
            }
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Language Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Content Language Distribution")
        
        lang_df = pd.DataFrame({
            "Type": ["Pure Hindi (Devanagari)", "Hinglish (Code-Mixed)", "Romanized Hindi"],
            "Count": [312, 142, 70],
            "Violation Rate": ["10.2%", "18.3%", "14.3%"]
        })
        
        fig = px.bar(
            lang_df,
            x="Type",
            y="Count",
            color="Type",
            color_discrete_sequence=["#0866FF", "#F7B928", "#31A24C"]
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("⚠️ Hinglish content has 80% higher violation rate than pure Hindi!")
    
    with col2:
        st.markdown("#### Top Hindi Slang/Terms Flagged")
        
        slang_df = pd.DataFrame({
            "Term": ["भंगी (caste slur)", "छोटी जात", "मार डालो", "कमीना", "निकालो", "हरामी"],
            "Count": [18, 15, 12, 28, 22, 19],
            "Category": ["Hate Speech", "Hate Speech", "Violence", "Harassment", "Hate Speech", "Harassment"]
        })
        
        st.dataframe(slang_df, use_container_width=True, hide_index=True)
        
        st.info("💡 Caste-related slurs account for 14% of all Hindi violations")

with tab4:
    st.subheader("💡 Key Insights & Recommendations")
    
    # Insight 1
    st.markdown("""
    <div style="background: rgba(250, 56, 62, 0.1); border-left: 4px solid #FA383E; 
                padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;">
        <h4 style="color: #FA383E; margin: 0;">🚨 Critical: Rising Caste-Based Content</h4>
        <p>Caste-related violations have increased by <strong>40%</strong> week-over-week. 
        This aligns with regional political events and requires immediate attention.</p>
        <p><strong>Recommendations:</strong></p>
        <ul>
            <li>Expand caste slur dictionary with regional variations</li>
            <li>Increase cultural sensitivity training for moderators</li>
            <li>Consider partnerships with Indian civil society organizations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Insight 2
    st.markdown("""
    <div style="background: rgba(247, 185, 40, 0.1); border-left: 4px solid #F7B928; 
                padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;">
        <h4 style="color: #F7B928; margin: 0;">⚠️ Warning: Hinglish Detection Gap</h4>
        <p>Code-mixed (Hinglish) content has an <strong>18.3% violation rate</strong> - 
        significantly higher than pure Hindi (10.2%). This suggests gaps in our detection.</p>
        <p><strong>Recommendations:</strong></p>
        <ul>
            <li>Enhance Hinglish pattern matching in Policy Agent</li>
            <li>Add more code-mixing examples to training data</li>
            <li>Consider dedicated Hinglish sentiment model</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Insight 3
    st.markdown("""
    <div style="background: rgba(8, 102, 255, 0.1); border-left: 4px solid #0866FF; 
                padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;">
        <h4 style="color: #0866FF; margin: 0;">ℹ️ Trend: Evening Peak Hours</h4>
        <p><strong>35.2%</strong> of violations occur between 6-9 PM IST, correlating with 
        peak social media usage in India.</p>
        <p><strong>Recommendations:</strong></p>
        <ul>
            <li>Increase automated moderation capacity during peak hours</li>
            <li>Schedule QMS shifts to align with violation peaks</li>
            <li>Consider real-time dashboards for peak hour monitoring</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Action items
    st.markdown("---")
    st.subheader("📋 Prioritized Action Items")
    
    actions = [
        {"priority": "🔴 High", "action": "Update caste slur dictionary with 15+ new regional terms", "owner": "Policy Team", "due": "3 days"},
        {"priority": "🔴 High", "action": "Review and retrain Hinglish detection model", "owner": "ML Team", "due": "1 week"},
        {"priority": "🟡 Medium", "action": "Implement peak hour monitoring dashboard", "owner": "Ops Team", "due": "2 weeks"},
        {"priority": "🟡 Medium", "action": "Schedule cultural sensitivity training", "owner": "HR", "due": "2 weeks"},
        {"priority": "🟢 Low", "action": "Document new slang patterns for Q2 review", "owner": "Research", "due": "1 month"}
    ]
    
    actions_df = pd.DataFrame(actions)
    st.dataframe(actions_df, use_container_width=True, hide_index=True)

# Export options
st.markdown("---")
st.subheader("📤 Export Report")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📄 Export as PDF", use_container_width=True):
        st.info("PDF generation in progress...")

with col2:
    if st.button("📊 Export as CSV", use_container_width=True):
        st.info("Preparing CSV download...")

with col3:
    if st.button("📧 Email Report", use_container_width=True):
        st.success("Report sent to stakeholders!")

# Footer
st.markdown("---")
st.caption(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data covers last 7 days")

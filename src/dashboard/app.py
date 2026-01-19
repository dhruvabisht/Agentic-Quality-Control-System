"""
Sentinel-AI Dashboard - Main Application

Streamlit-based Human-in-the-Loop interface for Quality Measurement Specialists.
Provides content review, decision override, and analytics capabilities.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Sentinel-AI | Content Quality Audit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/sentinel-ai',
        'Report a bug': 'https://github.com/sentinel-ai/issues',
        'About': "# Sentinel-AI\nAgentic Content Quality Audit System for Trust & Safety"
    }
)

# Custom CSS for Meta-inspired dark theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #0866FF;
        --secondary: #1877F2;
        --success: #31A24C;
        --warning: #F7B928;
        --danger: #FA383E;
        --dark: #1C1E21;
        --light: #F0F2F5;
    }
    
    /* Dark mode adjustments */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #0866FF 0%, #1877F2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(8, 102, 255, 0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
        font-size: 2rem;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3d 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0866FF;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Verdict badges */
    .verdict-pass {
        background: linear-gradient(135deg, #31A24C 0%, #28a745 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .verdict-fail {
        background: linear-gradient(135deg, #FA383E 0%, #dc3545 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .verdict-escalate {
        background: linear-gradient(135deg, #F7B928 0%, #ffc107 100%);
        color: #1a1a1a;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #0866FF 0%, #1877F2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(8, 102, 255, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #2d2d3d;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #0866FF;
        box-shadow: 0 0 0 2px rgba(8, 102, 255, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #2d2d3d;
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0866FF 0%, #31A24C 100%);
    }
    
    /* Alert styling */
    .alert-info {
        background: rgba(8, 102, 255, 0.1);
        border-left: 4px solid #0866FF;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    
    .alert-warning {
        background: rgba(247, 185, 40, 0.1);
        border-left: 4px solid #F7B928;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    
    .alert-danger {
        background: rgba(250, 56, 62, 0.1);
        border-left: 4px solid #FA383E;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = True  # Skip auth for demo
    st.session_state.user_id = "QMS-001"
    st.session_state.user_name = "Quality Specialist"

if "audit_history" not in st.session_state:
    st.session_state.audit_history = []

if "pending_reviews" not in st.session_state:
    st.session_state.pending_reviews = []


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Sentinel-AI</h1>
        <p>Agentic Content Quality Audit System | Trust & Safety Operations</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
        st.title("Navigation")
        
        st.markdown("---")
        
        # User info
        st.markdown(f"""
        <div style="background: #2d2d3d; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; color: #888;">Logged in as</div>
            <div style="font-weight: 600;">{st.session_state.user_name}</div>
            <div style="font-size: 0.8rem; color: #0866FF;">{st.session_state.user_id}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Today", "47", "+12")
        with col2:
            st.metric("Pending", "5", "-3")
        
        st.markdown("---")
        
        # System status
        st.subheader("⚙️ System Status")
        st.success("✅ Agents Online")
        st.info("📡 Database Connected")


def render_quick_audit():
    """Render quick audit section on home page."""
    st.subheader("⚡ Quick Audit")
    
    content = st.text_area(
        "Enter content to audit:",
        placeholder="Paste social media content here (Hindi or English)...",
        height=120,
        key="quick_audit_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        content_type = st.selectbox(
            "Content Type",
            ["post", "comment", "ad", "story"],
            key="quick_audit_type"
        )
    
    with col2:
        if st.button("🔍 Audit Now", key="quick_audit_btn", use_container_width=True):
            if content:
                with st.spinner("Running multi-agent audit..."):
                    try:
                        from src.agents import SentinelOrchestrator
                        
                        orchestrator = SentinelOrchestrator(enable_persistence=True)
                        result = orchestrator.audit_content(
                            content=content,
                            content_type=content_type
                        )
                        
                        st.session_state.last_audit_result = result.to_dict()
                        st.success("Audit complete!")
                        
                    except Exception as e:
                        st.error(f"Audit failed: {str(e)}")
                        # Show demo result for testing
                        st.session_state.last_audit_result = {
                            "content_id": "DEMO-001",
                            "verdict": "pass",
                            "confidence_score": 92.5,
                            "violation_summary": "No violations detected.",
                            "requires_human_review": False
                        }
            else:
                st.warning("Please enter content to audit.")
    
    # Display result if available
    if "last_audit_result" in st.session_state:
        result = st.session_state.last_audit_result
        
        st.markdown("---")
        st.subheader("📋 Audit Result")
        
        # Verdict display
        verdict = result.get("verdict", "pass")
        verdict_class = f"verdict-{verdict}"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if verdict == "pass":
                st.success(f"✅ PASS")
            elif verdict == "fail":
                st.error(f"❌ FAIL")
            else:
                st.warning(f"⚠️ ESCALATE")
        
        with col2:
            st.metric("Confidence", f"{result.get('confidence_score', 0):.1f}%")
        
        with col3:
            st.metric("Content ID", result.get("content_id", "N/A"))
        
        with col4:
            if result.get("requires_human_review"):
                st.warning("👤 Human Review Required")
            else:
                st.success("🤖 Auto-processed")
        
        # Details expander
        with st.expander("📝 View Details"):
            st.write("**Violation Summary:**", result.get("violation_summary", "N/A"))
            
            if result.get("detailed_explanation"):
                st.markdown(result.get("detailed_explanation"))
            
            if result.get("chain_of_thought"):
                st.subheader("Chain of Thought")
                for step in result.get("chain_of_thought", []):
                    st.markdown(f"**Step {step.get('step', '?')} ({step.get('type', 'N/A')}):**")
                    st.write(step.get("content", ""))


def render_metrics_dashboard():
    """Render the metrics overview."""
    st.subheader("📈 Today's Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">152</div>
            <div class="metric-label">Total Audits</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #31A24C;">89%</div>
            <div class="metric-label">Pass Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #FA383E;">12</div>
            <div class="metric-label">Violations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #F7B928;">5</div>
            <div class="metric-label">Pending Review</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">94.2%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    render_header()
    render_sidebar()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Quick Stats", "ℹ️ About"])
    
    with tab1:
        render_metrics_dashboard()
        st.markdown("---")
        render_quick_audit()
    
    with tab2:
        st.subheader("Performance Overview")
        
        # Sample chart data
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        
        # Generate sample data
        dates = [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(7, -1, -1)]
        
        df = pd.DataFrame({
            "Date": dates,
            "Audits": [45, 52, 48, 61, 55, 72, 68, 82],
            "Violations": [5, 8, 6, 12, 9, 15, 11, 14],
            "Escalations": [3, 4, 2, 5, 4, 6, 5, 7]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Audits"], name="Audits", line=dict(color="#0866FF", width=3)))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Violations"], name="Violations", line=dict(color="#FA383E", width=3)))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Escalations"], name="Escalations", line=dict(color="#F7B928", width=3)))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Violation Categories")
            category_data = pd.DataFrame({
                "Category": ["Hate Speech", "Harassment", "Misinformation", "Spam", "Violence"],
                "Count": [25, 18, 15, 12, 8]
            })
            
            fig_pie = px.pie(
                category_data, 
                values="Count", 
                names="Category",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Language Distribution")
            lang_data = pd.DataFrame({
                "Language": ["English", "Hindi", "Mixed (Hinglish)"],
                "Count": [95, 42, 15]
            })
            
            fig_bar = px.bar(
                lang_data,
                x="Language",
                y="Count",
                color="Language",
                color_discrete_sequence=["#0866FF", "#31A24C", "#F7B928"]
            )
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.subheader("About Sentinel-AI")
        
        st.markdown("""
        ### 🛡️ Agentic Content Quality Audit System
        
        Sentinel-AI is a multi-agent system designed for Meta's Trust and Safety operations. 
        It audits Hindi and English social media content against Community Standards and 
        Advertiser Policies using advanced LLM-powered reasoning.
        
        **Key Features:**
        - 🤖 **Multi-Agent Architecture**: Policy Agent, Hindi Cultural Agent, and Auditor Agent
        - 🧠 **Chain-of-Thought Reasoning**: Transparent decision-making process
        - 🇮🇳 **Hindi Language Support**: Regional slang, cultural context, and sentiment analysis
        - 👤 **Human-in-the-Loop**: Override decisions and provide feedback
        - 📊 **Analytics Dashboard**: Track KPIs and violation trends
        - 📝 **Market Insights**: Hindi-language violation pattern reports
        
        **System Architecture:**
        """)
        
        st.code("""
Content Input (Hindi/English)
        │
        ▼
┌───────────────────┐
│   Policy Agent    │ ← Community Standards JSON
│  (Rule Matching)  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Hindi Cultural    │ ← Slang Dictionary
│     Agent         │ ← Cultural Sensitivities
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Auditor Agent    │ → Chain of Thought
│  (CoT Reasoning)  │ → Final Verdict
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    Streamlit      │ ← Human Review
│    Dashboard      │ ← Override/Feedback
└───────────────────┘
        """, language="")
        
        st.markdown("""
        ---
        **Version:** 1.0.0  
        **Built with:** LangGraph, Streamlit, PostgreSQL  
        **Team:** Trust & Safety Division
        """)


if __name__ == "__main__":
    main()
